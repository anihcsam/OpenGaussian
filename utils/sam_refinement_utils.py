# Multi-view SAM mask refinement imports and utilities
from collections import defaultdict
import torch
from tqdm import tqdm
import math
import numpy as np
from gaussian_renderer import render
from scene.cameras import Camera

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

import rerun as rr
import cv2
import torch.nn.functional as F

def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=-1
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)

def log_camera_pose(
    log_name: str,
    translation: np.ndarray,
    rotation_q: np.ndarray,
    intrinsics: np.ndarray,
    frame_width: int = 518,
    frame_height: int = 294,
    image = None,
    translation_prev: np.ndarray = None,
    line_label: str = "",
):
    """
    Log camera (pose + intrinsics) into rerun viewer.

    Args:
        log_name - rerun space to append camera pose to (e.g. same as pcl space name)
        translation - 1x3 translation vector
        rotation_q - 1x4 rotation vector (quaternion)
        intrinsics - 3x3 K matrix
        frame_width, frame_height - image res
        translation_prev - translation vector of prev camera pose
        line_label - label to assign to a line segment connecting two consequtive poses
    """

    # Camera pose
    rr.log(
        f"{log_name}/camera_pose",
        rr.Transform3D(
            translation=translation,
            rotation=rr.Quaternion(xyzw=rotation_q),
            relation=rr.TransformRelation.ParentFromChild,
        ),
    )

    # Camera model
    rr.log(
        f"{log_name}/camera_pose/image",
        rr.Pinhole(
            resolution=[frame_width, frame_height],
            focal_length=[intrinsics[0, 0], intrinsics[1, 1]],
            principal_point=[intrinsics[0, 2], intrinsics[1, 2]],
        ),
    )
    if image is not None:
        rr.log(
            f"{log_name}/camera_pose/image",
            rr.Image(image).compress(jpeg_quality=75),
        )
    rr.log(
                f"{log_name}/camera_pose/coords",
                rr.Arrows3D(
                    vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                ),
    )

class MultiViewSAMMaskRefiner:
    """Refines SAM masks by enforcing consistency across overlapping views"""
    
    def __init__(self, overlap_threshold=0.3, consensus_strategy="majority_vote", log_to_rerun=False, visualize_matches=False):
        self.overlap_threshold = overlap_threshold
        self.consensus_strategy = consensus_strategy
        self.log_to_rerun = log_to_rerun
        self.visualize_matches = visualize_matches
        if self.log_to_rerun:
            print("MultiViewSAMMaskRefiner initialized with logging to rerun enabled")
        if self.visualize_matches:
            print("MultiViewSAMMaskRefiner initialized with enabled visualization for gaussian matches")
        
    def find_overlapping_cameras(self, cameras):
        """Find pairs of cameras with overlapping views using frustum intersection"""
        overlapping_pairs = []
        
        for i, cam1 in enumerate(cameras):
            for j, cam2 in enumerate(cameras[i+1:], i+1):
                if self._cameras_overlap(cam1, cam2):
                    overlapping_pairs.append((i, j))
                    
        return overlapping_pairs
    
    def _cameras_overlap(self, cam1, cam2):
        """Check if two cameras have overlapping views using position, orientation, and FoV"""
        # Extract camera positions and viewing directions
        pos1 = cam1.camera_center
        pos2 = cam2.camera_center
        
        # Calculate distance between cameras
        distance = torch.norm(pos1 - pos2)
        
        # If cameras are too far apart, they likely don't overlap
        max_distance = 3.0
        if distance > max_distance:
            return False
            
        # Extract viewing directions from world_view_transform
        # The viewing direction is the negative Z-axis in camera space, transformed to world space
        # world_view_transform is world-to-camera, so we need its inverse for camera-to-world
        world_view_inv1 = torch.inverse(cam1.world_view_transform)
        world_view_inv2 = torch.inverse(cam2.world_view_transform)
        
        # Camera forward direction (negative Z in camera space)
        forward_cam = torch.tensor([0., 0., -1., 0.], device=pos1.device)
        
        # Transform to world space (only rotation part, so w=0)
        forward1_world = (world_view_inv1 @ forward_cam)[:3]
        forward2_world = (world_view_inv2 @ forward_cam)[:3]
        
        # Normalize directions
        forward1_world = forward1_world / torch.norm(forward1_world)
        forward2_world = forward2_world / torch.norm(forward2_world)
        
        # Calculate angle between viewing directions
        dot_product = torch.dot(forward1_world, forward2_world)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Numerical stability
        angle_between_directions = torch.acos(dot_product)
        
        # If cameras are facing nearly opposite directions, they likely don't overlap
        max_direction_angle = math.pi * 0.75  # 135 degrees
        if angle_between_directions > max_direction_angle:
            return False
            
        # Check if cameras can potentially see each other's view based on FoV
        # Calculate vector from cam1 to cam2
        cam1_to_cam2 = pos2 - pos1
        cam1_to_cam2_normalized = cam1_to_cam2 / torch.norm(cam1_to_cam2)
        
        # Calculate vector from cam2 to cam1  
        cam2_to_cam1 = pos1 - pos2
        cam2_to_cam1_normalized = cam2_to_cam1 / torch.norm(cam2_to_cam1)
        
        # Check if cam2 is within cam1's field of view
        angle_cam1_to_cam2 = torch.acos(torch.clamp(torch.dot(forward1_world, cam1_to_cam2_normalized), -1.0, 1.0))
        angle_cam2_to_cam1 = torch.acos(torch.clamp(torch.dot(forward2_world, cam2_to_cam1_normalized), -1.0, 1.0))
        
        # Use larger FoV (horizontal or vertical) with some margin
        fov1_max = max(cam1.FoVx, cam1.FoVy)
        fov2_max = max(cam2.FoVx, cam2.FoVy)
        
        # Add margin to account for overlap at edges
        fov_margin = np.pi / 6  # 30 degrees margin
        
        # Check if there's potential overlap based on viewing angles and FoVs
        cam1_sees_cam2_direction = angle_cam1_to_cam2 < (fov1_max / 2 + fov_margin)
        cam2_sees_cam1_direction = angle_cam2_to_cam1 < (fov2_max / 2 + fov_margin)
        
        # Cameras overlap if they're reasonably close, not facing opposite directions,
        # and at least one can potentially see in the direction of the other
        overlap = cam1_sees_cam2_direction or cam2_sees_cam1_direction
        
        return overlap
    
    def project_3d_points_to_image(self, point_3d, camera: Camera, index_gs = 0, index_cam = 0):
        """
        Project 3D point in world coordinates into 2D image pixel coordinates.
        
        Parameters:
        - points_3d (torch.Tensor): A tensor of shape (3) representing point in world coordinates.
        - world_to_camera (torch.Tensor): A 4x4 transformation matrix mapping world coordinates to camera coordinates.
        - camera_to_pixel (torch.Tensor): A 4x4 transformation matrix mapping camera coordinates to pixel coordinates.
        
        Returns:
        - u,v.
        - flag visible or not
        """
        
        # Convert points to homogeneous coordinates (add a fourth '1' coordinate)
        point_3d_homogeneous = torch.cat([point_3d, torch.tensor([1.0], device=point_3d.device)])

        # Get coordinate of queried point in camera space (direct transform)
        point_camera = camera.world_view_transform_no_t @ point_3d_homogeneous
        if self.log_to_rerun:
            rr.log(f"gs_{index_gs}/camera_{index_cam}/camera_pose/gs_in_cam", rr.Points3D(point_camera[:3], radii=0.01, colors=[0, 0, 255]))

        # Check if points are in front of the camera (z > 0 in camera space)
        is_in_front = point_camera[2] > 0

        # Apply projection matrix to map to NDC
        point_clip = camera.projection_matrix_no_t @ point_camera  # Shape: (4)

        # 2. Perspective division
        w = point_clip[3]
        point_ndc = point_clip / w

        # 3. Viewport transformation
        u = point_ndc[0] * (camera.image_width / 2.0) + camera.cx
        v = point_ndc[1] * (camera.image_height / 2.0) + camera.cy

        return int(u), int(v), (0 <= u < camera.image_width and 0 <= v < camera.image_height and is_in_front)
    
    def collect_mask_votes_for_point(self, point_3d, cameras, sam_masks, sam_level=0):
        """Collect mask ID votes for a 3D point from all visible cameras"""
        votes = []
        
        for cam_idx, camera in enumerate(cameras):
            x, y, is_visible = self.project_3d_points_to_image(point_3d, camera)
            
            if is_visible and x is not None and y is not None:
                # Get SAM mask at this pixel
                sam_mask = sam_masks[cam_idx]  # [4, H, W]
                if sam_mask is not None:
                    mask_id = sam_mask[sam_level, y, x].item()
                    votes.append((cam_idx, mask_id))
                    
        return votes
    
    def apply_consensus_rule(self, votes):
        """Apply consensus rule to resolve mask ID conflicts"""
        if not votes:
            return -1  # Invalid/no consensus
            
        if self.consensus_strategy == "majority_vote":
            # Count votes for each mask ID
            vote_counts = defaultdict(int)
            for _, mask_id in votes:
                if mask_id >= 0:  # Only count valid mask IDs
                    vote_counts[mask_id] += 1
                    
            if vote_counts:
                # Return the mask ID with most votes
                return max(vote_counts, key=vote_counts.get)
                
        return -1  # No consensus
    
    def debug_visualize_projections(self, gaussian_3d, votes, consensus_id, cameras, sam_masks, sam_level=0, 
                            current_cam_idx=None, max_pairs=None, gaussian_idx=None):
        """Debug visualization showing ALL images where the Gaussian projects with valid votes"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        
        if len(votes) < 2:
            return  # Need at least 2 votes for visualization
        
        # Filter out votes with mask ID = -1 (irrelevant points)
        valid_votes = [(cam_idx, mask_id) for cam_idx, mask_id in votes if mask_id != -1]
        if len(valid_votes) < 2:
            return  # Need at least 2 valid votes for visualization
        
        # Get unique camera indices from valid votes
        voting_cameras = list(set([cam_idx for cam_idx, _ in valid_votes]))
        
        # If current camera is provided, add it to the list
        if current_cam_idx is not None and current_cam_idx not in voting_cameras:
            voting_cameras.append(current_cam_idx)
        
        # Sort cameras for consistent visualization
        voting_cameras.sort()
    
        # Calculate grid layout for subplots
        num_cameras = len(voting_cameras)
        if num_cameras <= 4:
            rows, cols = 1, num_cameras
        elif num_cameras <= 8:
            rows, cols = 2, 4
        elif num_cameras <= 12:
            rows, cols = 3, 4
        else:
            rows, cols = 4, 4  # Maximum 16 cameras
            voting_cameras = voting_cameras[:16]  # Limit to 16 for readability
        
        # Create figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        
        # Handle axes properly - ensure axes is always a 2D array
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Project 3D point to all cameras and visualize
        for idx, cam_idx in enumerate(voting_cameras):
            row = idx // cols
            col = idx % cols
            
            # Get the correct axis
            ax = axes[row, col]
            
            camera = cameras[cam_idx]
            
            # Project 3D point to this camera
            x, y, visible = self.project_3d_points_to_image(gaussian_3d, camera)
            
            # Get mask ID at projection point (with bounds checking)
            mask_id = -1
            if visible and x is not None and y is not None:
                if 0 <= y < sam_masks[cam_idx].shape[1] and 0 <= x < sam_masks[cam_idx].shape[2]:
                    mask_id = sam_masks[cam_idx][sam_level, y, x].item()
            
            # Display camera image
            if hasattr(camera, 'original_image') and camera.original_image is not None:
                img = camera.original_sam_mask.cpu().numpy().transpose(1, 2, 0)[:, :, 0]
                ax.imshow(img)
            else:
                # Create blank image if no original image
                ax.imshow(np.ones((camera.image_height, camera.image_width, 3)) * 0.5)
            
            # Overlay projected point (only if mask ID != -1)
            if visible and x is not None and mask_id != -1:
                circle = patches.Circle((x, y), radius=8, color='red', fill=False, linewidth=3)
                ax.add_patch(circle)
                # Show mask ID on image
                ax.text(x+15, y-15, f'ID: {mask_id}', color='red', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Add title with camera info
            vote_info = f"Vote: {mask_id}" if mask_id != -1 else "No vote"
            consensus_info = f"→ {consensus_id}" if mask_id != -1 else ""
            ax.set_title(f'Cam {cam_idx}\n{vote_info} {consensus_info}', fontsize=12)
            
            ax.axis('off')
            
            # Print projection details to console
            print(f"   📷 Cam{cam_idx}: projected=({x}, {y}), visible={visible}, mask_id={mask_id}")
        
        # Hide unused subplots
        for idx in range(len(voting_cameras), rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        # Add overall title with voting info
        vote_info = ', '.join([f'Cam{cam_idx}:ID{mask_id}' for cam_idx, mask_id in valid_votes])
        fig.suptitle(f'3D Point Projection Debug - Gaussian #{gaussian_idx}\n'
                    f'Votes: [{vote_info}] → Consensus: {consensus_id}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Wait for user input before continuing
        input("Press Enter to continue...")
        plt.close(fig)
    
    def refine_sam_masks(self, cameras, sam_masks, gaussians, sam_level=0):
        """Refine SAM masks using multi-view consistency with efficient overlap detection"""
        
        # Pre-compute overlapping camera pairs for efficiency
        print("Computing camera overlaps...")
        overlapping_pairs = self.find_overlapping_cameras(cameras)
        
        # Build adjacency map for quick lookup
        overlap_map = {}
        for i in range(len(cameras)):
            overlap_map[i] = set()
        
        for cam1_idx, cam2_idx in overlapping_pairs:
            overlap_map[cam1_idx].add(cam2_idx)
            overlap_map[cam2_idx].add(cam1_idx)
            
        print(f"Found {len(overlapping_pairs)} overlapping camera pairs")
        
        refined_masks = []
        
        for cam_idx, camera in tqdm(enumerate(cameras), total=len(cameras), desc="Refining masks"):
            if sam_masks[cam_idx] is None:
                refined_masks.append(None)
                continue
                
            original_mask = sam_masks[cam_idx].clone()
            refined_mask = original_mask.clone()
            
            # Get cameras that overlap with current camera
            overlapping_cam_indices = overlap_map[cam_idx]
            
            if not overlapping_cam_indices:
                # No overlaps, keep original mask
                refined_masks.append(refined_mask)
                continue
            
            # Sample subset of Gaussians for efficiency (every 10th Gaussian)
            sample_step = 10
            num_gaussians = gaussians.get_xyz.shape[0]
            
            if self.log_to_rerun:
                rr.init("sam_refinement",spawn=True)
                rr.log(
                    "world_frame",
                    rr.Arrows3D(
                        vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                    ),
                )
                rr.log(f"gaussian_pointcloud", rr.Points3D(gaussians.get_xyz.cpu(), radii=0.005, colors=[0, 255, 0]))
            
            # For each sampled Gaussian, collect votes from overlapping cameras only
            for gaussian_idx in range(0, num_gaussians, sample_step):
                gaussian_3d = gaussians.get_xyz[gaussian_idx]  # Actual 3D position
                
                if self.log_to_rerun:
                    gaussian_3d_cpu = gaussian_3d.cpu()
                    rr.log(f"gs_{gaussian_idx}", rr.Points3D(gaussian_3d_cpu, radii=0.01, colors=[255, 0, 0]))
                
                # Project this Gaussian only to overlapping cameras
                votes = []
                for i, other_cam_idx in enumerate(overlapping_cam_indices):
                    other_camera = cameras[other_cam_idx]
                    x, y, visible = self.project_3d_points_to_image(gaussian_3d, other_camera, gaussian_idx, other_cam_idx)
                    if self.log_to_rerun:
                        print(f"Camera {other_cam_idx} visibility: {visible}, x: {x}, y: {y}")
                        
                        image = other_camera.original_image.cpu().numpy().transpose(1, 2, 0)
                        if image.dtype != np.uint8:
                            image = np.clip(image * 255, 0, 255).astype(np.uint8)
                    
                    if visible and x is not None and y is not None:
                        mask_id = sam_masks[other_cam_idx][sam_level, y, x].item() # x y or y x???
                        votes.append((other_cam_idx, mask_id))
                        
                        if self.log_to_rerun:
                            cv2.circle(image, (int(x), int(y)), radius=5, color=(255, 0, 0), thickness=-1)
                    
                    if self.log_to_rerun:
                        # Transform from world frame to camera frame (inverse transform)
                        world2cam = np.linalg.inv(other_camera.world_view_transform_no_t.cpu().numpy())
                        t = world2cam[:3, 3]
                        R = world2cam[:3, :3]
                        rot_q = mat_to_quat(torch.from_numpy(R).unsqueeze(0)).squeeze(0).numpy()
                        K = other_camera.intrinsic_matrix.cpu().numpy()
                        log_camera_pose(
                            f"gs_{gaussian_idx}/camera_{other_cam_idx}",
                            t,
                            np.array([rot_q[0], rot_q[1], rot_q[2], rot_q[3]]),
                            K,
                            other_camera.image_width,
                            other_camera.image_height,
                            image=image,
                        )
                        
                # Apply consensus and update the current camera's mask
                if votes:
                    consensus_id = self.apply_consensus_rule(votes)
                    # Project Gaussian to current camera and update mask in local neighborhood
                    x_curr, y_curr, visible_curr = self.project_3d_points_to_image(gaussian_3d, camera)
                    
                    if visible_curr and consensus_id >= 0 and x_curr is not None and y_curr is not None:
                        # Update a small neighborhood around the projected point
                        H, W = refined_mask.shape[1], refined_mask.shape[2]
                        changes_made = 0  # Track changes for this Gaussian
    
                        if 0 <= y_curr < H and 0 <= x_curr < W:
                            if refined_mask[sam_level, y_curr, x_curr] != consensus_id:
                                refined_mask[sam_level, y_curr, x_curr] = consensus_id
                                changes_made += 1
                        
                        # Call debug visualization if changes were made
                        if self.visualize_matches and changes_made > 0:
                            self.debug_visualize_projections(
                                gaussian_3d, votes, consensus_id, cameras, sam_masks, 
                                sam_level, current_cam_idx=cam_idx, max_pairs=2, gaussian_idx=gaussian_idx
                            )                        
                if self.log_to_rerun:
                    input("Pause: press a key to continue")  # Pause AFTER FIRST GS

            refined_masks.append(refined_mask)
        
        return refined_masks