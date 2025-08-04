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

import matplotlib.pyplot as plt
import numpy as np
import time


def create_consistent_id_mapping(masks: list[torch.Tensor]):
    """
    Create a consistent mapping from original IDs to consecutive IDs (1, 2, 3, ...) 
    across all masks.
    
    Args:
        masks: List of masks, each with shape [H, W] containing segment IDs
        
    Returns:
        tuple: (id_mapping dict, remapped_masks list)
    """
    # Collect all unique IDs across all masks
    all_unique_ids = set()
    
    for mask in masks:
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                unique_ids = torch.unique(mask)
                # Convert tensor values to Python ints for set operations
                all_unique_ids.update(unique_ids.cpu().tolist())
    
    # Remove invalid IDs (like -1, 0) if needed
    all_unique_ids = {id_val for id_val in all_unique_ids if id_val > 0}
    
    # Sort IDs and create mapping to consecutive numbers
    sorted_ids = sorted(all_unique_ids)
    id_mapping = {}
    
    # Map sorted IDs to 1, 2, 3, ...
    for new_id, old_id in enumerate(sorted_ids, 1):
        id_mapping[old_id] = new_id
    
    # Add mapping for invalid IDs (preserve them)
    id_mapping[0] = 0
    id_mapping[-1] = -1
    
    print(f"Created mapping for {len(sorted_ids)} unique IDs")
    # print(f"ID range: {min(sorted_ids) if sorted_ids else 'N/A'} -> {max(sorted_ids) if sorted_ids else 'N/A'}")
    # print(f"Remapped to: 1 -> {len(sorted_ids)}")
    
    # Apply mapping to all masks
    remapped_masks = []
    for mask in masks:
        if mask is None:
            remapped_masks.append(None)
            continue
            
        # Create remapped mask using tensor operations
        if isinstance(mask, torch.Tensor):
            remapped_mask = torch.zeros_like(mask)
            for old_id, new_id in id_mapping.items():
                remapped_mask[mask == old_id] = new_id
                
        remapped_masks.append(remapped_mask)
    
    return id_mapping, remapped_masks

def save_weight_map_to_csv(weight_map: torch.Tensor, filename="weight_map.csv"):
    """
    Save a weight map to a CSV file.
    
    Args:
        weight_map: Weight map with shape [H, W, 1] or [H, W] as torch.Tensor
        filename: Output CSV filename
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(weight_map, torch.Tensor):
        weight_map = weight_map.detach().cpu().numpy()
    
    # If weight_map has shape [H, W, 1], squeeze to [H, W]
    if weight_map.ndim == 3 and weight_map.shape[2] == 1:
        weight_map = np.squeeze(weight_map, axis=2)
    
    # Save to CSV without headers or row indices
    np.savetxt(filename, weight_map, delimiter=",", fmt="%.6f")
    
    print(f"Saved weight map of shape {weight_map.shape} to {filename}")

def rgb_to_weight_map(rendered_image: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB splat render to a weight map (1 for white, 0 for black),
    normalized so the maximum value is exactly 1.0.
    
    Args:
        rendered_image: RGB image with shape [H, W, 3]
        
    Returns:
        weight_map: Single channel weight map with shape [H, W, 1] as torch.Tensor
    """
    # Convert to torch tensor if it's numpy
    # if isinstance(rendered_image, np.ndarray):
    #     rendered_image = torch.from_numpy(rendered_image).float()
    #     # Move to GPU if available
    #     if torch.cuda.is_available():
    #         rendered_image = rendered_image.cuda()
    
    # Ensure it's on the same device as the input if it was already a tensor
    original_device = rendered_image.device
    
    # Convert to float if it's uint8
    if rendered_image.dtype == torch.uint8:
        rendered_image = rendered_image.float() / 255.0
    elif rendered_image.max() > 1.0:
        rendered_image = rendered_image / 255.0
    
    # Method 2: Average RGB channels
    weight_map = torch.mean(rendered_image, dim=2)
    
    # Normalize by maximum value to ensure max is exactly 1.0
    max_val = weight_map.max()
    if max_val > 0:  # Avoid division by zero
        weight_map = weight_map / max_val
    
    # Add singleton dimension to get [H, W, 1] and ensure it's on the same device
    weight_map = weight_map.unsqueeze(2).to(original_device)
    
    return weight_map

def save_images_and_difference(img1, img2, x, y, filename="comparison.png", title1="Image all", title2="Image exclude"):
    # Pixelwise absolute difference
    diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16)).astype(np.uint8)

    # Add titles as text on images (optional)
    def add_title(image, title):
        img = image.copy()
        cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        return img

    img1 = add_title(img1, title1)
    img2 = add_title(img2, title2)
    diff = add_title(diff, "Pixelwise Difference")
    
    # Add projected point to first image
    cv2.circle(img1, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
    
    # Amplify diff
    diff *= 100

    # Concatenate images horizontally
    combined = np.concatenate([img1, img2, diff], axis=1)
    cv2.imwrite(filename, combined)
    
    threshold = 300
    diff_thr = (diff > threshold).any(axis=2)
    num_non_black_pixels = np.count_nonzero(diff_thr)
    # print(f"non-black pixels: {num_non_black_pixels}")
    NUM_VISIBLE_PIXELS_THRESHOLD = 20
    # if num_non_black_pixels > NUM_VISIBLE_PIXELS_THRESHOLD:
    #     return True
    return False
    
# NOTE: this is numpy version, replaced with torch version (below)
# def fix_image(rendered_image):
#     if not isinstance(rendered_image, np.ndarray):
#         rendered_image = rendered_image.detach().cpu().numpy()
#         rendered_image = np.transpose(rendered_image, (1, 2, 0))
#     if rendered_image.dtype != np.uint8:
#         rendered_image = np.clip(rendered_image * 255, 0, 255).astype(np.uint8)
#     if rendered_image.ndim == 2:
#         rendered_image = np.stack([rendered_image]*3, axis=-1)  # Make grayscale 3-channel
#     if rendered_image.shape[2] == 1:
#         rendered_image = np.repeat(rendered_image, 3, axis=2)
#     rendered_image = np.ascontiguousarray(rendered_image)
    
#     return rendered_image

def fix_image(rendered_image: torch.Tensor) -> torch.Tensor:
    """
    Fix image format - tensor version
    
    Args:
        rendered_image: torch.Tensor in various formats
        
    Returns:
        torch.Tensor: Fixed image in [H, W, 3] format with uint8 dtype
    """
    # Convert to tensor if it's numpy
    # if isinstance(rendered_image, np.ndarray):
    #     rendered_image = torch.from_numpy(rendered_image).float()
    
    # If tensor is [3, H, W], transpose to [H, W, 3]
    if rendered_image.ndim == 3 and rendered_image.shape[0] == 3:
        rendered_image = rendered_image.permute(1, 2, 0)
    
    # Normalize to 0-255 range and convert to uint8
    if rendered_image.dtype != torch.uint8:
        rendered_image = torch.clamp(rendered_image * 255, 0, 255).to(torch.uint8)
    
    # Handle grayscale [H, W] -> [H, W, 3]
    if rendered_image.ndim == 2:
        rendered_image = rendered_image.unsqueeze(2).repeat(1, 1, 3)
    
    # Handle single channel [H, W, 1] -> [H, W, 3]
    if rendered_image.shape[2] == 1:
        rendered_image = rendered_image.repeat(1, 1, 3)
    
    # Ensure contiguous memory layout
    rendered_image = rendered_image.contiguous()
    
    return rendered_image

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

from scene.gaussian_model import GaussianModel
from ashawkey_diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.sh_utils import eval_sh
def render_single_gaussian(viewpoint_camera, pc: GaussianModel, gaussian_idx: int, scaling_modifier=1.0, use_view_inv_white_shs: bool = False, **kwargs):
    """
    Render only a single Gaussian splat at the specified index from the model.
    """
    import math
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Select only the single Gaussian at gaussian_idx
    means3D = pc.get_xyz[gaussian_idx:gaussian_idx+1]
    means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
    try:
        means2D.retain_grad()
    except Exception:
        pass

    opacity = pc.get_opacity[gaussian_idx:gaussian_idx+1]
    scales = pc.get_scaling[gaussian_idx:gaussian_idx+1]
    rotations = pc.get_rotation[gaussian_idx:gaussian_idx+1]
    shs = pc.get_features[gaussian_idx:gaussian_idx+1]

    # If you have precomputed colors, handle here (optional)
    colors_precomp = None
    if kwargs.get("override_color") is not None:
        colors_precomp = kwargs["override_color"]
        if colors_precomp.ndim == 2:
            colors_precomp = colors_precomp[gaussian_idx:gaussian_idx+1]
        elif colors_precomp.ndim == 1:
            colors_precomp = colors_precomp.unsqueeze(0)
    features_dc = torch.ones((1, 1, 3), dtype=torch.float32, device=shs.device)
    features_rest = torch.zeros((1, 15, 3), dtype=torch.float32, device=shs.device)
    shs_white_dir_inv = torch.cat((features_dc, features_rest), dim=1)
    
    if use_view_inv_white_shs:
        shs = shs_white_dir_inv

    # Rasterize only this Gaussian
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs if colors_precomp is None else None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None
    )
    return rendered_image, radii, rendered_depth, rendered_alpha

def render_gaussians_with_exclusion(viewpoint_camera, pc: GaussianModel, exclude_indices=None, scaling_modifier=1.0, **kwargs):
    """
    Render all Gaussians except those specified in exclude_indices.
    If exclude_indices is empty or None, render all Gaussians.

    Args:
        viewpoint_camera: Camera object.
        pc (GaussianModel): Gaussian model.
        exclude_indices (list or None): Indices to exclude from rendering.
        scaling_modifier (float): Scaling modifier for rendering.
        kwargs: Additional arguments.

    Returns:
        rendered_image, radii, rendered_depth, rendered_alpha
    """
    import math
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Determine indices to render
    all_indices = list(range(pc.get_xyz.shape[0]))
    if exclude_indices:
        render_indices = [i for i in all_indices if i not in exclude_indices]
    else:
        render_indices = all_indices

    # Select Gaussians to render
    means3D = pc.get_xyz[render_indices]
    means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
    try:
        means2D.retain_grad()
    except Exception:
        pass

    opacity = pc.get_opacity[render_indices]
    scales = pc.get_scaling[render_indices]
    rotations = pc.get_rotation[render_indices]
    shs = pc.get_features[render_indices]

    colors_precomp = None
    if kwargs.get("override_color") is not None:
        colors_precomp = kwargs["override_color"]
        if colors_precomp.ndim == 2:
            colors_precomp = colors_precomp[render_indices]
        elif colors_precomp.ndim == 1:
            colors_precomp = colors_precomp.unsqueeze(0)

    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs if colors_precomp is None else None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None
    )
    return rendered_image, radii, rendered_depth, rendered_alpha

class MultiViewSAMMaskRefiner:
    """Refines SAM masks by enforcing consistency across overlapping views"""
    
    def __init__(self, overlap_threshold=0.3, consensus_strategy="majority_vote", log_to_rerun=False, visualize_matches=False, vote_collection_strategy = "projection"):
        self.overlap_threshold = overlap_threshold
        self.consensus_strategy = consensus_strategy
        self.log_to_rerun = log_to_rerun
        self.visualize_matches = visualize_matches
        self.vote_collection_strategy = vote_collection_strategy
        self.current_max_id = 0  # Track the current maximum ID
        if self.log_to_rerun:
            print("MultiViewSAMMaskRefiner initialized with logging to rerun enabled")
        if self.visualize_matches:
            print("MultiViewSAMMaskRefiner initialized with enabled visualization for gaussian matches")
        print(f"Using vote collection strategy: {self.vote_collection_strategy}")
        
    def find_overlapping_cameras(self, cameras):
        """Find pairs of cameras with overlapping views using frustum intersection"""
        overlapping_pairs = []
        camera_overlap_count = [0] * len(cameras)  # Track overlaps per camera
        max_overlaps_per_camera = 7
        
        for i, cam1 in enumerate(cameras):
            # Skip if this camera already has too many overlaps
            if camera_overlap_count[i] >= max_overlaps_per_camera:
                continue
                
            for j, cam2 in enumerate(cameras[i+1:], i+1):
                # Skip if either camera has too many overlaps
                if camera_overlap_count[j] >= max_overlaps_per_camera:
                    continue
                    
                if self._cameras_overlap(cam1, cam2):
                    overlapping_pairs.append((i, j))
                    camera_overlap_count[i] += 1
                    camera_overlap_count[j] += 1
                    
                    # Break if current camera has reached its limit
                    if camera_overlap_count[i] >= max_overlaps_per_camera:
                        break
                        
        return overlapping_pairs
    
    def _cameras_overlap(self, cam1: Camera, cam2: Camera):
        """Determine if two cameras have overlapping views using multiple tests"""
        is_overlap = False
        
        # Extract camera positions and viewing directions
        cam1_in_world = cam1.world_view_transform_no_t.inverse()
        cam2_in_world = cam2.world_view_transform_no_t.inverse()
        
        # Get camera positions and forward directions
        pos1 = cam1_in_world[:3, 3]
        pos2 = cam2_in_world[:3, 3]
        
        # Camera forward direction (POSITIVE Z in camera space)
        forward_cam = torch.tensor([0., 0., 1., 0.], device=cam1_in_world.device)
        
        # Transform to world space
        forward1_in_world = (cam1_in_world @ forward_cam)[:3]
        forward2_in_world = (cam2_in_world @ forward_cam)[:3]
        
        # Normalize directions
        forward1_in_world = forward1_in_world / torch.norm(forward1_in_world)
        forward2_in_world = forward2_in_world / torch.norm(forward2_in_world)
        
        # Calculate distance between cameras
        distance = torch.norm(pos1 - pos2)
        
        # Calculate angle between viewing directions
        dot_product = torch.dot(forward1_in_world, forward2_in_world)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Numerical stability
        angle_between_directions = torch.acos(dot_product)
        
        # 1. Fast rejection test - if cameras are too far apart or facing opposite directions
        max_distance = 100.0
        max_direction_angle = math.pi * 0.75  # 135 degrees
        
        if distance > max_distance or angle_between_directions > max_direction_angle:
            if self.log_to_rerun:
                print(f"Skipping overlap check: distance {distance:.2f} > {max_distance} or angle {angle_between_directions:.2f} > {max_direction_angle}")
            return False
        
        # 2. Special case: cameras are close and looking in similar directions
        # This catches the "side-by-side looking at same point" case
        if distance < 1.0 and angle_between_directions < math.pi / 6:  # 30 degrees
            is_overlap = True
                
        # Define the point-in-fov check function within the outer function
        def _is_point_in_camera_fov(point, camera_pos, camera_forward, fov_x, fov_y):
            """Check if a point is within a camera's field of view"""
            # Vector from camera to point
            to_point = point - camera_pos
            distance = torch.norm(to_point)
            
            # If point is too close to camera, it's not useful for overlap test
            if distance < 0.001:
                return False
                
            to_point_normalized = to_point / distance
            
            # Angle between camera forward and direction to point
            cos_angle = torch.dot(camera_forward, to_point_normalized)
            cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
            
            # If point is behind camera
            if cos_angle <= 0:
                return False
            
            # Check if within horizontal and vertical FOV
            # This is a simplification - for more precision, project to image plane
            angle = torch.acos(cos_angle)
            max_fov = max(fov_x, fov_y) / 2.0
            
            # Add a small margin for partial overlaps
            fov_margin = 0.1  # About 5.7 degrees
            return angle < (max_fov + fov_margin)
        
        # 3. Check intersection of viewing frusta using projected points
        # Generate sample points along each camera's central viewing ray
        # Use more points with varying depths for better coverage
        depths = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0, 20.0], device=pos1.device)
        
        # Also sample slightly off-axis points for better coverage of the view frustum
        # These offsets create points that are slightly away from the central ray
        offsets = [
            torch.tensor([0.0, 0.0, 0.0], device=pos1.device),  # Center ray
            torch.tensor([0.1, 0.0, 0.0], device=pos1.device),  # Slightly right
            torch.tensor([-0.1, 0.0, 0.0], device=pos1.device), # Slightly left
            torch.tensor([0.0, 0.1, 0.0], device=pos1.device),  # Slightly up
            torch.tensor([0.0, -0.1, 0.0], device=pos1.device)  # Slightly down
        ]
        
        # Extract camera right and up directions for offset calculation
        right1 = cam1_in_world[:3, 0]
        up1 = cam1_in_world[:3, 1]
        right2 = cam2_in_world[:3, 0]
        up2 = cam2_in_world[:3, 1]
        
        # Generate sample points for both cameras
        cam1_sample_points = []
        cam2_sample_points = []
        
        for d in depths:
            # Base point at this depth
            base_point1 = pos1 + d * forward1_in_world
            base_point2 = pos2 + d * forward2_in_world
            
            for offset in offsets:
                # Calculate points with offsets relative to camera orientation
                # Scale offsets by depth for wider coverage at greater distances
                offset_scale = d * 0.1  # 10% of depth as offset scale
                
                # Create offset points for camera 1
                offset_vector1 = offset[0] * right1 * offset_scale + offset[1] * up1 * offset_scale
                cam1_sample_points.append(base_point1 + offset_vector1)
                
                # Create offset points for camera 2
                offset_vector2 = offset[0] * right2 * offset_scale + offset[1] * up2 * offset_scale
                cam2_sample_points.append(base_point2 + offset_vector2)
        
        # Check if points from camera 1 are visible in camera 2
        for point in cam1_sample_points:
            if _is_point_in_camera_fov(point, pos2, forward2_in_world, cam2.FoVx, cam2.FoVy):
                is_overlap = True
                break  # Early exit if overlap found
                    
        # If no overlap found yet, check if points from camera 2 are visible in camera 1
        if not is_overlap:
            for point in cam2_sample_points:
                if _is_point_in_camera_fov(point, pos1, forward1_in_world, cam1.FoVx, cam1.FoVy):
                    is_overlap = True
                    break  # Early exit if overlap found
        
        # 4. Calculate midpoint test (additional test for better detection)
        if not is_overlap:
            # Find a point midway between the cameras
            midpoint = (pos1 + pos2) / 2.0
            
            # Create a point slightly offset from midpoint toward scene center
            # This helps with cameras looking at the same general area
            scene_center_direction = (forward1_in_world + forward2_in_world) / 2.0
            scene_center_direction = scene_center_direction / torch.norm(scene_center_direction)
            
            # Create sample points along the scene center direction from the midpoint
            mid_samples = [midpoint + d * scene_center_direction for d in [1.0, 3.0, 5.0]]
            
            # Check if both cameras can see any of these midpoint samples
            for mid_sample in mid_samples:
                if (_is_point_in_camera_fov(mid_sample, pos1, forward1_in_world, cam1.FoVx, cam1.FoVy) and
                    _is_point_in_camera_fov(mid_sample, pos2, forward2_in_world, cam2.FoVx, cam2.FoVy)):
                    is_overlap = True
                    break
        
        # Visualization and debugging
        if self.log_to_rerun and False:
            rot1_q = mat_to_quat(cam1_in_world[:3, :3].unsqueeze(0)).squeeze(0).cpu().numpy()
            rot2_q = mat_to_quat(cam2_in_world[:3, :3].unsqueeze(0)).squeeze(0).cpu().numpy()
            image1 = cam1.original_image.cpu().numpy().transpose(1, 2, 0)
            if image1.dtype != np.uint8:
                image1 = np.clip(image1 * 255, 0, 255).astype(np.uint8)
            image2 = cam2.original_image.cpu().numpy().transpose(1, 2, 0)
            if image2.dtype != np.uint8:
                image2 = np.clip(image2 * 255, 0, 255).astype(np.uint8)
            
            log_camera_pose(
                "camera_overlap_debug/1",
                cam1_in_world.cpu().numpy()[:3, 3],
                np.array([rot1_q[0], rot1_q[1], rot1_q[2], rot1_q[3]]),
                cam1.intrinsic_matrix.cpu().numpy(),
                cam1.image_width,
                cam1.image_height,
                image=image1,
            )
            log_camera_pose(
                "camera_overlap_debug/2",
                cam2_in_world.cpu().numpy()[:3, 3],
                np.array([rot2_q[0], rot2_q[1], rot2_q[2], rot2_q[3]]),
                cam2.intrinsic_matrix.cpu().numpy(),
                cam2.image_width,
                cam2.image_height,
                image=image2,
            )
            
            # Visualize camera forward directions
            rr.log(
                "camera_overlap_debug/z-vectors",
                rr.Arrows3D(
                    vectors=[forward1_in_world.cpu().numpy(),
                            forward2_in_world.cpu().numpy()],
                    colors=[[0, 0, 255],
                            [0, 50, 255]],
                ),
            )
            
            print("Overlap check between cameras:"
                f"\noverlap: {is_overlap},"
                f"\ndistance: {distance:.2f},"
                f"\nangle: {angle_between_directions:.2f} rad")
            input("Pause (checking camera overlap): press a key to continue")
        
        return is_overlap
    
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
        if self.log_to_rerun and False:
            rr.log(f"gs_{index_gs}/camera_{index_cam}/camera_pose/gs_in_cam", rr.Points3D(point_camera.cpu().numpy()[:3], radii=0.01, colors=[0, 0, 255]))

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
    
    def apply_consensus_rule(self, votes, confidence_threshold=0.8):
        """Apply consensus rule to resolve mask ID conflicts with confidence threshold"""
        if not votes:
            return -1  # Invalid/no consensus
            
        if self.consensus_strategy == "majority_vote":
            # Count votes for each mask ID
            vote_counts = defaultdict(int)
            total_valid_votes = 0
            
            for _, mask_id in votes:
                if mask_id >= 0:  # Only count valid mask IDs
                    vote_counts[mask_id] += 1
                    total_valid_votes += 1
            
            SPAXY_THRESHOLD = 4 # did 62th scene with 3
            if vote_counts and total_valid_votes >= SPAXY_THRESHOLD:
                # Find the mask ID with most votes
                winning_mask_id = max(vote_counts, key=vote_counts.get)
                max_votes = vote_counts[winning_mask_id]
                
                # Calculate confidence as percentage of total valid votes
                confidence = max_votes / total_valid_votes
                
                # Only return consensus if confidence exceeds threshold
                if confidence >= confidence_threshold:
                    return winning_mask_id
                
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
    
    def refine_sam_masks(self, cameras, sam_masks, gaussians: GaussianModel, sam_level=0):
        """Refine SAM masks using multi-view consistency with efficient overlap detection"""
        
        # DEBUG
        print(f"covariance: {gaussians.get_covariance()[0:5]}")
        print(f"rotation shape: {gaussians.get_rotation.shape}")

        
        if self.log_to_rerun:
            rr.init("sam_refinement",spawn=True)
            rr.log(
                "world_frame",
                rr.Arrows3D(
                    vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                ),
            )
        
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
                rr.log(f"gaussian_pointcloud", rr.Points3D(gaussians.get_xyz.cpu(), radii=0.005, colors=[0, 101, 189]))
            
            # For each sampled Gaussian, collect votes from overlapping cameras only
            for gaussian_idx in range(0, num_gaussians, sample_step):
                DEBUG_INDEX_TO_EXCLUDE = 26052
                # gaussian_idx = DEBUG_INDEX_TO_EXCLUDE
                gaussian_3d = gaussians.get_xyz[gaussian_idx]  # Actual 3D position
                
                if self.log_to_rerun:
                    gaussian_3d_cpu = gaussian_3d.cpu()
                    rr.log(f"gs_{gaussian_idx}", rr.Points3D(gaussian_3d_cpu, radii=0.02, colors=[227,114,34]))
                
                # Project this Gaussian only to overlapping cameras
                votes = []
                for other_cam_idx, other_camera in enumerate(cameras):
                    other_camera = other_camera
                    x, y, visible = self.project_3d_points_to_image(gaussian_3d, other_camera, gaussian_idx, other_cam_idx)
                    fetched_nonzero_diff = False
                    if self.log_to_rerun:
                        print(f"Camera {other_cam_idx} visibility: {visible}, x: {x}, y: {y}")
                        
                        # Testing rendering single gaussian splat
                        # rendered_image, _, _, _ = render_single_gaussian(other_camera, gaussians, gaussian_idx=gaussian_idx)
                        rendered_image_all, _,rendered_depth_all,_ = render_gaussians_with_exclusion(other_camera, gaussians, exclude_indices=[])
                        rendered_image_exclude, _,rendered_depth_exclude,_ = render_gaussians_with_exclusion(other_camera, gaussians, exclude_indices=[gaussian_idx])
                        rendered_image_all = fix_image(rendered_image_all)
                        rendered_image_exclude = fix_image(rendered_image_exclude)
                        
                        print(f"SHAPE DEPTH IMNPUT: {rendered_depth_all.shape}")
                        def depth_to_rgb(depth_map):
                            # Convert to numpy array if needed
                            if isinstance(depth_map, torch.Tensor):
                                depth_map = depth_map.detach().cpu().numpy()
                            # Squeeze channel dimension if present (e.g., [1, H, W] -> [H, W])
                            if depth_map.ndim == 3 and depth_map.shape[0] == 1:
                                depth_map = np.squeeze(depth_map, axis=0)
                            # Now depth_map should be [H, W]
                            depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
                            depth_norm = depth_norm.astype(np.uint8)
                            # Apply colormap (expects [H, W])
                            depth_rgb = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                            return depth_rgb
                        depth_image = depth_to_rgb(rendered_depth_all)
                        
                        if visible:
                            fetched_nonzero_diff = save_images_and_difference(rendered_image_all, rendered_image_exclude, int(x), int(y))
                        
                        # Get mask for the single gaussian
                        # non_black_mask = np.any(rendered_image != 0, axis=2)
                        # num_non_black_pixels = np.count_nonzero(non_black_mask)
                        # print(f"Render for {other_cam_idx} of non-black pixels: {num_non_black_pixels}")
                        
                        # NOTE: remplaced logging of original image with rendered image of single splat
                        image = other_camera.original_image.cpu().numpy().transpose(1, 2, 0)
                        if image.dtype != np.uint8:
                            image = np.clip(image * 255, 0, 255).astype(np.uint8)
                        image = depth_image
                        print(f"DEPTH SHAPE: {depth_image.shape}")
                    
                    if visible and x is not None and y is not None:
                        mask_id = sam_masks[other_cam_idx][sam_level, y, x].item() # x y or y x???
                        votes.append((other_cam_idx, mask_id))
                        
                        if self.log_to_rerun:
                            pass
                            # image[non_black_mask] = (227,114,34)
                            # cv2.circle(image, (int(x), int(y)), radius=5, color=(255, 0, 0), thickness=-1)
                    
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


    def project_3d_points_to_image_batch(self, camera: Camera, gaussians: GaussianModel, gaussian_indices=None, index_cam=0, use_depth=False):
        """
        Project batch of 3D points in world coordinates into 2D image pixel coordinates.
        
        Parameters:
        - points_3d (torch.Tensor): A tensor of shape (N, 3) representing points in world coordinates.
        - camera (Camera): Camera object
        - gaussian_indices: indices for logging (optional)
        - index_cam: camera index for logging
        
        Returns:
        - u, v (torch.Tensor): pixel coordinates of shape (N,) each
        - visible (torch.Tensor): boolean mask of shape (N,) indicating visibility
        """
        points_3d = gaussians.get_xyz[gaussian_indices] if gaussian_indices is not None else gaussians.get_xyz
        N = points_3d.shape[0]
        
        # Convert points to homogeneous coordinates (add a fourth '1' coordinate)
        ones = torch.ones(N, 1, device=points_3d.device)
        points_3d_homogeneous = torch.cat([points_3d, ones], dim=1)  # (N, 4)
        
        # Get coordinates in camera space (batch matrix multiplication)
        points_camera = (camera.world_view_transform_no_t @ points_3d_homogeneous.T).T  # (N, 4)
        
        # Logging only for first point if enabled and batch size is 1
        if self.log_to_rerun and N == 1 and gaussian_indices is not None:
            rr.log(f"gs_{gaussian_indices[0]}/camera_{index_cam}/camera_pose/gs_in_cam", 
                rr.Points3D(points_camera[0, :3].cpu(), radii=0.01, colors=[0, 0, 255]))
        
        # Apply projection matrix to map to clip space
        points_clip = (camera.projection_matrix_no_t @ points_camera.T).T  # (N, 4)
        
        # Perspective division
        w = points_clip[:, 3]  # (N,)
        # Avoid division by zero
        w = torch.where(torch.abs(w) < 1e-8, torch.sign(w) * 1e-8, w)
        points_ndc = points_clip / w.unsqueeze(1)  # (N, 4)
        
        # Viewport transformation
        u = points_ndc[:, 0] * (camera.image_width / 2.0) + camera.cx   # (N,)
        v = points_ndc[:, 1] * (camera.image_height / 2.0) + camera.cy  # (N,)
        
        # Check bounds
        in_bounds = (u >= 0) & (u < camera.image_width) & (v >= 0) & (v < camera.image_height)
        
        # Check if points are in front of the camera (z > 0 in camera space)
        is_in_front = points_camera[:, 2] > 0  # (N,)
        visible = is_in_front & in_bounds  # (N,)
        
        
        DEPTH_DIFF_THRESHOLD = 0.15 # 15cm
        DIST_OPTICAL_CENTER_IMAGE = 0.1 # 10cm (took from rerun image metadata)
        if use_depth==True:
            # print(f"Visibility before depth check: {visible}")
            u_int = u.cpu().numpy().astype(int)
            v_int = v.cpu().numpy().astype(int)
            
            # TODO: despaxx this setting to zero
            u_int[~visible.cpu().numpy()] = 0
            v_int[~visible.cpu().numpy()] = 0
            
            cam2world = np.linalg.inv(camera.world_view_transform_no_t.cpu().numpy())
            t = cam2world[:3, 3]
            diff = points_3d.cpu().numpy() - t  # shape: (N, 3)
            dist_image_point = np.linalg.norm(diff, axis=1) - DIST_OPTICAL_CENTER_IMAGE  # shape: (N,)
            # dist_image_point = np.linalg.norm(points_3d.cpu().numpy() - t) - DIST_OPTICAL_CENTER_IMAGE
            rendered_depth = camera.depth_map[:, v_int, u_int].detach()  
            visible_depth = (
                torch.abs(torch.from_numpy(dist_image_point).to(rendered_depth.device) - rendered_depth) < DEPTH_DIFF_THRESHOLD
            )
            visible = visible & visible_depth
            
            if self.log_to_rerun:
                print(f"Euclidean distance (from aprox image plane) = {dist_image_point}, shape: {dist_image_point.shape}")
                print(f"Rendered depth (u={int(u[0])},v={int(v[0])}) = {rendered_depth}, shape: {rendered_depth.shape}") # shape 1
                print(f"After depth chekcing: {visible_depth}, total verdict: {visible}")

            if self.log_to_rerun == True and visible.flatten().any() == True:
                image = camera.original_image.cpu().numpy().transpose(1, 2, 0)
                all_splat_masks = np.zeros_like(image[:, :, 0])
                # print(f"shape all masks {all_splat_masks.shape}")
                for i, gaussian_idx in enumerate(gaussian_indices):
                    if visible.flatten()[i] == True:
                        rendered_image, radii, _, _ = render_single_gaussian(camera, gaussians, gaussian_idx=gaussian_idx, use_view_inv_white_shs=True)
                        rendered_image = fix_image(rendered_image)
                        
                        weight_map = rgb_to_weight_map(rendered_image)
                        save_weight_map_to_csv(weight_map, filename="test.csv")
                        
                        # Get mask for the single gaussian
                        non_black_mask = np.any(rendered_image != 0, axis=2)
                        num_non_black_pixels = np.count_nonzero(non_black_mask)
                        covariance_matrix = gaussians.get_covariance_full_matrix()[gaussian_idx]
                        # print(f"splat ID = {gaussian_idx}):"
                        #     f"\n   non-black:{num_non_black_pixels},"
                        #     f"\n   opacity:{gaussians.get_opacity[gaussian_idx].item()},"
                        #     #   f"\n   cov:\n{covariance_matrix},"
                        #     #   f"\n   eigs:\n{torch.linalg.eigvals(covariance_matrix)},"
                        #     f"\n   radii value: {radii.item()}")
                    
                        rr.log(
                            f"trajectory_segment_{gaussian_idx}",
                            rr.LineStrips3D(
                                [t.tolist(), points_3d[i].tolist()],
                                colors=[0, 255, 255],
                                radii=0.002,
                            ),
                        )
                        all_splat_masks = np.logical_or(all_splat_masks, non_black_mask)
                R = cam2world[:3, :3]
                rot_q = mat_to_quat(torch.from_numpy(R).unsqueeze(0)).squeeze(0).numpy()
                K = camera.intrinsic_matrix.cpu().numpy()
                if image.dtype != np.uint8:
                    image = np.clip(image * 255, 0, 255).astype(np.uint8)
                # cv2.circle(image, (int(u[0]), int(v[0])), radius=5, color=(255, 0, 0), thickness=-1)
                image[all_splat_masks] = (227,114,34)
                log_camera_pose(
                    f"gs/camera",
                    t,
                    np.array([rot_q[0], rot_q[1], rot_q[2], rot_q[3]]),
                    K,
                    camera.image_width,
                    camera.image_height,
                    image=image,
                )       
        return u, v, visible

    def collect_mask_votes_batch(self, points_3d, overlapping_cam_indices, cameras, sam_masks, 
                            sam_level=0, gaussian_indices=None, gaussians=None):
        """
        Collect mask ID votes for a batch of 3D points from overlapping cameras.
        
        Parameters:
        - points_3d (torch.Tensor): shape (N, 3) - batch of 3D points
        - overlapping_cam_indices: list of camera indices that overlap with current camera
        - cameras: list of camera objects
        - sam_masks: list of SAM masks
        - sam_level: SAM level to use
        - gaussian_indices: indices for logging
        
        Returns:
        - votes_list: list of votes for each point, where each vote is [(cam_idx, mask_id), ...]
        """
        N = points_3d.shape[0]
        votes_list = [[] for _ in range(N)]
        
        for other_cam_idx in overlapping_cam_indices:
            other_camera = cameras[other_cam_idx]
            
            if self.vote_collection_strategy == "rasterizing":
                # print(f"Collecting votes for camera {other_cam_idx} using rasterizing strategy")
                
                i = 0
                for gaussian_index in gaussian_indices:
                    if i % 10 != 0:
                        # Skip 9 of 10 gaussians for now (runtime)
                        i += 1
                        continue
                    image, _, _, _ = render_single_gaussian(other_camera, gaussians, gaussian_idx=gaussian_index)
                    if not isinstance(image, np.ndarray):
                        image = image.detach().cpu().numpy()
                        image = np.transpose(image, (1, 2, 0))
                    if image.dtype != np.uint8:
                        image = np.clip(image * 255, 0, 255).astype(np.uint8)
                    if image.ndim == 2:
                        image = np.stack([image]*3, axis=-1)  # Make grayscale 3-channel
                    if image.shape[2] == 1:
                        image = np.repeat(image, 3, axis=2)
                    image = np.ascontiguousarray(image)
 
                    # Obtain mask from image
                    splat_binary_mask = np.any(image != 0, axis=2)
                    # print(f"Number of non-black pixels in rendered image for Gaussian {gaussian_index} in camera {other_cam_idx}: {np.count_nonzero(splat_binary_mask)}")
                    if sam_masks[other_cam_idx] is not None and np.count_nonzero(splat_binary_mask):
                        mask_id = self._get_most_common_id_in_mask(sam_masks[other_cam_idx], splat_binary_mask)
                        if mask_id >= 0:
                            votes_list[i].append((other_cam_idx, mask_id))
                            # print(f"Collected vote for Gaussian {gaussian_index}/{i} in camera {other_cam_idx}: mask_id={mask_id}")
                    i += 1
            elif self.vote_collection_strategy == "projection":
                # print(f"Collecting votes for camera {other_cam_idx} using projection strategy")
                # Project all points to this camera in parallel
                u, v, visible = self.project_3d_points_to_image_batch(
                    points_3d, other_camera, gaussian_indices, other_cam_idx
                )
                
                # Logging for single point if enabled
                if self.log_to_rerun and N == 1 and gaussian_indices is not None:
                    print(f"Camera {other_cam_idx} visibility: {visible[0].item()}, x: {u[0].item()}, y: {v[0].item()}")
                    
                    image = other_camera.original_image.cpu().numpy().transpose(1, 2, 0)
                    if image.dtype != np.uint8:
                        image = np.clip(image * 255, 0, 255).astype(np.uint8)
                
                # Get mask IDs for visible points
                if sam_masks[other_cam_idx] is not None:
                    mask_shape = sam_masks[other_cam_idx].shape
                    H, W = mask_shape[1], mask_shape[2]
                    
                    for i in range(N):
                        if visible[i]:
                            x, y = int(u[i].item()), int(v[i].item())
                            if 0 <= y < H and 0 <= x < W:
                                mask_id = sam_masks[other_cam_idx][sam_level, y, x].item()
                                votes_list[i].append((other_cam_idx, mask_id))
                                
                                # Logging for single point
                                if self.log_to_rerun and N == 1 and i == 0:
                                    cv2.circle(image, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
            
            # Camera pose logging for single point
            if self.log_to_rerun and N == 1 and gaussian_indices is not None:
                world2cam = np.linalg.inv(other_camera.world_view_transform_no_t.cpu().numpy())
                t = world2cam[:3, 3]
                R = world2cam[:3, :3]
                rot_q = mat_to_quat(torch.from_numpy(R).unsqueeze(0)).squeeze(0).numpy()
                K = other_camera.intrinsic_matrix.cpu().numpy()
                log_camera_pose(
                    f"gs_{gaussian_indices[0]}/camera_{other_cam_idx}",
                    t,
                    np.array([rot_q[0], rot_q[1], rot_q[2], rot_q[3]]),
                    K,
                    other_camera.image_width,
                    other_camera.image_height,
                    image=image,
                )
        
        return votes_list

    def refine_sam_masks_batch(self, cameras, sam_masks, gaussians, sam_level=0):
        """Refine SAM masks using multi-view consistency with GPU-parallelized processing"""
        
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
            overlapping_cam_indices = list(overlap_map[cam_idx])
            
            if not overlapping_cam_indices:
                # No overlaps, keep original mask
                refined_masks.append(refined_mask)
                continue
            
            # Sample subset of Gaussians for efficiency (every 10th Gaussian)
            sample_step = 1
            num_gaussians = gaussians.get_xyz.shape[0]
            
            if self.log_to_rerun:
                rr.init("sam_refinement", spawn=True)
                rr.log(
                    "world_frame",
                    rr.Arrows3D(
                        vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                    ),
                )
                rr.log(f"gaussian_pointcloud", rr.Points3D(gaussians.get_xyz.cpu(), radii=0.005, colors=[0, 255, 0]))
            
            # Process Gaussians in batches for GPU parallelization
            batch_size = 4096 if not self.log_to_rerun else 1  # Use batch size 1 for logging
            
            for batch_start in range(0, num_gaussians, batch_size * sample_step):
                batch_end = min(batch_start + batch_size * sample_step, num_gaussians)
                
                # Get batch of Gaussian indices
                gaussian_indices = list(range(batch_start, batch_end, sample_step))
                if not gaussian_indices:
                    continue
                    
                # Extract batch of 3D positions
                batch_gaussians_3d = gaussians.get_xyz[gaussian_indices]  # (batch_size, 3)
                
                # Logging for single point
                if self.log_to_rerun and len(gaussian_indices) == 1:
                    gaussian_3d_cpu = batch_gaussians_3d[0].cpu()
                    rr.log(f"gs_{gaussian_indices[0]}", rr.Points3D(gaussian_3d_cpu, radii=0.01, colors=[255, 0, 0]))
                
                # Collect votes for all points in batch from overlapping cameras
                votes_list = self.collect_mask_votes_batch(
                    batch_gaussians_3d, overlapping_cam_indices, cameras, sam_masks, 
                    sam_level, gaussian_indices, gaussians
                )
                
                if self.vote_collection_strategy == "projection":
                    # Project batch to current camera
                    u_curr, v_curr, visible_curr = self.project_3d_points_to_image_batch(
                        batch_gaussians_3d, camera
                    )
                    
                    # Process each point in the batch
                    for i, (gaussian_idx, votes) in enumerate(zip(gaussian_indices, votes_list)):
                        if not votes:
                            continue
                            
                        # Apply consensus rule
                        consensus_id = self.apply_consensus_rule(votes)
                        
                        if visible_curr[i] and consensus_id >= 0:
                            x_curr, y_curr = int(u_curr[i].item()), int(v_curr[i].item())
                            
                            # Update mask
                            H, W = refined_mask.shape[1], refined_mask.shape[2]
                            changes_made = 0
                            
                            if 0 <= y_curr < H and 0 <= x_curr < W:
                                if refined_mask[sam_level, y_curr, x_curr] != consensus_id:
                                    refined_mask[sam_level, y_curr, x_curr] = consensus_id
                                    changes_made += 1
                            
                            # Debug visualization for single point
                            if self.visualize_matches and changes_made > 0:
                                self.debug_visualize_projections(
                                    batch_gaussians_3d[i], votes, consensus_id, cameras, sam_masks, 
                                    sam_level, current_cam_idx=cam_idx, max_pairs=2, gaussian_idx=gaussian_idx
                                )
                        
                        # Logging pause for single point
                        if self.log_to_rerun and len(gaussian_indices) == 1:
                            input("Pause: press a key to continue")
                elif self.vote_collection_strategy == "rasterizing":
                    for i, (gaussian_idx, votes) in enumerate(zip(gaussian_indices, votes_list)):
                        if not votes:
                            continue
                        
                        # Apply consensus rule
                        consensus_id = self.apply_consensus_rule(votes)
                        
                        image, _, _, _ = render_single_gaussian(camera, gaussians, gaussian_idx=gaussian_idx)
                        if not isinstance(image, np.ndarray):
                            image = image.detach().cpu().numpy()
                            image = np.transpose(image, (1, 2, 0))
                        if image.dtype != np.uint8:
                            image = np.clip(image * 255, 0, 255).astype(np.uint8)
                        if image.ndim == 2:
                            image = np.stack([image]*3, axis=-1)  # Make grayscale 3-channel
                        if image.shape[2] == 1:
                            image = np.repeat(image, 3, axis=2)
                        image = np.ascontiguousarray(image)
    
                        # Obtain mask from image
                        splat_binary_mask = np.any(image != 0, axis=2)
                        mask_id = -1
                        if np.count_nonzero(splat_binary_mask):
                            mask_id = self._get_most_common_id_in_mask(refined_mask, splat_binary_mask)
                        if mask_id != consensus_id and consensus_id >= 0:
                            print(f"Updating mask for camera {cam_idx} for Gaussian {gaussian_idx} with consensus ID {consensus_id} (replacing {mask_id})")
                            refined_mask[sam_level, splat_binary_mask] = consensus_id
            refined_masks.append(refined_mask)
        
        return refined_masks
    
    def _get_most_common_id_in_mask_weighted(self, sam_mask: torch.Tensor, weight_matrix: torch.Tensor) -> int:
        """
        Find the most dominant ID in an image based on weighted counts using GPU parallelization.
        
        Args:
            sam_mask: torch.Tensor with shape [H, W] containing ID values
            weight_matrix: torch.Tensor with shape [H, W] containing weights for each pixel
            
        Returns:
            int: The ID with the highest weighted sum
        """
        # Ensure both tensors are on the same device
        device = sam_mask.device
        if weight_matrix.device != device:
            weight_matrix = weight_matrix.to(device)
        
        # Ensure weight_matrix has the right shape [H, W] (squeeze if needed)
        if weight_matrix.ndim == 3 and weight_matrix.shape[2] == 1:
            weight_matrix = weight_matrix.squeeze(2)
        
        # Flatten both tensors
        sam_mask_flat = sam_mask.flatten()
        weights_flat = weight_matrix.flatten()
        
        # Find min and max IDs to handle negative values
        min_id = sam_mask_flat.min().item()
        max_id = sam_mask_flat.max().item()
        
        if min_id == max_id:
            return int(min_id)  # All pixels have the same ID
        
        # Shift IDs to make them non-negative for bincount
        # This allows us to handle negative IDs like -1
        offset = -min_id if min_id < 0 else 0
        shifted_ids = sam_mask_flat + offset
        
        # Use bincount to accumulate weights - this is GPU parallelized!
        weighted_counts = torch.bincount(shifted_ids.long(), weights=weights_flat, 
                                    minlength=int(max_id + offset) + 1)
        
        # Find the ID with maximum weighted count
        most_common_shifted_id = torch.argmax(weighted_counts).item()
        
        # Shift back to original ID space
        most_common_id = most_common_shifted_id - offset
        max_weight = weighted_counts[most_common_shifted_id].item()
        
        if self.log_to_rerun:
            print(f"Most dominant ID: {most_common_id} with weighted count: {max_weight:.3f}")
            
        return int(most_common_id)


    def _get_most_common_id_in_mask(self, image: torch.Tensor, mask: torch.Tensor, channel: int = 0) -> int:
        """
        Given an image [C, H, W] and a binary mask [H, W], return the most common ID
        at the specified channel inside the masked region.

        Args:
            image (torch.Tensor): Input tensor of shape [C, H, W].
            mask (torch.Tensor): Binary mask of shape [H, W].
            channel (int): Channel index containing semantic IDs.

        Returns:
            int: Most common ID value inside the mask.
        """
        # Extract the semantic ID map from the specified channel
        id_map = image[channel]  # [H, W]

        # Apply mask
        masked_ids = id_map[mask.astype(bool)]

        if masked_ids.numel() == 0:
            return -1  # No valid IDs in mask

        # Find most common ID
        ids, counts = torch.unique(masked_ids, return_counts=True)
        most_common_id = ids[counts.argmax()].item()

        return most_common_id
    
    def visualize_results(self, 
                          gaussians: GaussianModel,
                          cameras: list[Camera], 
                          original_sam_masks: list[torch.Tensor],
                          refined_masks: list[torch.Tensor],
                          gaussian_indices: list, 
                          splat_camera_correspondence: torch.Tensor,
                          stride: int,
                          starting_index: int,
                          sam_level: int = 0):
        SCALE_STRIDE = 1
        STARTING_INDEX_VIZ = 0
        num_gaussians = gaussians.get_xyz.shape[0]
        gaussian_viz_data = []
        
        rr.init("Sam_Refinement_Multistage", spawn=True)
        rr.log(
            "world_frame",
            rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )
        rr.log(f"gaussian_pointcloud", rr.Points3D(gaussians.get_xyz.cpu(), radii=0.005, colors=[0, 255, 0]))
        if gaussian_indices is not None:
            rr.log(f"selected_splats", rr.Points3D(gaussians.get_xyz[gaussian_indices].cpu(), radii=0.1, colors=[255, 0, 0]))
            
        for gaussian_id, splat_visibility_in_cams in enumerate(splat_camera_correspondence):
            gaussian_id_shifted_viz = gaussian_id*stride*SCALE_STRIDE+starting_index
            if gaussian_id_shifted_viz < num_gaussians and gaussian_id_shifted_viz >= STARTING_INDEX_VIZ:
                # print(f"Splat {gaussian_id_shifted_viz} visibility:\n{splat_visibility_in_cams}")
                for i, camera in enumerate(cameras):
                    if splat_visibility_in_cams[i]:
                        # print(f"{gaussian_id_shifted_viz} visible in cam {i}")
                        # Re-render to get fresh data
                        rendered_image, _, _, _ = render_single_gaussian(camera, gaussians, gaussian_id_shifted_viz)
                        rendered_image = fix_image(rendered_image)
                        non_black_mask = torch.any(rendered_image != 0, dim=2)

                        if not non_black_mask.any():
                            if self.log_to_rerun:
                                print(f"Skipping splat {gaussian_id_shifted_viz} - projected to image no non-black pixels")
                            continue
                        
                        # Get the UPDATED state of refined masks (after this update)
                        sam_mask_channel_after = refined_masks[i][sam_level].cpu().numpy()
                        rendered_ids_after = sam_mask_channel_after * non_black_mask.cpu().numpy().astype(int)

                        gaussian_viz_data.append({
                            'camera': camera,
                            'camera_idx': i,
                            'gaussian_id': gaussian_id_shifted_viz,
                            'rendered_ids': rendered_ids_after,  # Use AFTER data
                            'sam_mask_channel': sam_mask_channel_after,  # Use AFTER data
                            'rendered_image': rendered_image.cpu().numpy(),
                            'non_black_mask': non_black_mask.cpu().numpy()
                        })
        if gaussian_viz_data:
            cam_number_for_rerun = 0
            splat_id_prev_for_rerun = -1
            for viz_data in gaussian_viz_data:
                # print(f"Debug: gaussian id shifted {viz_data['gaussian_id']}")
                if splat_id_prev_for_rerun != viz_data['gaussian_id']:
                    cam_number_for_rerun = 0
                    splat_id_prev_for_rerun = viz_data['gaussian_id']
                    input("Next splat, press one more time")
                self.plot_masks(
                    viz_data['camera'],
                    viz_data['rendered_ids'],
                    viz_data['gaussian_id'],
                    viz_data['sam_mask_channel'],
                    viz_data['rendered_image'],
                    viz_data['non_black_mask'],
                    viz_data['camera_idx'],
                    original_sam_masks,
                    cam_number_for_rerun
                )
                cam_number_for_rerun += 1
    
    def plot_masks(self, camera, rendered_ids, gaussian_id, sam_mask_channel, rendered_image, non_black_mask, camera_idx, original_sam_masks, cam_num_for_rerun):
        # Use the same color scheme as in train.py
        np.random.seed(42)
        colors_defined = np.random.randint(100, 256, size=(300, 3))
        colors_defined[0] = np.array([0, 0, 0])  # Ignore the mask ID of -1 and set it to black.
        colors_defined = torch.from_numpy(colors_defined)
        
        # Create visualizations
        # plt.figure(figsize=(30, 5))

        # 1. Original SAM mask - using predefined colors
        # plt.subplot(1, 6, 1)
        original_img = original_sam_masks[camera_idx][0].cpu().numpy()
        original_mask_colored = colors_defined[original_img.astype(int).clip(0, 299)].numpy().astype(np.uint8)
        original_mask_colored = original_mask_colored.transpose(2, 0, 1)  # [H, W, 3] -> [3, H, W]
        # plt.imshow(original_mask_colored.transpose(1, 2, 0))  # [3, H, W] -> [H, W, 3]
        rr.log(
            f"orig_sam_mask_cam_{cam_num_for_rerun}",
            rr.Image(original_mask_colored.transpose(1, 2, 0)).compress(jpeg_quality=75),
        )
        # plt.title(f'Original SAM Mask - Camera {camera_idx}')
        # plt.axis('off')

        # 2. Refined SAM mask - using predefined colors
        # plt.subplot(1, 6, 2)
        refined_img = sam_mask_channel  # This IS the refined mask!
        refined_mask_colored = colors_defined[refined_img.astype(int).clip(0, 299)].numpy().astype(np.uint8)
        refined_mask_colored = refined_mask_colored.transpose(2, 0, 1)  # [H, W, 3] -> [3, H, W]
        # plt.imshow(refined_mask_colored.transpose(1, 2, 0))  # [3, H, W] -> [H, W, 3]
        rr.log(
            f"refined_sam_mask_cam_{cam_num_for_rerun}",
            rr.Image(refined_mask_colored.transpose(1, 2, 0)).compress(jpeg_quality=75),
        )
        # plt.title(f'Refined SAM Mask - Camera {camera_idx}')
        # plt.axis('off')

        # 3. Rendered IDs - using predefined colors
        # plt.subplot(1, 6, 3)
        rendered_mask_colored = colors_defined[rendered_ids.astype(int).clip(0, 299)].numpy().astype(np.uint8)
        rendered_mask_colored = rendered_mask_colored.transpose(2, 0, 1)  # [H, W, 3] -> [3, H, W]
        # plt.imshow(rendered_mask_colored.transpose(1, 2, 0))  # [3, H, W] -> [H, W, 3]
        rr.log(
            f"splat_in_cam_{cam_num_for_rerun}",
            rr.Image(rendered_mask_colored.transpose(1, 2, 0)).compress(jpeg_quality=75),
        )
        # plt.title(f'Rendered IDs - Gaussian {gaussian_id}')
        # plt.axis('off')

        # # 4. Non-black mask (Gaussian footprint binary)
        # plt.subplot(1, 6, 4)
        # plt.imshow(non_black_mask, cmap='gray')
        # plt.title(f'Non-black Mask\n(Gaussian Footprint)')
        # plt.axis('off')

        # # 5. Gaussian footprint (original rendered image)
        # plt.subplot(1, 6, 5)
        # plt.imshow(rendered_image)
        # plt.title(f'Gaussian Footprint\n(Rendered Image)')
        # plt.axis('off')

        # # 6. Difference between original and refined masks
        # plt.subplot(1, 6, 6)
        difference = np.abs(original_img.astype(np.float32) - refined_img.astype(np.float32))
        # plt.imshow(difference, cmap='hot', vmin=0, vmax=10)
        # plt.title(f'Mask Difference\n(Original vs Refined)')
        # plt.colorbar(label='ID Difference')
        # plt.axis('off')

        # plt.suptitle(f'Gaussian {gaussian_id} in Camera {camera_idx}', fontsize=16)
        # plt.tight_layout()
        # plt.show()

        # Print info
        unique_sam_ids = np.unique(sam_mask_channel[sam_mask_channel > -100], return_counts=True)
        unique_rendered_ids = np.unique(rendered_ids[rendered_ids > -100], return_counts=True )
        unique_original_ids = np.unique(original_img[original_img > -100], return_counts=True)
        unique_refined_ids = np.unique(refined_img[refined_img > -100], return_counts=True)
        
        print(f"Gaussian {gaussian_id}, Camera {camera_idx}:")
        print(f"  Original SAM mask segmentation IDs: {unique_original_ids}")
        print(f"  Refined SAM mask segmentation IDs: {unique_refined_ids}")
        # print(f"  SAM mask channel unique IDs: {unique_sam_ids}")
        print(f"  Splat segmentation IDs: {unique_rendered_ids}")
        print(f"  Footprint pixels: {np.count_nonzero(non_black_mask)}")
        print(f"  Changed pixels: {np.count_nonzero(difference > 0)}")
        print("-" * 50)
        
        input("Press for next plot")

    def expand_masks(self, 
                     cameras: list[Camera],
                     gaussian_id: int, 
                     sam_masks: list[torch.Tensor], 
                     cam_idx_splat_segment_id_weight_mask_pairs: list[int, int, torch.Tensor], 
                     initial_value = 0.0, 
                     sam_level: int = 0):
        """
        Create pixel-level ID-value mappings for each segmentation image using GPU parallelization.
        
        Args:
            gaussian_id: ID of the current Gaussian
            sam_masks: List of segmentation tensors
            initial_value: Initial value to assign to each ID
            
        Returns:
            List of pixel maps with efficient tensor-based storage
        """
            
        
        # Step 2: voting        
        id_vote_counts = {}
        for camera_idx, most_dominant_id, _ in cam_idx_splat_segment_id_weight_mask_pairs:
            if sam_masks[camera_idx] is None:
                continue
            
            # Count votes for this ID
            if most_dominant_id not in id_vote_counts:
                id_vote_counts[most_dominant_id] = 0
            id_vote_counts[most_dominant_id] += 1

        if id_vote_counts:
            winning_id = max(id_vote_counts, key=id_vote_counts.get)
            max_votes = id_vote_counts[winning_id]
            
            print(f"Splat {gaussian_id}: winning ID: {winning_id} with {max_votes} votes out of {len(cam_idx_splat_segment_id_weight_mask_pairs)} cameras")
            print(f"All vote counts: {id_vote_counts}")
            
            # Extend segments with projected splats
            for camera_idx, most_dominant_id, weight_map in cam_idx_splat_segment_id_weight_mask_pairs:
                if winning_id == most_dominant_id:
                    # Update count of base mask (s1)
                    object_mask_of_dominant_id = (sam_masks[camera_idx][sam_level] == winning_id)
                    idx = cameras[camera_idx].id_to_idx[winning_id]
                    y_indices, x_indices = torch.where(object_mask_of_dominant_id)
                    cameras[camera_idx].pixel_value_tensor[y_indices, x_indices, idx] += 1.0
                    print(f"Cam {camera_idx} counts: \n{cameras[camera_idx].pixel_value_tensor[y_indices, x_indices, idx]}")
                    
                    # Extend over mask (s2)
                    if weight_map.ndim == 3 and weight_map.shape[2] == 1:
                        weight_map = weight_map.squeeze(2)  # Remove the singleton dimension [H, W, 1] -> [H, W]
                        
                    # Get the weight matrix and create mask for non-zero weights
                    non_zero_weight_mask = weight_map > 0  # Pixels covered by the Gaussian
                    
                    # Find pixels that have non-zero weight BUT are NOT part of the main segment
                    extension_mask = non_zero_weight_mask & (~object_mask_of_dominant_id)
                    
                    if extension_mask.any():
                        # Get the weight values for pixels in the extension area
                        extension_weights = weight_map[extension_mask]
                        
                        # Get y, x indices for extension pixels
                        ext_y_indices, ext_x_indices = torch.where(extension_mask)
                        
                        # Increment these pixels by their corresponding weight values
                        cameras[camera_idx].pixel_value_tensor[ext_y_indices, ext_x_indices, idx] += extension_weights
                        print(f"Extended {len(ext_y_indices)} pixels for winning_id {winning_id} with weights")
                else:
                    pass
                    # TODO
                    # Check if winning_id is among image IDs
                    # if is there then add count for it in the non-overlap splat part with the winning_id segment (s2)
                    # else do nothing

    def update_masks(self, 
                     gaussian_id: int, 
                     cam_idx_splat_segment_id_pairs: list[int, int], 
                     sam_masks: list[torch.Tensor], 
                     sam_level: int = 0) -> list[torch.Tensor]:
        """
        Find the most common ID within the Gaussian's rendered footprint across all cameras,
        then update ALL pixels with that ID to a new global ID for consistency.

        Args:
            gaussian_id: ID of the current Gaussian
            cam_idx_splat_segment_id_pairs: list of tuples (camera_idx, most_dominant_id)
            sam_masks: list of all SAM masks to be refined
            sam_level: SAM level to use
        
        Returns:
            refined_masks: updated list of SAM masks
        """        
        if not cam_idx_splat_segment_id_pairs:
            return sam_masks  # No masks to process
        
        # Step 1: Update current_max_id with all existing SAM mask IDs to track what's in use
        for sam_mask in sam_masks:
            if sam_mask is not None:
                existing_ids = torch.unique(sam_mask[sam_level])
                valid_ids = existing_ids[existing_ids > 0]
                if len(valid_ids) > 0:
                    current_max = valid_ids.max().item()
                    self.current_max_id = max(self.current_max_id, current_max)
        
        # Step 2: Generate a new unique global ID
        self.current_max_id += 1
        new_global_id = self.current_max_id
        
        if self.log_to_rerun:
            print(f"Generated new global ID: {new_global_id}")
        
        # Step 3: Update ALL pixels with the most common ID to the new global ID
        # Create a copy of sam_masks to avoid modifying the original
        refined_masks = [mask.clone() if mask is not None else None for mask in sam_masks]
        
        for camera_idx, most_dominant_id in cam_idx_splat_segment_id_pairs:
            if refined_masks[camera_idx] is None:
                continue
            
            # Find ALL pixels with the most common ID in this camera's mask
            object_mask_of_dominant_id = (refined_masks[camera_idx][sam_level] == most_dominant_id)
            pixels_before = torch.sum(object_mask_of_dominant_id).item()
            
            if pixels_before > 0:
                # We don't overwrite 'void' classified objects, since there can be multiple different objects of class 'void' in the scene
                if most_dominant_id != -1:
                    # Update ALL pixels with the most common ID to the new global ID
                    refined_masks[camera_idx][sam_level][object_mask_of_dominant_id] = new_global_id
                    if self.log_to_rerun:
                        print(f"Cam ID={camera_idx}, splat ID={gaussian_id}: {pixels_before} pixels changed from ID {most_dominant_id} to {new_global_id}")
                else:
                    if self.log_to_rerun:
                        print(f"Cam ID={camera_idx}, splat ID={gaussian_id}: skipping change to {new_global_id} ID, actual ID={most_dominant_id}")
        
        return refined_masks


    def refine_sam_masks_multistage(self, cameras: list[Camera], sam_masks, gaussians, sam_level=0):
        
        # Initialize refined_masks as a copy of original sam_masks
        original_sam_masks = [mask.clone() if mask is not None else None for mask in sam_masks]
        refined_masks = [mask.clone() if mask is not None else None for mask in sam_masks]
        
        # Write depth map to camera instance.
        for cam_idx, camera in tqdm(enumerate(cameras), total=len(cameras), desc="Writing depth maps to camera frames"):
            _, _, rendered_depth, _ = render_gaussians_with_exclusion(camera, gaussians, exclude_indices=None)
            camera.depth_map = rendered_depth
        
        num_gaussians = gaussians.get_xyz.shape[0]
        STARTING_INDEX = 0
        STRIDE = 1
        gaussian_indices = range(STARTING_INDEX, 10, STRIDE)
        splat_camera_correspondence = torch.empty(
            (len(gaussian_indices) if gaussian_indices is not None else num_gaussians, len(cameras)), dtype=torch.bool)

        if self.log_to_rerun:
            rr.init("Sam_Refinement_Multistage", spawn=True)
            rr.log(
                "world_frame",
                rr.Arrows3D(
                    vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                ),
            )
            rr.log(f"gaussian_pointcloud", rr.Points3D(gaussians.get_xyz.cpu(), radii=0.005, colors=[0, 255, 0]))
            if gaussian_indices is not None:
                rr.log(f"selected_splats", rr.Points3D(gaussians.get_xyz[gaussian_indices].cpu(), radii=0.01, colors=[255, 0, 0]))
        
        # Stage 0: detect visible splats in camera frames
        for cam_idx, camera in tqdm(enumerate(cameras), total=len(cameras), desc="Writing splat to cam correspondence"):
            _, _, visible_curr = self.project_3d_points_to_image_batch(
                camera=camera, gaussians=gaussians, gaussian_indices=gaussian_indices, use_depth=True
            )
            if self.log_to_rerun:
                input("Pause: press a key to continue")
                
            splat_camera_correspondence[:, cam_idx] = visible_curr
        print(f"Splat to camera correspondence of shape {splat_camera_correspondence.shape}")
                
        # Stage 1: get cross-view consistent object IDs
        OPACITY_THRESHOLD = 0.99
        num_non_skipped = torch.sum(gaussians.get_opacity < OPACITY_THRESHOLD).item()
        print(f"Number not skipped splats for stage 1: {num_non_skipped}")
        for gaussian_id, splat_visibility_in_cams in tqdm(enumerate(splat_camera_correspondence), total=num_gaussians):
            gaussian_id_shifted = gaussian_id*STRIDE+STARTING_INDEX
            # start_time = time.time()
            cam_idx_splat_segment_id_pairs = []  # contains tuples (camera_idx, most_dominant_id)

            # Collect mask data before updating
            for i, camera in enumerate(cameras):
                if splat_visibility_in_cams[i]:
                    if gaussians.get_opacity[gaussian_id_shifted] < OPACITY_THRESHOLD:
                        if self.log_to_rerun:
                            print(f"Skipping splat {gaussian_id_shifted} - low opacity")
                        continue  # skip unreliable gaussians

                    rendered_image, _, _, _ = render_single_gaussian(camera, gaussians, gaussian_id_shifted, use_view_inv_white_shs=True)
                    rendered_image = fix_image(rendered_image)  # Convert to proper format
                    non_black_mask = torch.any(rendered_image != 0, dim=2)
                    weights_mask = rgb_to_weight_map(rendered_image)
                    most_dominant_id = self._get_most_common_id_in_mask_weighted(sam_mask=sam_masks[i][sam_level], weight_matrix=weights_mask)

                    if not non_black_mask.any():
                        if self.log_to_rerun:
                            print(f"Skipping splat {gaussian_id_shifted} - projected to image no non-black pixels")
                        continue

                    cam_idx_splat_segment_id_pairs.append((i, most_dominant_id))

            # Update refined_masks with the result from update_masks
            refined_masks = self.update_masks(gaussian_id_shifted, cam_idx_splat_segment_id_pairs, refined_masks, sam_level)
            
            # end_time = time.time()
            # print(f"Time: {end_time - start_time} s")
        
        id_mapping, refined_masks = create_consistent_id_mapping(refined_masks)
        print(f"Remapping:\n{id_mapping}")
        
        
        # VIZUALIZATION OF FIRST STAGE
        VIZUALIZE=False
        if VIZUALIZE:
            self.visualize_results(gaussians=gaussians,
                                   cameras=cameras,
                                   original_sam_masks=original_sam_masks,
                                   refined_masks=refined_masks,
                                   gaussian_indices=gaussian_indices,
                                   splat_camera_correspondence=splat_camera_correspondence,
                                   stride=STRIDE,
                                   starting_index=STARTING_INDEX,
                                   sam_level=sam_level)
                

        # Stage 2: expand splats
        # Init pixel lookup tables
        start_time = time.time()
        for i, camera in enumerate(cameras):
            if refined_masks[i][sam_level] is None:
                continue
            
            device = refined_masks[i][sam_level].device
            H, W = refined_masks[i][sam_level].shape
            
            # Get all unique IDs in this image
            camera.unique_ids = torch.unique(refined_masks[i][sam_level], sorted=True)
            num_ids = len(camera.unique_ids)
            
            # Create a mapping from ID values to indices (0, 1, 2, ...), find channel in tensor for an ID
            camera.id_to_idx = {id_val.item(): idx for idx, id_val in enumerate(camera.unique_ids)}
            
            # Create a 3D tensor [H, W, num_ids] where each "channel" represents one ID
            # This replaces the nested dictionary structure
            camera.pixel_value_tensor = torch.full((H, W, num_ids), 0.0, 
                                        dtype=torch.float32, device=device)
            
            camera.id_range = (camera.unique_ids.min().item(), camera.unique_ids.max().item())
            
            if self.log_to_rerun:
                print(f"Cam {i}: unique_ids={camera.unique_ids}"
                    f"\npixel_values shape={camera.pixel_value_tensor.shape}"
                    f"\nid_to_idx={camera.id_to_idx}"
                    f"\nid_range={camera.id_range}")
        
        end_time = time.time()
        print(f"Init duration pixel maps: {end_time - start_time}")
                
        # Mask expanding
        for gaussian_id, splat_visibility_in_cams in tqdm(enumerate(splat_camera_correspondence), total=num_gaussians):
            gaussian_id_shifted = gaussian_id*STRIDE+STARTING_INDEX
            # start_time = time.time()
            cam_idx_splat_segment_id_weight_mask_pairs = []  # contains tuples (camera_idx, most_dominant_id, weight_mask)

            # Collect mask data before updating
            for i, camera in enumerate(cameras):
                if splat_visibility_in_cams[i]:
                    if gaussians.get_opacity[gaussian_id_shifted] < OPACITY_THRESHOLD:
                        if self.log_to_rerun:
                            print(f"Skipping splat {gaussian_id_shifted} - low opacity")
                        continue  # skip unreliable gaussians

                    rendered_image, _, _, _ = render_single_gaussian(camera, gaussians, gaussian_id_shifted, use_view_inv_white_shs=True)
                    rendered_image = fix_image(rendered_image)  # Convert to proper format
                    non_black_mask = torch.any(rendered_image != 0, dim=2)
                    weights_mask = rgb_to_weight_map(rendered_image)
                    most_dominant_id = self._get_most_common_id_in_mask_weighted(sam_mask=refined_masks[i][sam_level], weight_matrix=weights_mask)

                    if not non_black_mask.any():
                        if self.log_to_rerun:
                            print(f"Skipping splat {gaussian_id_shifted} - projected to image no non-black pixels")
                        continue

                    cam_idx_splat_segment_id_weight_mask_pairs.append((i, most_dominant_id, weights_mask))

            # Update refined_masks with the result from update_masks
            self.expand_masks(cameras=cameras,
                              gaussian_id=gaussian_id_shifted, 
                              sam_masks=refined_masks, 
                              cam_idx_splat_segment_id_weight_mask_pairs=cam_idx_splat_segment_id_weight_mask_pairs, 
                              sam_level=sam_level)
            
            # Fetch ID of highest accumulated value
            expanded_masks = [mask.clone() if mask is not None else None for mask in refined_masks]
            for i, camera in enumerate(cameras):
                _, indices_with_max_counts = torch.max(camera.pixel_value_tensor, dim=2)
                # Convert indices back to actual IDs using the unique_ids tensor
                expanded_mask = camera.unique_ids[indices_with_max_counts]
                expanded_masks[i][sam_level] = expanded_mask

        # VIZUALIZATION OF SECOND STAGE
        VIZUALIZE=True
        if VIZUALIZE:
            self.visualize_results(gaussians=gaussians,
                                   cameras=cameras,
                                   original_sam_masks=refined_masks,
                                   refined_masks=expanded_masks,
                                   gaussian_indices=gaussian_indices,
                                   splat_camera_correspondence=splat_camera_correspondence,
                                   stride=STRIDE,
                                   starting_index=STARTING_INDEX,
                                   sam_level=sam_level)
        
        return expanded_masks
