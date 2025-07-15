# Multi-view SAM mask refinement imports and utilities
from collections import defaultdict
import torch
from tqdm import tqdm
import math
import numpy as np

class MultiViewSAMMaskRefiner:
    """Refines SAM masks by enforcing consistency across overlapping views"""
    
    def __init__(self, overlap_threshold=0.3, consensus_strategy="majority_vote"):
        self.overlap_threshold = overlap_threshold
        self.consensus_strategy = consensus_strategy
        
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
    
    def project_3d_to_2d(self, point_3d, camera):
        """Project 3D point to 2D camera coordinates"""
        # Use OpenGaussian's existing camera projection
        world_view_transform = camera.world_view_transform
        full_proj_transform = camera.full_proj_transform
        
        # Convert to homogeneous coordinates
        point_4d = torch.cat([point_3d, torch.ones(1, device=point_3d.device)])
        
        # Apply transformations
        point_cam = world_view_transform @ point_4d
        point_proj = full_proj_transform @ point_4d
        
        # Perspective divide
        if point_proj[3] != 0:
            point_ndc = point_proj[:3] / point_proj[3]
            
            # Convert from NDC [-1,1] to pixel coordinates
            H, W = camera.image_height, camera.image_width
            x_pixel = int((point_ndc[0] + 1) * W / 2)
            y_pixel = int((point_ndc[1] + 1) * H / 2)
            
            # Check if point is within image bounds
            if 0 <= x_pixel < W and 0 <= y_pixel < H:
                return x_pixel, y_pixel, point_cam[2] > 0  # Return x, y, is_in_front
                
        return None, None, False
    
    def collect_mask_votes_for_point(self, point_3d, cameras, sam_masks, sam_level=0):
        """Collect mask ID votes for a 3D point from all visible cameras"""
        votes = []
        
        for cam_idx, camera in enumerate(cameras):
            x, y, is_visible = self.project_3d_to_2d(point_3d, camera)
            
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
            
            # For each sampled Gaussian, collect votes from overlapping cameras only
            for gaussian_idx in range(0, num_gaussians, sample_step):
                gaussian_3d = gaussians.get_xyz[gaussian_idx]  # Actual 3D position
                
                # Project this Gaussian only to overlapping cameras
                votes = []
                for other_cam_idx in overlapping_cam_indices:
                    other_camera = cameras[other_cam_idx]
                    x, y, visible = self.project_3d_to_2d(gaussian_3d, other_camera)
                    if visible and x is not None and y is not None:
                        mask_id = sam_masks[other_cam_idx][sam_level, y, x].item()
                        votes.append((other_cam_idx, mask_id))
                
                # Apply consensus and update the current camera's mask
                if votes:
                    consensus_id = self.apply_consensus_rule(votes)
                    # Project Gaussian to current camera and update mask in local neighborhood
                    x_curr, y_curr, visible_curr = self.project_3d_to_2d(gaussian_3d, camera)
                    if visible_curr and consensus_id >= 0 and x_curr is not None and y_curr is not None:
                        # Update a small neighborhood around the projected point
                        H, W = refined_mask.shape[1], refined_mask.shape[2]
                        for dy in range(-5, 6):  # 11x11 neighborhood
                            for dx in range(-5, 6):
                                new_y, new_x = y_curr + dy, x_curr + dx
                                if 0 <= new_y < H and 0 <= new_x < W:
                                    refined_mask[sam_level, new_y, new_x] = consensus_id
            
            refined_masks.append(refined_mask)
        
        return refined_masks