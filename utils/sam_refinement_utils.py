# Multi-view SAM mask refinement imports and utilities
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

import time
from scene.gaussian_model import GaussianModel
from ashawkey_diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

# Utils


def save_weight_map_to_csv(weight_map: torch.Tensor, filename: str = "weight_map.csv"):
    """
    Save a weight map to a CSV file.

    Args:
        weight_map (torch.Tensor): Weight map with shape [H, W, 1] or [H, W] as torch.Tensor
        filename (str): Output CSV filename
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


def fix_image(rendered_image: torch.Tensor) -> torch.Tensor:
    """
    Fix image format for vizualisation.

    Args:
        rendered_image (torch.Tensor): in various formats

    Returns:
        torch.Tensor: Fixed image in [H, W, 3] format with uint8 dtype
    """

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
        matrix (torch.Tensor): Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        torch.Tensor: quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """

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

    def _standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
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

    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
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
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = _standardize_quaternion(out)

    return out


def log_camera_pose(
    log_name: str,
    translation: np.ndarray,
    rotation_q: np.ndarray,
    intrinsics: np.ndarray,
    frame_width: int = 518,
    frame_height: int = 294,
    image=None,
):
    """
    Log camera (pose + intrinsics) into rerun viewer.

    Args:
        log_name (str): rerun space to append camera pose to (e.g. same as pcl space name)
        translation (np.ndarray): 1x3 translation vector
        rotation_q (np.ndarray): 1x4 rotation vector (quaternion)
        intrinsics (np.ndarray): 3x3 K matrix
        frame_width, frame_height (int): image resolution
        image (np.ndarray): image to place inside of a frustum
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
    """Class to refine SAM masks by enforcing their consistency across "overlapping" viewpoints"""

    def __init__(self, verbose_logging=False):
        self.verbose_logging = verbose_logging

        # Member holding the current maximum segment ID across all segmented images
        self.current_max_id = 0

        self.cameras = []
        self.gaussians = None

    def rgb_to_weight_map(self, rendered_image: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB splat render to a weight map
        (assuming that a single splat with white viewpoint-invariant color is rendered).

        Args:
            rendered_image (torch.Tensor): RGB image with shape [H, W, 3]

        Returns:
            weight_map (torch.Tensor): Single channel weight map with shape [H, W, 1] as torch.Tensor
        """

        original_device = rendered_image.device

        # Convert to float if it's uint8
        if rendered_image.dtype == torch.uint8:
            rendered_image = rendered_image.float() / 255.0
        elif rendered_image.max() > 1.0:
            rendered_image = rendered_image / 255.0

        # Average RGB channels
        weight_map = torch.mean(rendered_image, dim=2)

        # Normalize by maximum value to ensure max is exactly 1.0
        max_val = weight_map.max()
        if max_val > 0:  # Avoid division by zero
            weight_map = weight_map / max_val

        # Add singleton dimension to get [H, W, 1]
        weight_map = weight_map.unsqueeze(2).to(original_device)

        return weight_map

    def create_consistent_id_mapping(
        self, masks: list[torch.Tensor]
    ) -> tuple[dict, list]:
        """
        Create a consistent mapping from original IDs to consecutive IDs (1, 2, 3, ...)
        across all masks.

        Args:
            masks (list): List of masks, each with shape [H, W] containing segment IDs

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

        for new_id, old_id in enumerate(sorted_ids, 1):
            id_mapping[old_id] = new_id

        # Add mapping for invalid IDs (preserve them)
        id_mapping[0] = 0
        id_mapping[-1] = -1

        print(f"Created mapping for {len(sorted_ids)} unique IDs")

        # Apply mapping to all masks
        remapped_masks = []
        for mask in masks:
            if mask is None:
                remapped_masks.append(None)
                continue

            # Create remapped mask
            if isinstance(mask, torch.Tensor):
                remapped_mask = torch.zeros_like(mask)
                for old_id, new_id in id_mapping.items():
                    remapped_mask[mask == old_id] = new_id

            remapped_masks.append(remapped_mask)

        return id_mapping, remapped_masks

    def render_single_gaussian(
        self,
        cam_idx: int,
        gaussian_idx: int,
        scaling_modifier=1.0,
        use_view_inv_white_shs: bool = False,
        **kwargs,
    ):
        """
        Render only a single Gaussian splat at the specified index from the model.

        This method is inspired by gaussian_renderer.render method and isolates a single
        Gaussian for individual rendering to analyze its contribution to the scene.

        Args:
            cam_idx (int): Index of the camera to use for rendering
            gaussian_idx (int): Index of the specific Gaussian to render (0-based)
            scaling_modifier (float, optional): Scaling factor applied to Gaussian sizes. Defaults to 1.0
            use_view_inv_white_shs (bool, optional): If True, uses view-invariant white spherical harmonics
            instead of the Gaussian's original SH coefficients. Defaults to False
            **kwargs: Additional keyword arguments, including:
            - override_color: Optional color override for the Gaussian

        Returns:
            tuple: A 4-tuple containing:
            - rendered_image (torch.Tensor): RGB image of shape [3, H, W] with the rendered Gaussian
            - radii (torch.Tensor): Screen-space radii of the rendered Gaussian
            - rendered_depth (torch.Tensor): Depth map of shape [1, H, W]
            - rendered_alpha (torch.Tensor): Alpha/opacity map of shape [1, H, W]
        """
        bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        viewpoint_camera = self.cameras[cam_idx]

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
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # Select only the single Gaussian at gaussian_idx
        means3D = self.gaussians.get_xyz[gaussian_idx : gaussian_idx + 1]
        means2D = torch.zeros_like(
            means3D, dtype=means3D.dtype, requires_grad=True, device="cuda"
        )
        try:
            means2D.retain_grad()
        except Exception:
            pass

        opacity = self.gaussians.get_opacity[gaussian_idx : gaussian_idx + 1]
        scales = self.gaussians.get_scaling[gaussian_idx : gaussian_idx + 1]
        rotations = self.gaussians.get_rotation[gaussian_idx : gaussian_idx + 1]
        shs = self.gaussians.get_features[gaussian_idx : gaussian_idx + 1]

        # If you have precomputed colors, handle here (optional)
        colors_precomp = None
        if kwargs.get("override_color") is not None:
            colors_precomp = kwargs["override_color"]
            if colors_precomp.ndim == 2:
                colors_precomp = colors_precomp[gaussian_idx : gaussian_idx + 1]
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
            cov3D_precomp=None,
        )
        return rendered_image, radii, rendered_depth, rendered_alpha

    def render_gaussians_with_exclusion(
        self, cam_idx: int, exclude_indices=None, scaling_modifier=1.0, **kwargs
    ):
        """
        Render all Gaussians except those specified in exclude_indices.
        If exclude_indices is empty or None, render all Gaussians.

        This method is inspired by gaussian_renderer.render method.

        Args:
            cam_idx (int): Index of the camera to use for rendering
            exclude_indices (list or None, optional): List of Gaussian indices to exclude
                from rendering. If None or empty, renders all Gaussians. Defaults to None.
            scaling_modifier (float, optional): Scaling factor applied to Gaussian sizes
                during rasterization. Defaults to 1.0.
            **kwargs: Additional keyword arguments passed to the rasterizer, including:
                - override_color: Optional color override for rendered Gaussians

        Returns:
            rendered_image, radii, rendered_depth, rendered_alpha
        """
        bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        viewpoint_camera = self.cameras[cam_idx]

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
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # Determine indices to render
        all_indices = list(range(self.gaussians.get_xyz.shape[0]))
        if exclude_indices:
            render_indices = [i for i in all_indices if i not in exclude_indices]
        else:
            render_indices = all_indices

        # Select Gaussians to render
        means3D = self.gaussians.get_xyz[render_indices]
        means2D = torch.zeros_like(
            means3D, dtype=means3D.dtype, requires_grad=True, device="cuda"
        )
        try:
            means2D.retain_grad()
        except Exception:
            pass

        opacity = self.gaussians.get_opacity[render_indices]
        scales = self.gaussians.get_scaling[render_indices]
        rotations = self.gaussians.get_rotation[render_indices]
        shs = self.gaussians.get_features[render_indices]

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
            cov3D_precomp=None,
        )
        return rendered_image, radii, rendered_depth, rendered_alpha

    def project_3d_points_to_image(
        self, point_3d, camera: Camera, index_gs=0, index_cam=0
    ):
        """
        Original version of project_3d_points_to_image_batch for a single point (not up-to-date).
        """

        # Convert points to homogeneous coordinates (add a fourth '1' coordinate)
        point_3d_homogeneous = torch.cat(
            [point_3d, torch.tensor([1.0], device=point_3d.device)]
        )

        # Get coordinate of queried point in camera space (direct transform)
        point_camera = camera.world_view_transform_no_t @ point_3d_homogeneous

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

        return (
            int(u),
            int(v),
            (
                0 <= u < camera.image_width
                and 0 <= v < camera.image_height
                and is_in_front
            ),
        )

    def project_3d_points_to_image_batch(
        self, cam_idx: int, gaussian_indices=None, use_depth=False
    ):
        """
        Project batch of 3D Gaussian points from world space to camera space and determine visibility.

        This method transforms 3D Gaussian points from world space to camersa pixel coords and determines
        visibility by comparing an euclidean distance to the point in 3D space to the rendered depth.

        Args:
            cam_idx (int): Index of the camera to use for projection
            gaussian_indices (array-like, optional): Indices of specific Gaussians to project.
                If None, projects all Gaussians. Used for selective processing and logging.
            use_depth (bool, optional): Whether to perform depth map validation for visibility
                determination. When True, compares projected point depths with rendered depth map
                values using a configurable threshold (DEPTH_DIFF_THRESHOLD = 0.15m). Defaults to False.

        Returns:
            tuple: A 3-tuple containing:
                - u (torch.Tensor): Pixel x-coordinates of shape (N,) where N is the number of points
                - v (torch.Tensor): Pixel y-coordinates of shape (N,) where N is the number of points
                - visible (torch.Tensor): Boolean mask of shape (N,) indicating which points are visible.
                    Points are considered visible if they:
                    - Are in front of the camera (z > 0 in camera space)
                    - Fall within image bounds (0 <= u < width, 0 <= v < height)
                    - Pass depth validation (if use_depth=True)
        """

        camera = self.cameras[cam_idx]

        points_3d = (
            self.gaussians.get_xyz[gaussian_indices]
            if gaussian_indices is not None
            else self.gaussians.get_xyz
        )
        N = points_3d.shape[0]

        # Convert points to homogeneous coordinates (add a fourth '1' coordinate)
        ones = torch.ones(N, 1, device=points_3d.device)
        points_3d_homogeneous = torch.cat([points_3d, ones], dim=1)  # (N, 4)

        # Get coordinates in camera space (batch matrix multiplication)
        points_camera = (
            camera.world_view_transform_no_t @ points_3d_homogeneous.T
        ).T  # (N, 4)

        # Logging only for first point if enabled and batch size is 1
        if self.verbose_logging and N == 1 and gaussian_indices is not None:
            rr.log(
                f"gs_{gaussian_indices[0]}/camera_{cam_idx}/camera_pose/gs_in_cam",
                rr.Points3D(points_camera[0, :3].cpu(), radii=0.01, colors=[0, 0, 255]),
            )

        # Apply projection matrix to map to clip space
        points_clip = (camera.projection_matrix_no_t @ points_camera.T).T  # (N, 4)

        # Perspective division
        w = points_clip[:, 3]  # (N,)
        # Avoid division by zero
        w = torch.where(torch.abs(w) < 1e-8, torch.sign(w) * 1e-8, w)
        points_ndc = points_clip / w.unsqueeze(1)  # (N, 4)

        # Viewport transformation
        u = points_ndc[:, 0] * (camera.image_width / 2.0) + camera.cx  # (N,)
        v = points_ndc[:, 1] * (camera.image_height / 2.0) + camera.cy  # (N,)

        # Check bounds
        in_bounds = (
            (u >= 0) & (u < camera.image_width) & (v >= 0) & (v < camera.image_height)
        )

        # Check if points are in front of the camera (z > 0 in camera space)
        is_in_front = points_camera[:, 2] > 0  # (N,)
        visible = is_in_front & in_bounds  # (N,)

        DEPTH_DIFF_THRESHOLD = 0.15  # 15cm
        DIST_OPTICAL_CENTER_IMAGE = 0.1  # 10cm (took from rerun image metadata)
        if use_depth == True:
            # print(f"Visibility before depth check: {visible}")
            u_int = u.cpu().numpy().astype(int)
            v_int = v.cpu().numpy().astype(int)

            # TODO: currently setting pixel coordinates to zero that don't project into an image plane to
            # avoid having negative entries for depth map querying, fix
            u_int[~visible.cpu().numpy()] = 0
            v_int[~visible.cpu().numpy()] = 0

            cam2world = np.linalg.inv(camera.world_view_transform_no_t.cpu().numpy())
            t = cam2world[:3, 3]
            diff = points_3d.cpu().numpy() - t  # shape: (N, 3)
            dist_image_point = (
                np.linalg.norm(diff, axis=1) - DIST_OPTICAL_CENTER_IMAGE
            )  # shape: (N,)
            rendered_depth = camera.depth_map[:, v_int, u_int].detach()
            visible_depth = (
                torch.abs(
                    torch.from_numpy(dist_image_point).to(rendered_depth.device)
                    - rendered_depth
                )
                < DEPTH_DIFF_THRESHOLD
            )
            visible = visible & visible_depth

            if self.verbose_logging:
                print(
                    f"Euclidean distance (from aprox image plane) = {dist_image_point}, shape: {dist_image_point.shape}"
                )
                print(
                    f"Rendered depth (u={int(u[0])},v={int(v[0])}) = {rendered_depth}, shape: {rendered_depth.shape}"
                )  # shape 1
                print(
                    f"After depth chekcing: {visible_depth}, total verdict: {visible}"
                )

            if self.verbose_logging == True and visible.flatten().any() == True:
                image = camera.original_image.cpu().numpy().transpose(1, 2, 0)
                all_splat_masks = np.zeros_like(image[:, :, 0])
                for i, gaussian_idx in enumerate(gaussian_indices):
                    if visible.flatten()[i] == True:
                        rendered_image, _, _, _ = self.render_single_gaussian(
                            cam_idx=cam_idx,
                            gaussian_idx=gaussian_idx,
                            use_view_inv_white_shs=True,
                        )
                        rendered_image = fix_image(rendered_image)

                        weight_map = self.rgb_to_weight_map(rendered_image)
                        save_weight_map_to_csv(weight_map, filename="test.csv")

                        # Get mask for the single gaussian
                        non_black_mask = np.any(rendered_image != 0, axis=2)

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
                image[all_splat_masks] = (227, 114, 34)
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

    def get_most_common_id_in_mask_weighted(
        self, sam_mask: torch.Tensor, weight_matrix: torch.Tensor
    ) -> int:
        """
        Find the most dominant ID in an image based on weighted counts.

        Args:
            sam_mask: torch.Tensor with shape [H, W]
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
        weighted_counts = torch.bincount(
            shifted_ids.long(), weights=weights_flat, minlength=int(max_id + offset) + 1
        )

        # Find the ID with maximum weighted count
        most_common_shifted_id = torch.argmax(weighted_counts).item()

        # Shift back to original ID space
        most_common_id = most_common_shifted_id - offset
        max_weight = weighted_counts[most_common_shifted_id].item()

        if self.verbose_logging:
            print(
                f"Most dominant ID: {most_common_id} with weighted count: {max_weight:.3f}"
            )

        return int(most_common_id)

    def visualize_results(
        self,
        original_sam_masks: list[torch.Tensor],
        modified_sam_masks: list[torch.Tensor],
        gaussian_indices: torch.Tensor,
        splat_camera_correspondence: torch.Tensor,
        sam_level: int = 0,
        orig_stride: int = 1,
        orig_starting_idx: int = 0,
        orig_ending_idx: int = -1,
        num_points_to_viz: int = 10,
    ):
        """
        Method to aggregate results to vizualize.
        """

        rr.init("Sam_Refinement_Multistage", spawn=True)
        rr.log(
            "world_frame",
            rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )
        rr.log(
            f"gaussian_pointcloud",
            rr.Points3D(self.gaussians.get_xyz.cpu(), radii=0.005, colors=[0, 255, 0]),
        )
        if gaussian_indices is not None:
            strided_indices_orig = []
            for i in range(orig_starting_idx, len(gaussian_indices), orig_stride):
                strided_indices_orig.append(i)
            gaussian_indices_orig_stride = [
                gaussian_indices[id].item() for id in strided_indices_orig
            ]
            if orig_ending_idx != -1:
                gaussian_indices_orig_stride = [
                    idx for idx in gaussian_indices_orig_stride if idx < orig_ending_idx
                ]
            rr.log(
                f"selected_splats",
                rr.Points3D(
                    self.gaussians.get_xyz[gaussian_indices_orig_stride].cpu(),
                    radii=0.02,
                    colors=[255, 0, 0],
                ),
            )

        strided_indices = []
        for i in range(
            0, len(gaussian_indices), len(gaussian_indices) // num_points_to_viz
        ):
            strided_indices.append(i)
        num_gaussians = self.gaussians.get_xyz.shape[0]
        gaussian_viz_data = []

        for filtered_idx in strided_indices:
            # Get the correspondence row for this filtered index
            splat_visibility_in_cams = splat_camera_correspondence[filtered_idx]

            # Map to the actual gaussian ID (global, w/o opacity filtering - for indexing other methods with it)
            gaussian_id = gaussian_indices[filtered_idx].item()

            if gaussian_id < num_gaussians:
                for cam_idx, camera in enumerate(self.cameras):
                    if splat_visibility_in_cams[cam_idx]:
                        # Re-render to get fresh data
                        rendered_image, _, _, _ = self.render_single_gaussian(
                            cam_idx=cam_idx, gaussian_idx=gaussian_id
                        )
                        rendered_image = fix_image(rendered_image)
                        non_black_mask = torch.any(rendered_image != 0, dim=2)

                        # if not non_black_mask.any():
                        #     if self.verbose_logging:
                        #         print(f"Skipping splat {gaussian_id} - projected to image no non-black pixels")
                        #     continue

                        # Get the UPDATED state of refined masks (after this update)
                        sam_mask_channel_after = (
                            modified_sam_masks[cam_idx][sam_level].cpu().numpy()
                        )
                        rendered_ids_after = (
                            sam_mask_channel_after
                            * non_black_mask.cpu().numpy().astype(int)
                        )

                        gaussian_viz_data.append(
                            {
                                "camera": camera,
                                "camera_idx": cam_idx,
                                "gaussian_id": gaussian_id,
                                "rendered_ids": rendered_ids_after,  # Use AFTER data
                                "sam_mask_channel": sam_mask_channel_after,  # Use AFTER data
                                "rendered_image": rendered_image.cpu().numpy(),
                                "non_black_mask": non_black_mask.cpu().numpy(),
                            }
                        )
        if gaussian_viz_data:
            cam_number_for_rerun = 0
            splat_id_prev_for_rerun = -1
            for viz_data in gaussian_viz_data:
                # print(f"Debug: gaussian id shifted {viz_data['gaussian_id']}")
                if splat_id_prev_for_rerun != viz_data["gaussian_id"]:
                    input(
                        f"{cam_number_for_rerun} frames in prev log. "
                        f"Press to log results for splat {viz_data['gaussian_id']}"
                    )
                    cam_number_for_rerun = 0
                    splat_id_prev_for_rerun = viz_data["gaussian_id"]

                    rr.log(f"stage2", rr.Clear(recursive=True))

                rr.log(
                    f"current_splat",
                    rr.Points3D(
                        self.gaussians.get_xyz[viz_data["gaussian_id"]].cpu(),
                        radii=0.1,
                        colors=[0, 0, 255],
                    ),
                )
                self.plot_masks(
                    viz_data["camera"],
                    viz_data["rendered_ids"],
                    viz_data["gaussian_id"],
                    viz_data["sam_mask_channel"],
                    viz_data["rendered_image"],
                    viz_data["non_black_mask"],
                    viz_data["camera_idx"],
                    original_sam_masks,
                    cam_number_for_rerun,
                )
                cam_number_for_rerun += 1

    def plot_masks(
        self,
        camera,
        rendered_ids,
        gaussian_id,
        sam_mask_channel,
        rendered_image,
        non_black_mask,
        camera_idx,
        original_sam_masks,
        cam_num_for_rerun,
    ):
        """
        Method to vizualize results.
        """

        PLOT_USING_MATPLOTLIB = False

        # Use the same color scheme as in train.py
        np.random.seed(42)
        colors_defined = np.random.randint(100, 256, size=(300, 3))
        colors_defined[0] = np.array(
            [0, 0, 0]
        )  # Ignore the mask ID of -1 and set it to black.
        colors_defined = torch.from_numpy(colors_defined)

        # Create visualizations
        if PLOT_USING_MATPLOTLIB:
            plt.figure(figsize=(30, 5))

        # 1. Original SAM mask - using predefined colors
        original_img = original_sam_masks[camera_idx][0].cpu().numpy()
        original_mask_colored = (
            colors_defined[original_img.astype(int).clip(0, 299)]
            .numpy()
            .astype(np.uint8)
        )
        original_mask_colored = original_mask_colored.transpose(
            2, 0, 1
        )  # [H, W, 3] -> [3, H, W]
        rr.log(
            f"stage2/orig_sam_mask_cam_{cam_num_for_rerun}",
            rr.Image(original_mask_colored.transpose(1, 2, 0)).compress(
                jpeg_quality=75
            ),
        )
        if PLOT_USING_MATPLOTLIB:
            plt.subplot(1, 6, 1)
            plt.imshow(
                original_mask_colored.transpose(1, 2, 0)
            )  # [3, H, W] -> [H, W, 3]
            plt.title(f"Original SAM Mask - Camera {camera_idx}")
            plt.axis("off")

        # 2. Refined SAM mask - using predefined colors
        refined_img = sam_mask_channel  # This IS the refined mask!
        refined_mask_colored = (
            colors_defined[refined_img.astype(int).clip(0, 299)]
            .numpy()
            .astype(np.uint8)
        )
        refined_mask_colored = refined_mask_colored.transpose(
            2, 0, 1
        )  # [H, W, 3] -> [3, H, W]
        rr.log(
            f"stage2/refined_sam_mask_cam_{cam_num_for_rerun}",
            rr.Image(refined_mask_colored.transpose(1, 2, 0)).compress(jpeg_quality=75),
        )
        if PLOT_USING_MATPLOTLIB:
            plt.subplot(1, 6, 2)
            plt.imshow(
                refined_mask_colored.transpose(1, 2, 0)
            )  # [3, H, W] -> [H, W, 3]
            plt.title(f"Refined SAM Mask - Camera {camera_idx}")
            plt.axis("off")

        # 3. Rendered IDs - using predefined colors
        rendered_mask_colored = (
            colors_defined[rendered_ids.astype(int).clip(0, 299)]
            .numpy()
            .astype(np.uint8)
        )
        rendered_mask_colored = rendered_mask_colored.transpose(
            2, 0, 1
        )  # [H, W, 3] -> [3, H, W]
        rr.log(
            f"stage2/splat_in_cam_{cam_num_for_rerun}",
            rr.Image(rendered_mask_colored.transpose(1, 2, 0)).compress(
                jpeg_quality=75
            ),
        )
        if PLOT_USING_MATPLOTLIB:
            plt.subplot(1, 6, 3)
            plt.imshow(
                rendered_mask_colored.transpose(1, 2, 0)
            )  # [3, H, W] -> [H, W, 3]
            plt.title(f"Rendered IDs - Gaussian {gaussian_id}")
            plt.axis("off")

            # 4. Non-black mask (Gaussian footprint binary)
            plt.subplot(1, 6, 4)
            plt.imshow(non_black_mask, cmap="gray")
            plt.title(f"Non-black Mask\n(Gaussian Footprint)")
            plt.axis("off")

            # 5. Gaussian footprint (original rendered image)
            plt.subplot(1, 6, 5)
            plt.imshow(rendered_image)
            plt.title(f"Gaussian Footprint\n(Rendered Image)")
            plt.axis("off")

        # # 6. Difference between original and refined masks
        difference = np.abs(
            original_img.astype(np.float32) - refined_img.astype(np.float32)
        )
        # Convert difference to 3-channel image for rerun
        if difference.ndim == 2:
            # Convert grayscale difference to 3-channel RGB
            difference_rgb = np.stack([difference, difference, difference], axis=2)
        else:
            difference_rgb = difference
        # Convert to uint8 for JPEG compression
        if difference_rgb.dtype != np.uint8:
            # Normalize to 0-255 range and convert to uint8
            difference_rgb = np.clip(difference_rgb * 255, 0, 255).astype(np.uint8)
        rr.log(
            f"stage2/diff_in_cam_{cam_num_for_rerun}",
            rr.Image(difference_rgb).compress(jpeg_quality=75),
        )
        if PLOT_USING_MATPLOTLIB:
            plt.subplot(1, 6, 6)

            plt.imshow(difference, cmap="hot", vmin=0, vmax=10)
            plt.title(f"Mask Difference\n(Original vs Refined)")
            plt.colorbar(label="ID Difference")
            plt.axis("off")

            plt.suptitle(f"Gaussian {gaussian_id} in Camera {camera_idx}", fontsize=16)
            plt.tight_layout()
            plt.show()

        # Print info
        unique_rendered_ids = np.unique(
            rendered_ids[rendered_ids > -100], return_counts=True
        )
        unique_original_ids = np.unique(
            original_img[original_img > -100], return_counts=True
        )
        unique_refined_ids = np.unique(
            refined_img[refined_img > -100], return_counts=True
        )

        print(f"Gaussian {gaussian_id}, Camera {camera_idx}:")
        print(f"  Original SAM mask segmentation IDs: {unique_original_ids}")
        print(f"  Refined SAM mask segmentation IDs: {unique_refined_ids}")
        print(f"  Splat segmentation IDs: {unique_rendered_ids}")
        print(f"  Footprint pixels: {np.count_nonzero(non_black_mask)}")
        print(f"  Changed pixels: {np.count_nonzero(difference > 0)}")
        print("-" * 50)

    def get_splat_id_and_weights(
        self, cam_idx: int, gaussian_id: int, sam_mask: torch.Tensor
    ) -> tuple[int, torch.Tensor, bool]:

        rendered_image, _, _, _ = self.render_single_gaussian(
            cam_idx=cam_idx, gaussian_idx=gaussian_id, use_view_inv_white_shs=True
        )
        rendered_image = fix_image(rendered_image)  # Convert to proper format
        non_black_mask = torch.any(rendered_image != 0, dim=2)
        weights_mask = self.rgb_to_weight_map(rendered_image)
        most_dominant_id = self.get_most_common_id_in_mask_weighted(
            sam_mask=sam_mask, weight_matrix=weights_mask
        )

        return (most_dominant_id, weights_mask, non_black_mask.any())

    def init_cam_id_counts(self, cam_idx: int, refined_mask: torch.Tensor):
        device = refined_mask.device
        H, W = refined_mask.shape

        camera = self.cameras[cam_idx]

        # Get all unique IDs in this image
        camera.unique_ids = torch.unique(refined_mask, sorted=True)
        num_ids = len(camera.unique_ids)

        # Create a mapping from ID values to indices (0, 1, 2, ...), find channel in tensor for an ID
        camera.id_to_idx = {
            id_val.item(): idx for idx, id_val in enumerate(camera.unique_ids)
        }

        # Create a 3D tensor [H, W, num_ids] where each "channel" represents one ID
        # This replaces the nested dictionary structure
        camera.pixel_value_tensor = torch.full(
            (H, W, num_ids), 0.0, dtype=torch.float32, device=device
        )

        # Initialize counts of IDs of pixels in refined_masks to 1.0
        for id_val, idx in camera.id_to_idx.items():
            # Skip initialization for invalid/background ID
            if id_val == -1:
                continue

            # Set pixels that belong to this ID to 1.0
            mask_for_id = refined_mask == id_val
            y_indices, x_indices = torch.where(mask_for_id)
            camera.pixel_value_tensor[y_indices, x_indices, idx] = 1.0

        camera.id_range = (
            camera.unique_ids.min().item(),
            camera.unique_ids.max().item(),
        )

    def expand_splat(
        self, weight_map: torch.Tensor, object_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expands the splat region by identifying pixels with non-zero weights that are not part of the main object mask.
        Args:
            weight_map (torch.Tensor): A tensor of shape [H, W] or [H, W, 1] representing the weight map of the splat).
            object_mask (torch.Tensor): A boolean tensor of shape [H, W] indicating the object segment the splat relates to.
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - ext_y_indices (torch.Tensor): 1D tensor containing the y-coordinates of the extension pixels.
                - ext_x_indices (torch.Tensor): 1D tensor containing the x-coordinates of the extension pixels.
                - extension_weights (torch.Tensor): 1D tensor containing the weight values for the extension pixels.
        """

        if weight_map.ndim == 3 and weight_map.shape[2] == 1:
            weight_map = weight_map.squeeze(
                2
            )  # Remove the singleton dimension [H, W, 1] -> [H, W]

        # Get the weight matrix and create mask for non-zero weights
        non_zero_weight_mask = weight_map > 0  # Pixels covered by the Gaussian

        # Find pixels that have non-zero weight BUT are NOT part of the main segment
        extension_mask = non_zero_weight_mask & (~object_mask)

        if extension_mask.any():
            # Get the weight values for pixels in the extension area
            extension_weights = weight_map[extension_mask]

            # Get y, x indices for extension pixels
            y_indices, x_indices = torch.where(extension_mask)

            return (y_indices, x_indices, extension_weights)
        return (None, None, None)

    def expand_masks(
        self,
        gaussian_id: int,
        sam_masks: list[torch.Tensor],
        cam_idx_splat_segment_id_weight_mask_pairs: list[int, int, torch.Tensor],
        sam_level: int = 0,
    ):
        """
        Expand Gaussian splat masks across camera views using voting and weight-based extension.

        This method determines the most dominant segment ID across multiple camera views through
        voting, then expands the corresponding splat masks by updating pixel value tensors with
        base mask counts and weighted extensions.

        Args:
            gaussian_id (int): ID of the current Gaussian splat being processed
            sam_masks (list[torch.Tensor]): List of SAM segmentation mask tensors for each camera view
            cam_idx_splat_segment_id_weight_mask_pairs (list[tuple[int, int, torch.Tensor]]):
                List of tuples containing:
                - camera_idx (int): Index of the camera
                - most_dominant_id (int): The dominant splat ID for this camera view
                - weight_map (torch.Tensor): Weight map for splat expansion
            sam_level (int, optional): Level/scale of SAM segmentation to use. Defaults to 0.
        """

        # Step 2: voting
        id_vote_counts = {}
        for (
            camera_idx,
            most_dominant_id,
            _,
        ) in cam_idx_splat_segment_id_weight_mask_pairs:
            if sam_masks[camera_idx] is None:
                continue

            # Count votes for this ID
            if most_dominant_id not in id_vote_counts:
                id_vote_counts[most_dominant_id] = 0
            id_vote_counts[most_dominant_id] += 1

        if id_vote_counts:
            winning_id = max(id_vote_counts, key=id_vote_counts.get)
            max_votes = id_vote_counts[winning_id]

            # print(
            #     f"Splat {gaussian_id}: winning ID: {winning_id} with {max_votes} votes out of {len(cam_idx_splat_segment_id_weight_mask_pairs)} cameras"
            # )
            # print(f"All vote counts: {id_vote_counts}")

            # Extend segments with projected splats
            for (
                camera_idx,
                most_dominant_id,
                weight_map,
            ) in cam_idx_splat_segment_id_weight_mask_pairs:
                if winning_id == most_dominant_id:
                    # Skip expanding over -1 (void)
                    if winning_id != -1:
                        object_mask_of_dominant_id = (
                            sam_masks[camera_idx][sam_level] == winning_id
                        )
                        idx = self.cameras[camera_idx].id_to_idx[winning_id]

                        # Update count of base mask (s1)
                        y_indices, x_indices = torch.where(object_mask_of_dominant_id)
                        self.cameras[camera_idx].pixel_value_tensor[
                            y_indices, x_indices, idx
                        ] += 1.0
                        # print(
                        #     f"Cam {camera_idx} counts: \n{self.cameras[camera_idx].pixel_value_tensor[y_indices, x_indices, idx]}"
                        # )

                        # Expand splat over winning mask (s2)
                        ext_y_indices, ext_x_indices, extension_weights = (
                            self.expand_splat(
                                weight_map=weight_map,
                                object_mask=object_mask_of_dominant_id,
                            )
                        )
                        if (
                            ext_y_indices is not None
                            and ext_x_indices is not None
                            and extension_weights is not None
                        ):
                            self.cameras[camera_idx].pixel_value_tensor[
                                ext_y_indices, ext_x_indices, idx
                            ] += extension_weights
                            # print(
                            #     f"Extended {len(ext_y_indices)} pixels for winning_id {winning_id} with weights"
                            # )
                else:
                    # continue
                    # TODO: figure out whether this works
                    # Check if winning_id is among image IDs and isn't equal to -1 (void)
                    if (
                        torch.any(
                            self.cameras[camera_idx].unique_ids == winning_id
                        ).item()
                        and winning_id != -1
                    ):
                        object_mask_of_dominant_id = (
                            sam_masks[camera_idx][sam_level] == winning_id
                        )
                        idx = self.cameras[camera_idx].id_to_idx[winning_id]

                        # Expand splat over winning mask (s2)
                        ext_y_indices, ext_x_indices, extension_weights = (
                            self.expand_splat(
                                weight_map=weight_map,
                                object_mask=object_mask_of_dominant_id,
                            )
                        )
                        if (
                            ext_y_indices is not None
                            and ext_x_indices is not None
                            and extension_weights is not None
                        ):
                            self.cameras[camera_idx].pixel_value_tensor[
                                ext_y_indices, ext_x_indices, idx
                            ] += extension_weights
                            # print(f"Extended {len(ext_y_indices)} pixels for winning_id {winning_id} (opposite direction) with weights")
                    else:
                        # Else do nothing with the splat
                        pass

    def sync_segment_ids(
        self,
        gaussian_id: int,
        cam_idx_splat_segment_id_pairs: list[int, int],
        sam_masks: list[torch.Tensor],
        sam_level: int = 0,
    ) -> list[torch.Tensor]:
        """
        Synchronize segment IDs across multiple camera views by assigning a new global ID.

        This method finds the segment ID with the most weight within a Gaussian's rendered footprint
        across all cameras and updates all pixels with that ID to a new globally unique ID
        for consistency across views.

        Args:
            gaussian_id (int): Unique identifier of the current Gaussian being processed
            cam_idx_splat_segment_id_pairs (list[tuple[int, int]]): List of tuples containing
                (camera_index, most_dominant_segment_id) pairs indicating which segment ID
                is most dominant in each camera view
            sam_masks (list[torch.Tensor]): List of SAM segmentation masks to be refined,
                where each tensor corresponds to a camera view. Can contain None values
                for cameras without masks
            sam_level (int, optional): SAM hierarchy level to operate on. Defaults to 0

        Returns:
            list[torch.Tensor]: Updated list of SAM masks with synchronized segment IDs.
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

        if self.verbose_logging:
            print(f"Generated new global ID: {new_global_id}")

        # Step 3: Update ALL pixels with the most common ID to the new global ID
        # Create a copy of sam_masks to avoid modifying the original
        refined_masks = [
            mask.clone() if mask is not None else None for mask in sam_masks
        ]

        for camera_idx, most_dominant_id in cam_idx_splat_segment_id_pairs:
            if refined_masks[camera_idx] is None:
                continue

            # Find ALL pixels with the most common ID in this camera's mask
            object_mask_of_dominant_id = (
                refined_masks[camera_idx][sam_level] == most_dominant_id
            )
            pixels_before = torch.sum(object_mask_of_dominant_id).item()

            if pixels_before > 0:
                # We don't overwrite 'void' classified objects, since there can be multiple different objects of class 'void' in the scene
                if most_dominant_id != -1:
                    # Update ALL pixels with the most common ID to the new global ID
                    refined_masks[camera_idx][sam_level][
                        object_mask_of_dominant_id
                    ] = new_global_id
                    if self.verbose_logging:
                        print(
                            f"Cam ID={camera_idx}, splat ID={gaussian_id}: {pixels_before} pixels changed from ID {most_dominant_id} to {new_global_id}"
                        )
                else:
                    if self.verbose_logging:
                        print(
                            f"Cam ID={camera_idx}, splat ID={gaussian_id}: skipping change to {new_global_id} ID, actual ID={most_dominant_id}"
                        )

        return refined_masks

    def refine_sam_masks(
        self,
        cameras: list[Camera],
        sam_masks: list[torch.Tensor],
        gaussians: GaussianModel,
        sam_level: int = 0,
    ) -> list[torch.Tensor]:
        """
        Refines SAM (Segment Anything Model) masks using 3D Gaussian splatting for cross-view consistency.

        This method performs a two-stage refinement process:
        1. Stage 1: Achieves cross-view consistent object IDs by synchronizing segment IDs across cameras
        2. Stage 2: Expands segment masks using rendered Gaussian splats to fill gaps and improve coverage

        Args:
            cameras (list[Camera]): List of camera objects containing pose and intrinsic parameters
            sam_masks (list[torch.Tensor]): List of SAM segmentation masks for each camera view
            gaussians (GaussianModel): 3D Gaussian splatting model containing point positions, opacities, etc.
            sam_level (int, optional): Hierarchical level of SAM masks to process. Defaults to 0.

        Returns:
            list[torch.Tensor]: List of refined segmentation masks with improved cross-view consistency.

        Process:
            - Renders depth maps for each camera view
            - Establishes splat-to-camera visibility correspondence
            - Stage 1: Synchronizes segment IDs across views for high-opacity Gaussians (strided sampling)
            - Stage 2: Expands masks by accumulating weights from all visible Gaussians
            - Rewrites pixel segment IDs with values containing highest accumulated weights
        """

        num_gaussians = gaussians.get_xyz.shape[0]
        gaussian_indices = range(0, num_gaussians, 1)
        self.cameras = cameras
        self.gaussians = gaussians

        ### PARAMETERS ###
        # Stage 1
        STARTING_INDEX_STAGE_1 = 0
        STRIDE_STAGE_1 = 250
        OPACITY_THRESHOLD_STAGE_1 = 0.5
        VIZUALIZE_STAGE_1 = False
        NUM_POINTS_TO_VIZ_STAGE_1 = 10
        # Stage 2
        STARTING_INDEX_STAGE_2 = 0
        ENDING_INDEX_STAGE_2 = len(gaussian_indices)
        STRIDE_STAGE_2 = 1
        # Minimal value of an accumulated weight in a pixel to overwrite it from void
        THRESHOLD_ACCUMULATED_WEIGHT_STAGE_2 = 0.5
        VIZUALIZE_STAGE_2 = True
        NUM_POINTS_TO_VIZ_STAGE_2 = 10
        ### END PARAMETERS ###

        # Initialize refined_masks as a copy of original sam_masks
        original_sam_masks = [
            mask.clone() if mask is not None else None for mask in sam_masks
        ]
        refined_masks = [
            mask.clone() if mask is not None else None for mask in sam_masks
        ]

        # Write depth maps to camera instances
        for cam_idx, camera in tqdm(
            enumerate(cameras),
            total=len(cameras),
            desc="Writing depth maps to camera frames",
        ):
            _, _, rendered_depth, _ = self.render_gaussians_with_exclusion(
                cam_idx=cam_idx, exclude_indices=None
            )
            camera.depth_map = rendered_depth

        splat_camera_correspondence = torch.empty(
            (
                (
                    len(gaussian_indices)
                    if gaussian_indices is not None
                    else num_gaussians
                ),
                len(cameras),
            ),
            dtype=torch.bool,
            device=gaussians.get_xyz.device,
        )

        if self.verbose_logging:
            rr.init("Sam_Refinement_Multistage", spawn=True)
            rr.log(
                "world_frame",
                rr.Arrows3D(
                    vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                ),
            )
            rr.log(
                f"gaussian_pointcloud",
                rr.Points3D(gaussians.get_xyz.cpu(), radii=0.005, colors=[0, 255, 0]),
            )
            if gaussian_indices is not None:
                rr.log(
                    f"selected_splats",
                    rr.Points3D(
                        gaussians.get_xyz[gaussian_indices].cpu(),
                        radii=0.01,
                        colors=[255, 0, 0],
                    ),
                )

        # Stage 0: detect visible splats in camera frames
        for cam_idx, camera in tqdm(
            enumerate(cameras),
            total=len(cameras),
            desc="Writing splat to cam correspondence",
        ):
            _, _, visible_curr = self.project_3d_points_to_image_batch(
                cam_idx=cam_idx, gaussian_indices=gaussian_indices, use_depth=True
            )
            if self.verbose_logging:
                input("Pause: press a key to continue")

            splat_camera_correspondence[:, cam_idx] = visible_curr
        print(
            f"Splat to camera correspondence of shape {splat_camera_correspondence.shape}"
        )

        # Stage 1: get cross-view consistent object IDs
        # TODO: consider add checking for variance ratio of principal axes

        # Filter out splats with low opacity
        high_opacity_gaussian_indices = torch.where(
            gaussians.get_opacity >= OPACITY_THRESHOLD_STAGE_1
        )[0]
        splat_camera_correspondence_high_opacity = splat_camera_correspondence[
            high_opacity_gaussian_indices
        ]
        print(
            f"Number not skipped splats for stage 1: {splat_camera_correspondence_high_opacity.shape[0]}"
        )
        # Apply stride to the filtered indices, not during iteration
        strided_indices = []
        for i in range(
            STARTING_INDEX_STAGE_1, len(high_opacity_gaussian_indices), STRIDE_STAGE_1
        ):
            strided_indices.append(i)

        print(
            f"Not skipped due to opacity: {splat_camera_correspondence_high_opacity.shape[0]}; "
            f"processing (with stride {STRIDE_STAGE_1}): {len(strided_indices)}"
        )
        # 1. Aggregate splat IDs and weights for mask expanding
        for filtered_idx in tqdm(
            strided_indices, desc="Updating masks for cross-view ID consistency"
        ):
            # Get the correspondence row for this filtered index
            splat_visibility_in_cams = splat_camera_correspondence_high_opacity[
                filtered_idx
            ]

            # Map to the actual gaussian ID (global, w/o opacity filtering - for indexing other methods with it)
            gaussian_id_stage_1 = high_opacity_gaussian_indices[filtered_idx].item()

            if gaussian_id_stage_1 < num_gaussians:
                cam_idx_splat_segment_id_pairs = (
                    []
                )  # contains tuples (camera_idx, most_dominant_id)
                # Collect mask data before updating
                for cam_idx, camera in enumerate(cameras):
                    if splat_visibility_in_cams[cam_idx]:
                        most_dominant_id, _, is_render_visible = (
                            self.get_splat_id_and_weights(
                                cam_idx=cam_idx,
                                gaussian_id=gaussian_id_stage_1,
                                sam_mask=sam_masks[cam_idx][sam_level],
                            )
                        )
                        if not is_render_visible:
                            if self.verbose_logging:
                                print(
                                    f"Skipping splat {gaussian_id_stage_1} - projected to image no non-black pixels"
                                )
                            continue
                        cam_idx_splat_segment_id_pairs.append(
                            (cam_idx, most_dominant_id)
                        )

                # 2. Synchronize segment IDs of same instances from crossing views.
                refined_masks = self.sync_segment_ids(
                    gaussian_id_stage_1,
                    cam_idx_splat_segment_id_pairs,
                    refined_masks,
                    sam_level,
                )

        # 3. Remap assigned IDs to avoid having big numbers
        id_mapping, refined_masks = self.create_consistent_id_mapping(refined_masks)
        print(f"Remapping:\n{id_mapping}")

        # VIZUALIZATION OF FIRST STAGE
        if VIZUALIZE_STAGE_1:
            self.visualize_results(
                original_sam_masks=original_sam_masks,
                modified_sam_masks=refined_masks,
                gaussian_indices=high_opacity_gaussian_indices,
                splat_camera_correspondence=splat_camera_correspondence_high_opacity,
                sam_level=sam_level,
                orig_stride=STRIDE_STAGE_1,
                orig_starting_idx=STARTING_INDEX_STAGE_1,
                num_points_to_viz=NUM_POINTS_TO_VIZ_STAGE_1,
            )

        # Stage 2: expand splats
        # 1. Init pixel lookup tables
        start_time = time.time()
        for cam_idx, camera in enumerate(cameras):
            if refined_masks[cam_idx] is None:
                continue

            self.init_cam_id_counts(
                cam_idx=cam_idx, refined_mask=refined_masks[cam_idx][sam_level]
            )
            if self.verbose_logging:
                print(
                    f"Cam {cam_idx}: unique_ids={camera.unique_ids}"
                    f"\npixel_values shape={camera.pixel_value_tensor.shape}"
                    f"\nid_to_idx={camera.id_to_idx}"
                    f"\nid_range={camera.id_range}"
                )

        end_time = time.time()
        print(f"Init duration pixel maps: {end_time - start_time}")

        # 2. Aggregate splat IDs and weights for mask expanding
        strided_indices = []
        for i in range(STARTING_INDEX_STAGE_2, ENDING_INDEX_STAGE_2, STRIDE_STAGE_2):
            strided_indices.append(i)

        debug_num_non_visible_renders = 0

        for filtered_idx in tqdm(
            strided_indices, desc="Expanding segments with rendered splats"
        ):
            # Get the correspondence row for this filtered index
            splat_visibility_in_cams = splat_camera_correspondence[filtered_idx]

            # If no filtering, global ID is directly the iteration index
            gaussian_id_stage_2 = filtered_idx

            debug_is_render_visible_in_any_cam = False

            if gaussian_id_stage_2 < num_gaussians:
                cam_idx_splat_segment_id_weight_mask_pairs = (
                    []
                )  # contains tuples (camera_idx, most_dominant_id, weight_mask)
                # Collect mask data before updating
                for cam_idx, camera in enumerate(cameras):
                    if splat_visibility_in_cams[cam_idx]:

                        most_dominant_id, weights_mask, is_render_visible = (
                            self.get_splat_id_and_weights(
                                cam_idx=cam_idx,
                                gaussian_id=gaussian_id_stage_2,
                                sam_mask=refined_masks[cam_idx][sam_level],
                            )
                        )
                        if not is_render_visible:
                            if self.verbose_logging:
                                print(
                                    f"Skipping splat {gaussian_id_stage_2} - projected to image no non-black pixels"
                                )
                            continue
                        debug_is_render_visible_in_any_cam = True

                        cam_idx_splat_segment_id_weight_mask_pairs.append(
                            (cam_idx, most_dominant_id, weights_mask)
                        )

                if debug_is_render_visible_in_any_cam:
                    debug_num_non_visible_renders += 1

            # 3. Expand masks
            self.expand_masks(
                gaussian_id=gaussian_id_stage_2,
                sam_masks=refined_masks,
                cam_idx_splat_segment_id_weight_mask_pairs=cam_idx_splat_segment_id_weight_mask_pairs,
                sam_level=sam_level,
            )

        print(f"Num non-visible splat renders: {debug_num_non_visible_renders}")

        # 4. Postprocessing: fetch ID of highest accumulated value and set it as pixel segment ID
        expanded_masks = [
            mask.clone() if mask is not None else None for mask in refined_masks
        ]
        for i, camera in enumerate(cameras):
            # Get max accumulated values and respective tensor indices (channels)
            max_values, indices_with_max_counts = torch.max(
                camera.pixel_value_tensor, dim=2
            )
            # Convert indices back to actual segment IDs using the unique_ids tensor
            expanded_mask = camera.unique_ids[indices_with_max_counts]

            # For initially 'void'-classified regions
            # (initialized to zero during weight accumulation)
            # only allow rewriting if accumulated weight is above certain threshold
            below_threshold = max_values < THRESHOLD_ACCUMULATED_WEIGHT_STAGE_2
            expanded_mask[below_threshold] = -1

            expanded_masks[i][sam_level] = expanded_mask

        # VIZUALIZATION OF SECOND STAGE
        if VIZUALIZE_STAGE_2:
            self.visualize_results(
                original_sam_masks=refined_masks,
                modified_sam_masks=expanded_masks,
                gaussian_indices=torch.tensor(gaussian_indices),
                splat_camera_correspondence=splat_camera_correspondence,
                sam_level=sam_level,
                orig_stride=STRIDE_STAGE_2,
                orig_starting_idx=STARTING_INDEX_STAGE_2,
                orig_ending_idx=ENDING_INDEX_STAGE_2,
                num_points_to_viz=NUM_POINTS_TO_VIZ_STAGE_2,
            )

        return expanded_masks
