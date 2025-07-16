import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
from scene import Scene
from argparse import ArgumentParser
from arguments import ModelParams
import random
from train_refined_sam_masks import MultiViewSAMMaskRefiner

class MultiViewRefinementVisualizer:
    """Visualize camera poses, FOVs, Gaussians, and SAM mask refinement process"""
    
    def __init__(self, source_path, output_dir="./visualization_output"):
        self.source_path = source_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Auto-detect if this is a dataset directory or training output directory
        self.dataset_path, self.model_path = self._detect_paths(source_path)
        
        # Load scene data
        parser = ArgumentParser()
        lp = ModelParams(parser)
        
        # Build arguments list - always use dataset path for scene, model path for loading
        args_list = ['-s', self.dataset_path]
        if self.model_path:
            args_list.extend(['-m', self.model_path])
            
        args = parser.parse_args(args_list)
        self.dataset = lp.extract(args)
        
        # Load scene and gaussians
        from scene import GaussianModel
        self.gaussians = GaussianModel(self.dataset.sh_degree)
        
        # Load the scene - handle both dataset and training output directories
        if self.model_path:
            # Training output directory - load trained model
            self.scene = Scene(self.dataset, self.gaussians, load_iteration=-1)
        else:
            # Dataset directory - initialize without loading trained model
            self.scene = Scene(self.dataset, self.gaussians, load_iteration=None, shuffle=False)
        
        # Sample subset for visualization
        self.sample_cameras(max_cameras=8)
        self.sample_gaussians(max_gaussians=500)
        
    def _detect_paths(self, source_path):
        """Auto-detect whether source_path is dataset or training output directory"""
        
        # Check if it's a training output directory
        if (os.path.exists(os.path.join(source_path, 'point_cloud')) and 
            os.path.exists(os.path.join(source_path, 'cfg_args'))):
            
            print(f"🔍 Detected training output directory: {source_path}")
            
            # Read cfg_args to find original dataset path
            cfg_args_path = os.path.join(source_path, 'cfg_args')
            try:
                with open(cfg_args_path, 'r') as f:
                    content = f.read()
                    
                # Extract source_path from cfg_args (it's stored as a Namespace string)
                import re
                match = re.search(r"source_path='([^']+)'", content)
                if match:
                    original_dataset_path = match.group(1)
                    
                    # Check if original dataset path exists
                    if os.path.exists(original_dataset_path):
                        print(f"📂 Found original dataset: {original_dataset_path}")
                        return original_dataset_path, source_path
                    else:
                        print(f"⚠️  Original dataset path not found: {original_dataset_path}")
                        print("🔍 Trying to find dataset in current project structure...")
                        
                        # Try to find scene in current project structure
                        scene_name = os.path.basename(original_dataset_path)
                        possible_paths = [
                            f"/home/andrii/TUM/ml3dg/project/data/OpenGaussian/OpenGaussian/scannet/{scene_name}",
                            f"/home/andrii/TUM/ml3dg/project/data/opengaussian/{scene_name}",
                            f"/home/andrii/TUM/ml3dg/project/LangSplat/scenes/{scene_name}"
                        ]
                        
                        for path in possible_paths:
                            if os.path.exists(path):
                                print(f"✅ Found dataset at: {path}")
                                return path, source_path
                        
                        raise FileNotFoundError(f"Could not find dataset for scene: {scene_name}")
                else:
                    raise ValueError("Could not parse source_path from cfg_args")
                    
            except Exception as e:
                raise ValueError(f"Could not read cfg_args file: {e}")
                
        # Check if it's a dataset directory
        elif (os.path.exists(os.path.join(source_path, 'transforms_train.json')) or
              os.path.exists(os.path.join(source_path, 'images')) or
              os.path.exists(os.path.join(source_path, 'color'))):
            
            print(f"📁 Detected dataset directory: {source_path}")
            return source_path, None
            
        else:
            raise ValueError(f"Could not recognize directory type: {source_path}\n"
                           f"Expected either:\n"
                           f"- Training output (with point_cloud/ and cfg_args)\n"
                           f"- Dataset directory (with transforms_train.json, images/, or color/)")
        
    def sample_cameras(self, max_cameras=8):
        """Sample a subset of cameras for visualization, prioritizing those with SAM masks"""
        all_cameras = self.scene.getTrainCameras()
        
        # Filter cameras that have SAM masks
        cameras_with_sam = []
        cameras_without_sam = []
        
        for cam in all_cameras:
            if hasattr(cam, 'original_sam_mask') and cam.original_sam_mask is not None:
                cameras_with_sam.append(cam)
            else:
                cameras_without_sam.append(cam)
        
        print(f"Found {len(cameras_with_sam)} cameras with SAM masks, {len(cameras_without_sam)} without")
        
        # Prioritize cameras with SAM masks
        if len(cameras_with_sam) > 0:
            if len(cameras_with_sam) <= max_cameras:
                self.cameras = cameras_with_sam
            else:
                # Sample cameras with good spatial distribution from those with SAM masks
                indices = np.linspace(0, len(cameras_with_sam)-1, max_cameras, dtype=int)
                self.cameras = [cameras_with_sam[i] for i in indices]
        else:
            # Fallback to all cameras if none have SAM masks
            if len(all_cameras) <= max_cameras:
                self.cameras = all_cameras
            else:
                indices = np.linspace(0, len(all_cameras)-1, max_cameras, dtype=int)
                self.cameras = [all_cameras[i] for i in indices]
            
        print(f"Selected {len(self.cameras)} cameras for visualization")
        
        # Load camera data to GPU
        for cam in self.cameras:
            if not cam.data_on_gpu:
                cam.to_gpu()
                
    def sample_gaussians(self, max_gaussians=500):
        """Sample a subset of Gaussians for visualization"""
        total_gaussians = self.gaussians.get_xyz.shape[0]
        
        if total_gaussians <= max_gaussians:
            self.gaussian_indices = torch.arange(total_gaussians)
        else:
            # Random sampling
            self.gaussian_indices = torch.randperm(total_gaussians)[:max_gaussians]
            
        self.gaussian_positions = self.gaussians.get_xyz[self.gaussian_indices]
        print(f"Selected {len(self.gaussian_indices)} Gaussians for visualization")
        
    def visualize_camera_setup(self):
        """Create 3D visualization of camera poses, FOVs, and Gaussians"""
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Gaussians as points
        gaussians_np = self.gaussian_positions.detach().cpu().numpy()
        ax.scatter(gaussians_np[:, 0], gaussians_np[:, 1], gaussians_np[:, 2], 
                  c='gray', s=1, alpha=0.6, label=f'{len(gaussians_np)} Gaussians')
        
        # Plot cameras
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.cameras)))
        
        for i, (camera, color) in enumerate(zip(self.cameras, colors)):
            # Camera position
            pos = camera.camera_center.detach().cpu().numpy()
            ax.scatter(pos[0], pos[1], pos[2], c=[color], s=100, marker='^', 
                      label=f'Camera {i}')
            
            # Camera orientation and FOV visualization
            self._draw_camera_frustum(ax, camera, color, scale=0.5)
            
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('Multi-View Camera Setup with Gaussians')
        
        plt.savefig(os.path.join(self.output_dir, 'camera_setup_3d.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _draw_camera_frustum(self, ax, camera, color, scale=0.5):
        """Draw camera frustum to visualize FOV"""
        # Get camera parameters
        pos = camera.camera_center.detach().cpu().numpy()
        
        # Extract viewing direction from transformation matrix
        world_view_inv = torch.inverse(camera.world_view_transform)
        forward_cam = torch.tensor([0., 0., -1., 0.], device=camera.camera_center.device)
        forward_world = (world_view_inv @ forward_cam)[:3].cpu().numpy()
        forward_world = forward_world / np.linalg.norm(forward_world)
        
        # Create frustum corners
        fov_x, fov_y = camera.FoVx, camera.FoVy
        
        # Simple frustum representation
        frustum_length = scale
        half_width = frustum_length * np.tan(fov_x / 2)
        half_height = frustum_length * np.tan(fov_y / 2)
        
        # Right vector (approximate)
        up = np.array([0, 0, 1])
        right = np.cross(forward_world, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward_world)
        
        # Frustum corners
        center = pos + forward_world * frustum_length
        corners = [
            center + right * half_width + up * half_height,
            center - right * half_width + up * half_height, 
            center - right * half_width - up * half_height,
            center + right * half_width - up * half_height
        ]
        
        # Draw frustum lines
        for corner in corners:
            ax.plot([pos[0], corner[0]], [pos[1], corner[1]], [pos[2], corner[2]], 
                   color=color, alpha=0.6, linewidth=1)
            
        # Draw frustum rectangle
        corners.append(corners[0])  # Close the loop
        corners = np.array(corners)
        ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], 
               color=color, alpha=0.8, linewidth=2)
               
    def visualize_camera_overlaps(self):
        """Visualize which cameras have overlapping views"""
        refiner = MultiViewSAMMaskRefiner(overlap_threshold=0.3)
        overlapping_pairs = refiner.find_overlapping_cameras(self.cameras)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot cameras
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.cameras)))
        positions = []
        
        for i, (camera, color) in enumerate(zip(self.cameras, colors)):
            pos = camera.camera_center.detach().cpu().numpy()
            positions.append(pos)
            ax.scatter(pos[0], pos[1], pos[2], c=[color], s=100, marker='^', 
                      label=f'Camera {i}')
                      
        # Draw overlap connections
        positions = np.array(positions)
        for cam1_idx, cam2_idx in overlapping_pairs:
            pos1, pos2 = positions[cam1_idx], positions[cam2_idx]
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 
                   'r-', alpha=0.7, linewidth=2)
                   
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title(f'Camera Overlaps ({len(overlapping_pairs)} pairs)')
        
        plt.savefig(os.path.join(self.output_dir, 'camera_overlaps.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return overlapping_pairs
        
    def visualize_gaussian_projections(self, max_gaussians_to_show=50):
        """Show how Gaussians project to different camera views"""
        # Sample even fewer Gaussians for projection visualization
        sample_indices = torch.randperm(len(self.gaussian_indices))[:max_gaussians_to_show]
        sample_gaussians = self.gaussian_positions[sample_indices].detach()
        
        refiner = MultiViewSAMMaskRefiner()
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for cam_idx, camera in enumerate(self.cameras[:8]):  # Max 8 cameras
            ax = axes[cam_idx]
            
            # Create white background image
            H, W = camera.image_height, camera.image_width
            img = np.ones((H, W, 3)) * 255  # White background
            
            # Project Gaussians to this camera
            projected_points = []
            visible_points = []
            
            for gauss_idx, gaussian_3d in enumerate(sample_gaussians):
                x, y, visible = refiner.project_3d_to_2d(gaussian_3d, camera)
                
                if visible and x is not None and y is not None:
                    projected_points.append((x, y))
                    visible_points.append(gauss_idx)
                    
                    # Draw point on image with different colors
                    color = (int(255 * ((gauss_idx % 7) / 7)), 
                            int(255 * ((gauss_idx % 5) / 5)), 
                            int(255 * ((gauss_idx % 3) / 3)))
                    cv2.circle(img, (x, y), 4, color, -1)  # Colored dots
                    
            # Convert to proper format for imshow
            img = img.astype(np.uint8)
            ax.imshow(img)
            ax.set_title(f'Camera {cam_idx}\n{len(visible_points)}/{len(sample_gaussians)} visible')
            ax.axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'gaussian_projections.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_sam_masks(self):
        """Visualize original SAM masks for selected cameras"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for cam_idx, camera in enumerate(self.cameras[:8]):
            ax = axes[cam_idx]
            
            if camera.original_sam_mask is not None:
                # Use SAM level 0 for visualization
                sam_mask = camera.original_sam_mask[0].cpu().numpy()
                
                # Create colored mask visualization
                colored_mask = self._colorize_mask(sam_mask)
                    
                ax.imshow(colored_mask)
                ax.set_title(f'Camera {cam_idx} SAM Mask\n{len(np.unique(sam_mask))} segments')
            else:
                ax.text(0.5, 0.5, 'No SAM mask', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Camera {cam_idx} - No mask')
                
            ax.axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sam_masks_original.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def demonstrate_refinement_process(self):
        """Show before/after refinement comparison"""
        # Apply refinement
        refiner = MultiViewSAMMaskRefiner(overlap_threshold=0.3, consensus_strategy="majority_vote")
        
        # Collect original masks
        original_sam_masks = []
        for camera in self.cameras:
            if camera.original_sam_mask is not None:
                original_sam_masks.append(camera.original_sam_mask.cuda())
            else:
                original_sam_masks.append(None)
                
        # Apply refinement
        refined_sam_masks = refiner.refine_sam_masks(
            self.cameras, 
            original_sam_masks, 
            self.gaussians, 
            sam_level=0
        )
        
        # Visualize comparison
        fig, axes = plt.subplots(3, len(self.cameras), figsize=(4*len(self.cameras), 12))
        
        for cam_idx, camera in enumerate(self.cameras):
            # Original mask
            if original_sam_masks[cam_idx] is not None:
                original_mask = original_sam_masks[cam_idx][0].cpu().numpy()
                colored_original = self._colorize_mask(original_mask)
                axes[0, cam_idx].imshow(colored_original)
                axes[0, cam_idx].set_title(f'Original Cam {cam_idx}')
            else:
                axes[0, cam_idx].text(0.5, 0.5, 'No mask', ha='center', va='center', transform=axes[0, cam_idx].transAxes)
                
            # Refined mask
            if refined_sam_masks[cam_idx] is not None:
                refined_mask = refined_sam_masks[cam_idx][0].cpu().numpy()
                colored_refined = self._colorize_mask(refined_mask)
                axes[1, cam_idx].imshow(colored_refined)
                axes[1, cam_idx].set_title(f'Refined Cam {cam_idx}')
            else:
                axes[1, cam_idx].text(0.5, 0.5, 'No mask', ha='center', va='center', transform=axes[1, cam_idx].transAxes)
                
            # Difference
            if original_sam_masks[cam_idx] is not None and refined_sam_masks[cam_idx] is not None:
                diff = (original_mask != refined_mask).astype(float)
                axes[2, cam_idx].imshow(diff, cmap='Reds')
                changed_pixels = np.sum(diff)
                total_pixels = diff.size
                axes[2, cam_idx].set_title(f'Diff: {changed_pixels}/{total_pixels}\n({100*changed_pixels/total_pixels:.1f}%)')
            else:
                axes[2, cam_idx].text(0.5, 0.5, 'No comparison', ha='center', va='center', transform=axes[2, cam_idx].transAxes)
                
            for row in range(3):
                axes[row, cam_idx].axis('off')
                
        axes[0, 0].set_ylabel('Original Masks', rotation=90, size='large')
        axes[1, 0].set_ylabel('Refined Masks', rotation=90, size='large')
        axes[2, 0].set_ylabel('Differences', rotation=90, size='large')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'refinement_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return refined_sam_masks
        
    def _colorize_mask(self, mask):
        """Convert integer mask to colored visualization"""
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
        unique_ids = np.unique(mask)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ids)))
        
        for i, mask_id in enumerate(unique_ids):
            mask_pixels = (mask == mask_id)
            colored_mask[mask_pixels] = colors[i][:3]
            
        return colored_mask
        
    def generate_statistics_report(self):
        """Generate comprehensive statistics about the scene and refinement"""
        refiner = MultiViewSAMMaskRefiner(overlap_threshold=0.3)
        overlapping_pairs = refiner.find_overlapping_cameras(self.cameras)
        
        stats = {
            'scene_path': self.source_path,
            'dataset_path': self.dataset_path,
            'model_path': self.model_path,
            'total_cameras': len(self.scene.getTrainCameras()),
            'visualized_cameras': len(self.cameras),
            'total_gaussians': self.gaussians.get_xyz.shape[0],
            'visualized_gaussians': len(self.gaussian_indices),
            'overlapping_camera_pairs': len(overlapping_pairs),
            'overlap_percentage': 100 * len(overlapping_pairs) / (len(self.cameras) * (len(self.cameras) - 1) / 2)
        }
        
        # Camera-specific stats
        camera_stats = []
        for i, camera in enumerate(self.cameras):
            cam_stat = {
                'camera_id': i,
                'image_size': (camera.image_width, camera.image_height),
                'fov_x_degrees': np.degrees(camera.FoVx),
                'fov_y_degrees': np.degrees(camera.FoVy),
                'has_sam_mask': camera.original_sam_mask is not None,
            }
            
            if camera.original_sam_mask is not None:
                mask = camera.original_sam_mask[0].cpu().numpy()
                cam_stat['num_segments'] = len(np.unique(mask))
                
            camera_stats.append(cam_stat)
            
        stats['cameras'] = camera_stats
        
        # Save statistics
        import json
        with open(os.path.join(self.output_dir, 'scene_statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
            
        # Print summary
        print("\n" + "="*50)
        print("MULTI-VIEW REFINEMENT VISUALIZATION SUMMARY")
        print("="*50)
        print(f"Source: {self.source_path}")
        print(f"Dataset: {self.dataset_path}")
        print(f"Model: {self.model_path or 'None (using initial Gaussians)'}")
        print(f"Cameras: {stats['visualized_cameras']}/{stats['total_cameras']}")
        print(f"Gaussians: {stats['visualized_gaussians']}/{stats['total_gaussians']}")
        print(f"Overlapping pairs: {stats['overlapping_camera_pairs']} ({stats['overlap_percentage']:.1f}%)")
        print(f"Output directory: {self.output_dir}")
        print("="*50)
        
        return stats
        
    def run_full_visualization(self):
        """Run all visualization steps"""
        print("🎥 Visualizing camera setup...")
        self.visualize_camera_setup()
        
        print("🔗 Computing camera overlaps...")
        overlaps = self.visualize_camera_overlaps()
        
        print("📍 Projecting Gaussians...")
        self.visualize_gaussian_projections()
        
        print("🎭 Visualizing SAM masks...")
        self.visualize_sam_masks()
        
        print("🔧 Demonstrating refinement...")
        refined_masks = self.demonstrate_refinement_process()
        
        print("📊 Generating statistics...")
        stats = self.generate_statistics_report()
        
        print("✅ Visualization complete!")
        return stats

def main():
    parser = ArgumentParser(description="Visualize multi-view SAM mask refinement")
    parser.add_argument('-s', '--source_path', type=str, required=True, 
                       help="Path to scene directory (dataset or training output)")
    parser.add_argument('-o', '--output_dir', type=str, default="./multiview_visualization",
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # Create visualizer and run
    visualizer = MultiViewRefinementVisualizer(args.source_path, args.output_dir)
    stats = visualizer.run_full_visualization()
    
    print(f"\n📁 All visualizations saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
