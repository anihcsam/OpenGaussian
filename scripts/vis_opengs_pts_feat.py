import argparse
import numpy as np
from plyfile import PlyData
import open3d as o3d

def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))

def visualize_ply(ply_path):
    # Load the PLY file
    ply_data = PlyData.read(ply_path)
    vertex_data = ply_data['vertex'].data

    # Extract the point cloud attributes
    points = np.array([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    colors = np.array([vertex_data['red'], vertex_data['green'], vertex_data['blue']]).T / 255.0
    opacity = vertex_data['opacity']

    # Apply the opacity filter
    sigmoid_opacity = sigmoid(opacity)
    filtered_indices = sigmoid_opacity >= 0.1
    filtered_points = points[filtered_indices]
    filtered_colors = colors[filtered_indices]

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a PLY point cloud with opacity filtering.")
    parser.add_argument("ply_path", type=str, help="Path to the PLY file")
    args = parser.parse_args()
    visualize_ply(args.ply_path)