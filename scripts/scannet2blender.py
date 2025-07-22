import os
import json
import numpy as np
import argparse

def load_transform_matrix(file_path):
    """
    Load the transform matrix from a text file.
    """
    with open(file_path, 'r') as file:
        matrix = [list(map(float, line.strip().split())) for line in file]
    return matrix

def process_directory(directory_path):
    """
    Process each directory and create a JSON file with the transform matrices.
    """
    color_dir = os.path.join(directory_path, "color")           # TODO
    pose_dir = os.path.join(directory_path, "pose")             # TODO
    intrinsic_dir = os.path.join(directory_path, "intrinsic")   # TODO

    # Check if both directories exist
    if not os.path.isdir(color_dir) or not os.path.isdir(pose_dir):
        return

    # ScanNet camera intrinsics (high-resolution: 1296x968)
    # These are the standard ScanNet parameters for the high-res images
    transform_data = {
            'w': 1296,
            'h': 968,
            'fl_x': 1170.187988,
            'fl_y': 1170.187988,
            'cx': 647.75,
            'cy': 483.75,
            # 'aabb_scale': 2,
            'frames': [],
        }
    # Alternative lower-resolution parameters (640x512) - uncomment if needed
    # transform_data = {
    #         'w': 640,
    #         'h': 512,
    #         'fl_x': 534.56,
    #         'fl_y': 534.80,
    #         'cx': 314.27,
    #         'cy': 259.96,
    #         # 'aabb_scale': 2,
    #         'frames': [],
    #     }
    # Collect all image names and sort them
    img_names = [img_name for img_name in os.listdir(color_dir) if img_name.endswith(".jpg")]
    # img_names.sort(key=lambda x: int(os.path.splitext(x)[0]))  # Sort by image number
    img_names.sort(key=lambda x: os.path.splitext(x)[0])  # Sort by image number

    # Iterate over the color images
    for img_name in img_names:
        if img_name.endswith(".jpg"):
            # Construct the corresponding pose file path
            pose_file = os.path.splitext(img_name)[0] + ".txt"
            pose_file_path = os.path.join(pose_dir, pose_file)

            intrinsic_file = os.path.splitext(img_name)[0] + ".txt"
            intrinsic_file_path = os.path.join(intrinsic_dir, intrinsic_file)

            # Check if the pose file exists
            if os.path.isfile(pose_file_path):
                transform_matrix = load_transform_matrix(pose_file_path)
                
                # note: colmap --> blender
                transform_matrix = np.array(transform_matrix)
                transform_matrix[:3, 1:3] *= -1     
                transform_matrix = transform_matrix.tolist()

                frame_data = {
                    "file_path": os.path.join("color", os.path.splitext(img_name)[0]),
                    "transform_matrix": transform_matrix
                }

                if os.path.isfile(intrinsic_file_path):
                    intrinsic_info = load_transform_matrix(intrinsic_file_path)
                    frame_data.update({
                        'fl_x': intrinsic_info[0][0],
                        'fl_y': intrinsic_info[1][1],
                        'cx':  intrinsic_info[0][2],
                        'cy': intrinsic_info[1][2]
                    })

                transform_data["frames"].append(frame_data)

    return transform_data

def main():
    parser = argparse.ArgumentParser(description='Convert ScanNet data to OpenGaussian format')
    parser.add_argument('scannet_path', type=str, 
                       help='Path to the ScanNet data directory (should contain scene directories)')
    parser.add_argument('--scene', type=str, 
                       help='Process only this specific scene (e.g., scene0062_00)')
    
    args = parser.parse_args()
    
    # Use provided scannet_path
    base_directory = args.scannet_path
    
    if not os.path.exists(base_directory):
        print(f"Error: Directory not found at {base_directory}")
        return
    
    print(f"Converting ScanNet data from: {base_directory}")
    
    # Process each scene directory and create JSON files
    for scene_dir in os.listdir(base_directory):
        # If a specific scene is requested, process only that one
        if args.scene and scene_dir != args.scene:
            continue
            
        scene_path = os.path.join(base_directory, scene_dir)
        if os.path.isdir(scene_path):
            print(f"Processing scene: {scene_dir}")
            
            # Process the directory and get the transform data
            transform_data = process_directory(scene_path)

            print(f"  - Processed {scene_path}")
            
            # Create the JSON file
            if transform_data:
                json_file_path = os.path.join(scene_path, "transforms_train.json")
                with open(json_file_path, 'w') as json_file:
                    json.dump(transform_data, json_file, indent=4)
                print(f"  - Created {json_file_path}")
            else:
                print(f"  - No transform data generated for {scene_dir}")

if __name__ == "__main__":
    main()
