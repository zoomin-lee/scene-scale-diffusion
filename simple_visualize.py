import os
import numpy as np
import open3d as o3d
import argparse
import yaml

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config["learning_map"], config["remap_color_map"]

def load_pointcloud(filepath, learning_map, color_map):
    data = np.loadtxt(filepath, delimiter=' ')
    if data.shape[1] < 4:
        raise ValueError(f"Expected at least 4 columns (label + x y z), got shape {data.shape}")

    raw_labels = data[:, 0].astype(int)
    points = data[:, 1:4]

    # Map raw labels → remapped labels → colors
    remapped_labels = np.array([learning_map.get(int(l), 0) for l in raw_labels])
    colors = np.array([color_map.get(int(l), [255, 255, 255]) for l in remapped_labels]) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='result_for_l_gen/Completion/result_0.txt',
                        help='Path to the point cloud .txt file')
    parser.add_argument('--config', default='datasets/carla.yaml',
                        help='Path to Carla YAML config file')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise FileNotFoundError(f"Point cloud file not found: {args.file}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"YAML config file not found: {args.config}")

    learning_map, color_map = load_config(args.config)
    pcd = load_pointcloud(args.file, learning_map, color_map)

    o3d.visualization.draw([pcd])

if __name__ == "__main__":
    main()
