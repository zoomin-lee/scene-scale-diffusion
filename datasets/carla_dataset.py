import os
import numpy as np
import random
import json
import yaml
import torch
import numba as nb
from torch.utils.data import Dataset

config_file = os.path.join('/home/jumin/multinomial_diffusion/datasets/carla.yaml')
carla_config = yaml.safe_load(open(config_file, 'r'))
LABELS_REMAP = carla_config["learning_map"]
REMAP_FREQUENCIES = carla_config["remap_content"]
FREQUENCIES= carla_config["content"]

LABELS_REMAP = np.asarray(list(LABELS_REMAP.values()))
frequencies_cartesian = np.asarray(list(FREQUENCIES.values()))
remap_frequencies_cartesian = np.asarray(list(REMAP_FREQUENCIES.values()))

class CarlaDataset(Dataset):
    """Carla Simulation Dataset for 3D mapping project
    Access to the processed data, including evaluation labels predictions velodyne poses times
    """
    def __init__(self, directory,
        voxelize_input=True,
        binary_counts=True,
        random_flips=False,
        remap=True,
        num_frames=1,
        transform_pose=True,
        get_gt=True,
        ):
        '''Constructor.
        Parameters:
            directory: directory to the dataset
        '''
        self.get_gt = get_gt
        self.voxelize_input = voxelize_input
        self.binary_counts = binary_counts
        self._directory = directory
        self._num_frames = num_frames
        self.random_flips = random_flips
        self.remap = remap
        self.transform_pose = transform_pose
        self.sparse_output = True
        
        self._scenes = sorted(os.listdir(self._directory))
        self._scenes = [os.path.join(scene, "cartesian") for scene in self._scenes]

        self._num_scenes = len(self._scenes)
        self._num_frames_scene = []

        param_file = os.path.join(self._directory, self._scenes[0], 'evaluation', 'params.json')
        with open(param_file) as f:
            self._eval_param = json.load(f)
        
        self._out_dim = self._eval_param['num_channels']
        self._grid_size = self._eval_param['grid_size']
        self.grid_dims = np.asarray(self._grid_size)
        self._eval_size = list(np.uint32(self._grid_size))
        
        self.coor_ranges = self._eval_param['min_bound'] + self._eval_param['max_bound']
        self.voxel_sizes = [abs(self.coor_ranges[3] - self.coor_ranges[0]) / self._grid_size[0], 
                      abs(self.coor_ranges[4] - self.coor_ranges[1]) / self._grid_size[1],
                      abs(self.coor_ranges[5] - self.coor_ranges[2]) / self._grid_size[2]]
        self.min_bound = np.asarray(self.coor_ranges[:3])
        self.max_bound = np.asarray(self.coor_ranges[3:])
        self.voxel_sizes = np.asarray(self.voxel_sizes)

        self._velodyne_list = []
        self._label_list = []
        self._pred_list = []
        self._eval_labels = []
        self._eval_counts = []
        self._frames_list = []
        self._timestamps = []
        self._poses = [] 

        for scene in self._scenes:
            velodyne_dir = os.path.join(self._directory, scene, 'velodyne')
            label_dir = os.path.join(self._directory, scene, 'labels')
            pred_dir = os.path.join(self._directory, scene, 'predictions')
            eval_dir = os.path.join(self._directory, scene, 'evaluation')
            
            self._num_frames_scene.append(len(os.listdir(velodyne_dir)))

            frames_list = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(velodyne_dir))]
            self._frames_list.extend(frames_list)
            self._velodyne_list.extend([os.path.join(velodyne_dir, str(frame).zfill(6)+'.bin') for frame in frames_list])
            self._label_list.extend([os.path.join(label_dir, str(frame).zfill(6)+'.label') for frame in frames_list])
            self._pred_list.extend([os.path.join(pred_dir, str(frame).zfill(6)+'.bin') for frame in frames_list])
            self._eval_labels.extend([os.path.join(eval_dir, str(frame).zfill(6)+'.label') for frame in frames_list])
            self._eval_counts.extend([os.path.join(eval_dir, str(frame).zfill(6) + '.bin') for frame in frames_list])
            self._timestamps.append(np.loadtxt(os.path.join(self._directory, scene, 'times.txt')))
            self._poses.append(np.loadtxt(os.path.join(self._directory, scene, 'poses.txt')))
            # for poses and timestamps
        self._timestamps = np.array(self._timestamps).reshape(sum(self._num_frames_scene))
        self._poses = np.array(self._poses).reshape(sum(self._num_frames_scene), 12)
        
        self._cum_num_frames = np.cumsum(np.array(self._num_frames_scene) - self._num_frames + 1)

    # Use all frames, if there is no data then zero pad
    def __len__(self):
        return sum(self._num_frames_scene)
    
    def collate_fn(self, data):
        voxel_batch = [bi[0] for bi in data]
        output_batch = [bi[1] for bi in data]
        counts_batch = [bi[2] for bi in data]
        return voxel_batch, output_batch, counts_batch
    
    def points_to_voxels(self, voxel_grid, points, t_i):
        # Valid voxels (make sure to clip)
        voxels = np.floor((points - self.min_bound) / self.voxel_sizes).astype(np.int)
        # Clamp to account for any floating point errors
        maxes = np.reshape(self.grid_dims - 1, (1, 3))
        mins = np.zeros_like(maxes)
        voxels = np.clip(voxels, mins, maxes).astype(np.int)
        # This line is needed to create a mask with number of points, not just binary occupied
        if self.binary_counts:
            voxel_grid[t_i, voxels[:, 0], voxels[:, 1], voxels[:, 2]] += 1
        else:
            unique_voxels, counts = np.unique(voxels, return_counts=True, axis=0)
            unique_voxels = unique_voxels.astype(np.int)
            voxel_grid[t_i, unique_voxels[:, 0], unique_voxels[:, 1], unique_voxels[:, 2]] += counts
        return voxel_grid

    def get_pose(self, idx):
        pose = np.zeros((4, 4))
        pose[3, 3] = 1
        pose[:3, :4] = self._poses[idx].reshape(3, 4)
        return pose

    def __getitem__(self, idx):
        # -1 indicates no data
        # the final index is the output
        idx_range = self.find_horizon(idx)
        if self.transform_pose:
            ego_pose = self.get_pose(idx_range[-1])
            to_ego = np.linalg.inv(ego_pose)
         
        if self.voxelize_input:
            voxel_input = np.zeros((idx_range.shape[0], int(self.grid_dims[0]), int(self.grid_dims[1]), int(self.grid_dims[2])), dtype=np.float32)
        t_i = 0

        for i in idx_range:
            if i == -1: # Zero pad
                points = np.zeros((1, 3), dtype=np.float32)
                
            else:
                points = np.fromfile(self._velodyne_list[i],dtype=np.float32).reshape(-1, 4)[:, :3]

                if self.transform_pose:
                    to_world = self.get_pose(i)
                    relative_pose = np.matmul(to_ego, to_world)
                    points = np.dot(relative_pose[:3, :3], points.T).T + relative_pose[:3, 3]

                valid_point_mask= np.all((points < self.max_bound) & (points >= self.min_bound), axis=1)
                valid_points = points[valid_point_mask, :]

            if self.voxelize_input:
                voxel_input = self.points_to_voxels(voxel_input, valid_points, t_i)

            t_i += 1

        if self.get_gt:
            output = np.fromfile(self._eval_labels[idx_range[-1]],dtype=np.uint32).reshape(self._eval_size).astype(np.uint8)
            counts = np.fromfile(self._eval_counts[idx_range[-1]],dtype=np.float32).reshape(self._eval_size)
        else:
            output = None
            counts = None

        if self.voxelize_input and self.random_flips:
            # X flip
            if np.random.randint(2):
                output = np.flip(output, axis=0)
                counts = np.flip(counts, axis=0)
                voxel_input = np.flip(voxel_input, axis=1) # Because there is a time dimension
            # Y Flip
            if np.random.randint(2):
                output = np.flip(output, axis=1)
                counts = np.flip(counts, axis=1)
                voxel_input = np.flip(voxel_input, axis=2) # Because there is a time dimension
                
        if self.remap:
            output = LABELS_REMAP[output].astype(np.uint8)            

        return voxel_input, output, counts
        
        # no enough frames
    
    def find_horizon(self, idx):
        end_idx = idx
        idx_range = np.arange(idx-self._num_frames, idx)+1
        diffs = np.asarray([int(self._frames_list[end_idx]) - int(self._frames_list[i]) for i in idx_range])
        good_difs = -1 * (np.arange(-self._num_frames, 0) + 1)
        
        idx_range[good_difs != diffs] = -1

        return idx_range
