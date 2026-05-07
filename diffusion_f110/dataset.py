import numpy as np
import torch
from torch.utils.data import Dataset


class F110Dataset(Dataset):
    """
    Dataset wrapper for expert driving data used to train the diffusion model.

    Each sample consists of:
    - an observation vector
    - the corresponding expert action
    """

    def __init__(self, dataset_path, augment_mirror=False):
        data = np.load(dataset_path)

        self.observations = torch.from_numpy(data["obs"]).float()
        self.actions = torch.from_numpy(data["act"]).float()
        self.augment_mirror = augment_mirror
        self.lidar_feature_count = self.observations.shape[1] - 4

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, index):
        observation = self.observations[index].clone()
        action = self.actions[index].clone()

        if self.augment_mirror and np.random.rand() < 0.5:
            observation[: self.lidar_feature_count] = torch.flip(
                observation[: self.lidar_feature_count], dims=[0]
            )
            observation[self.lidar_feature_count + 1] *= -1.0
            observation[self.lidar_feature_count + 2] *= -1.0
            action[0] *= -1.0

        return observation, action
