import torch
import torch.nn as nn


class DiffusionPolicy(nn.Module):
    """
    Simple conditional diffusion model for action prediction.

    The network learns to predict the noise added to an action given:
    - the current observation
    - a noisy version of the action
    - a diffusion timestep
    """

    def __init__(self, observation_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Keep the original module name so existing checkpoints with `net.*`
        # parameter keys still load without conversion.
        self.net = nn.Sequential(
            nn.Linear(observation_dim + action_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observation, noisy_action, timestep):
        """
        Predicts the noise component added to an action at a given timestep.
        """
        timestep = timestep.unsqueeze(1)
        model_input = torch.cat(
            [observation, noisy_action, timestep], dim=1
        )
        return self.net(model_input)
