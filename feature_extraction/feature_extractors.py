import numpy as np
import torch as th
from torch import nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


class CombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):

        # These features are input to policy/value/q nets.
        stack_obs = observation_space['pallet'].sample()
        flat_stack_shape = stack_obs.squeeze().shape
        features_dim = flat_stack_shape[-1] * flat_stack_shape[-2]

        super().__init__(observation_space, features_dim=features_dim)

        # self.features_dim = features_dim
        # Calculate the number of channels needed to attend to upcoming items
        upcoming_obs = observation_space['upcoming_items'].sample()
        num_upcoming_items, num_item_ft = upcoming_obs.shape

        n_input_channels = num_upcoming_items*(num_item_ft-1) + 1  # TODO: add weight feature??

        self.cnn = nn.Sequential(
            nn.Conv2d(
                n_input_channels, 4, stride=1, padding=1, kernel_size=3
            ),
            nn.ReLU(),
            nn.Conv2d(4, 4, stride=1, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(4, 4, stride=1, padding=1, kernel_size=3),
            nn.ReLU(),
            # nn.Conv2d(4, 4, stride=1, padding=1, kernel_size=3),
            # nn.ReLU(),
            # nn.Conv2d(4, 4, stride=1, padding=1, kernel_size=3),
            # nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            # add batch dims
            pallet = np.expand_dims(stack_obs, axis=0)
            upcoming_items = np.expand_dims(upcoming_obs, axis=0)

            # size=(1, *pallet-shape)
            pallet = th.as_tensor(pallet)
            # size=(1, num_upcoming_items, num_item_features)
            upcoming_items = th.as_tensor(upcoming_items)

            # stretching upcoming item features over the pallet shape
            # size=(1, num_upcoming_items * item_features, *pallet-shape)
            stretched_items = th.cat([
                th.stack(
                    [upcoming_items[:, n, i].expand_as(pallet) for i in range(num_item_ft-1)],  # TODO: add weight feature??
                    dim=1) for n in range(num_upcoming_items)
            ], dim=1)

            # concatenating pallet and stretched upcoming items
            new_observations = th.cat(
                (stretched_items, pallet.unsqueeze(0)), dim=1
            )

            n_flatten = self.cnn(new_observations).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), nn.ReLU()
        )

    def forward(self, observations) -> th.Tensor:
        # See comments in init func for understanding this
        pallet = observations['pallet']
        upcoming_items = observations['upcoming_items']
        upcoming_size = upcoming_items.size()
        num_upcoming_items = upcoming_size[1]
        num_item_ft = upcoming_size[2]  # TODO: add weight feature??
        stretched_items = th.cat([
            th.stack(
                [upcoming_items[:, n, i].view(-1, 1, 1).expand_as(pallet) for i in range(num_item_ft-1)],  # TODO: add weight feature??
                dim=1) for n in range(num_upcoming_items)
        ], dim=1)

        new_observations = th.cat(
            (stretched_items, pallet.unsqueeze(1)), dim=1
        )

        return self.linear(self.cnn(new_observations))


class CustomCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 96):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

# class CustomCNN(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Box):

#         features_dim = observation_space.sample().flatten().shape[0]

#         super().__init__(observation_space, features_dim=features_dim)

#         # self.features_dim = features_dim
#         n_input_channels = 4

#         self.cnn = nn.Sequential(
#             nn.Conv2d(
#                 n_input_channels, 32, kernel_size=3, stride=1, padding=1
#             ),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         with th.no_grad():
#             observation = th.as_tensor(observation_space.sample())
#             n_flatten = self.cnn(observation).shape[1]

#         self.linear = nn.Sequential(
#             nn.Linear(n_flatten, features_dim), nn.ReLU()
#         )

#     def forward(self, observations) -> th.Tensor:
#         observations = th.as_tensor(observations)
#         return self.linear(self.cnn(observations))


class FlattenDictExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        hidden_layers: list[int],
        activation_fn: type[nn.Module] = nn.ReLU,
    ):
        pallet_dim = int(np.prod(observation_space.spaces['pallet'].shape))
        upcoming_dim = int(np.prod(observation_space.spaces['upcoming_items'].shape))
        input_dim = pallet_dim + upcoming_dim

        out_dim = hidden_layers[-1] if len(hidden_layers) > 0 else input_dim
        super().__init__(observation_space, features_dim=out_dim)

        layer_sizes = [input_dim] + hidden_layers
        layers: list[nn.Module] = []
        for in_size, out_size in zip(layer_sizes, layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(activation_fn())

        self.mlp = nn.Sequential(*layers)

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        bsz = observations['pallet'].shape[0]

        x_pallet = observations['pallet'].view(bsz, -1)
        x_up = observations['upcoming_items'].view(bsz, -1)
        x = th.cat([x_pallet, x_up], dim=1)

        return self.mlp(x)
