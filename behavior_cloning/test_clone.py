import os
import sys
import git
import json
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import torch.optim as optim
import torch.nn.functional as F

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
sys.path.append(path_to_root)
from stacking_environment.environment import StackingWorld
from data import cut
from data import random_sampling
from behavior_cloning.cp_sat_cloner import MultiInputBCPolicy

if __name__ == '__main__':
    th.manual_seed(829)

    name = '3d_spp_random_data_16_items_10000_instances_12_8_grid_wide_net'
    load_path = path_to_root + f'/behavior_cloning/saved_models/{name}.pt'

    trained_model = MultiInputBCPolicy()
    trained_model.load_state_dict(th.load(load_path, map_location=th.device('cpu')))
    trained_model.eval()

    act = []
    pallet_obs = []
    upcoming_items_obs = []

    data_file = f'3d_spp_random_data_16_items_10000_instances_12_8_grid_29'
    data_path = path_to_root + f'/data/cp_sat_data/{data_file}.json'
    with open(data_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    act.append(data['actions'])
    pallet_obs.append(data['observations']['pallet'])
    upcoming_items_obs.append(data['observations']['upcoming_items'])

    pallet_obs = np.concatenate(np.asarray(pallet_obs))
    upcoming_items_obs = np.concatenate(np.asarray(upcoming_items_obs))
    act = np.hstack(act)
    obs = {
        'pallet': pallet_obs,
        'upcoming_items': upcoming_items_obs,
    }

    TRAIN_ENV_KWARGS = {
        'sim': True,
        'gravity': 0,
        'render_mode': 'human',
        'renderer': 'Tiny',
        'size': np.asarray([1.2, 0.8]),  # Y, X
        'lookahead': 16,
        'discretisation': 10,
        'max_height': 10_000,
        'item_getter_function': random_sampling.random_sample,
        # 'item_getter_function': cut.layer_cut_4_pieces,
        'i_g_f_kwargs': {
            'grid_shape': np.asarray([12, 8]),  # Y, X
            'num_items': 16,
            'mass': 10,
            'min_': [2, 2, 2],
            'max_': [6, 4, 4],
        },
    }

    env = StackingWorld(**TRAIN_ENV_KWARGS)

    # act = trained_model(th.from_numpy(obs['pallet']).float().unsqueeze(0), th.from_numpy(obs['upcoming_items']).float().unsqueeze(0))
    # act = th.argmax(act)

    # term = False
    # for k in range(100):
    #     obs, _ = env.reset()
    #     height_map_, upcoming_items_, action_ = validation_set[k*16]
    #     env.items = upcoming_items_.int().detach().numpy()
    #     obs = {'pallet': height_map_, 'upcoming_items': upcoming_items_}
    #     while not term:
    #         if isinstance(obs['pallet'], th.Tensor):
    #             pallet = obs['pallet'].float().unsqueeze(0)
    #             upci = obs['upcoming_items'].float().unsqueeze(0)
    #         else:
    #             pallet = th.from_numpy(obs['pallet']).float().unsqueeze(0)
    #             upci = th.from_numpy(obs['upcoming_items']).float().unsqueeze(0)
    #         act = trained_model(pallet, upci)
    #         act = th.argmax(act)
    #         obs, r, term, trunc, info = env.step(act.int().detach().numpy())
    #     print(env.stack.compactness())
    #     term = False

    term = False
    for k in range(100):
        obs, _ = env.reset()
        while not term:
            act = trained_model(th.from_numpy(obs['pallet']).float().unsqueeze(0), th.from_numpy(obs['upcoming_items']).float().unsqueeze(0))
            act = th.argmax(act)
            obs, r, term, trunc, info = env.step(act.int().detach().numpy())
        print('Compactness:', env.stack.compactness())
        term = False

# class MultiInputBCPolicy(nn.Module):
#     def __init__(self):
#         super().__init__()

#         # 1) Feature extractor(s) for "pallet" and "upcoming_items"
#         #    Each is simply Flatten for you, so we replicate that here.

#         # Suppose we know the shapes:
#         #   pallet.shape = (some_dim1, ...)
#         #   upcoming_items.shape = (some_dim2, ...)
#         # Flatten + concat => 192 dimension total
#         # For BC, we just do the flatten + cat ourselves in forward().

#         # 2) The policy MLP (matching MlpExtractor.policy_net).
#         #    Input: 192, hidden layers: [64, 64, 64], ReLU, Output: 64.
#         self.policy_net = nn.Sequential(
#             nn.Linear(160, 96),
#             nn.ReLU(),
#             nn.Linear(96, 96),
#             nn.ReLU(),
#             nn.Linear(96, 96),
#             nn.ReLU(),
#             # nn.Linear(96, 96),
#             # nn.ReLU(),
#             # nn.Linear(96, 96),
#             # nn.ReLU(),
#             # nn.Linear(96, 96),
#             # nn.ReLU(),
#         )

#         # 3) Final action layer: 64 -> 96
#         self.action_net = nn.Linear(96, 96)

#     def forward(self, pallet_obs, upcoming_items_obs):
#         # Flatten & concatenate
#         # e.g. pallet_obs.shape = (batch_size, dim1...)
#         # For safety, do .reshape(batch_size, -1).

#         pallet_flat = pallet_obs.view(pallet_obs.size(0), -1)
#         upcoming_flat = upcoming_items_obs.view(upcoming_items_obs.size(0), -1)
#         combined = th.cat([pallet_flat, upcoming_flat], dim=1)  # => shape [batch_size, 192]

#         # pass through policy MLP
#         hidden = self.policy_net(combined)  # => [batch_size, 64]
#         # final logits (for discrete action space of size 96)
#         action_logits = self.action_net(hidden)  # => [batch_size, 96]

#         return action_logits
