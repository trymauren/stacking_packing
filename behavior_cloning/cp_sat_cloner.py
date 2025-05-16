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


# class MultiInputBCPolicy(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.items_net = nn.LSTM(input_size=4, hidden_size=4, num_layers=2, batch_first=True)
#         self.pallet_net = nn.Sequential(
#             nn.Conv2d(1, 4, stride=1, padding=1, kernel_size=3),
#             nn.ReLU(),
#             nn.Conv2d(4, 4, stride=1, padding=1, kernel_size=3),
#             nn.ReLU(),
#             nn.Conv2d(4, 2, stride=1, padding=1, kernel_size=3),
#             nn.ReLU(),
#         )

#         self.policy_net = nn.Sequential(
#             nn.Linear(256, 96),
#             nn.ReLU(),
#             nn.Linear(96, 96),
#             nn.ReLU(),
#             nn.Linear(96, 96),
#             nn.ReLU(),
#         )

#         self.dropout = nn.Dropout(0.1)
#         # 3) Final action layer: 64 -> 96
#         self.action_net = nn.Linear(96, 96)

#     def forward(self, pallet_obs, upcoming_items_obs):
#         # Flatten & concatenate
#         # e.g. pallet_obs.shape = (batch_size, dim1...)
#         # For safety, do .reshape(batch_size, -1).

#         # pallet_flat = pallet_obs.view(pallet_obs.size(0), -1)
#         # print(upcoming_items_obs.shape)  # B,num_items,4

#         pallet_obs = pallet_obs.unsqueeze(1)
#         # upcoming_flat = upcoming_items_obs.view(upcoming_items_obs.size(0), -1)
#         # print(upcoming_flat.shape)  # B,num_items*4

#         # combined = th.cat([pallet_flat, upcoming_flat], dim=1)  # => shape [batch_size, 192]
#         upcoming_output, (hn, cn) = self.items_net(upcoming_items_obs)
#         # print(upcoming_output.shape)  # B,16,4
#         # print(upcoming_output)
#         # print(hn.shape)  # num_lstm_layers,B,4
#         # print(hn)
#         # print(cn.shape)  # num_lstm_layers,B,4
#         # print(cn)
#         # exit()
#         flat_upcoming = upcoming_output.reshape(upcoming_output.size(0), -1)
#         # print(flat_upcoming.shape)  # B,16*4
#         # print(flat_upcoming)

#         pallet_output = self.pallet_net(pallet_obs)
#         # print(pallet_output.shape)  # B, 2, 12, 8
#         # print(pallet_output)

#         flat_pallet = pallet_output.view(pallet_obs.size(0), -1)
#         # print(flat_pallet.shape)  # B, 2*12*8
#         # print(flat_pallet)

#         combined = th.cat([flat_pallet, flat_upcoming], dim=1)
#         hidden = self.policy_net(combined)  # => [batch_size, 64]

#         # final logits (for discrete action space of size 96)
#         actions = self.action_net(hidden)  # => [batch_size, 96]
#         # actions = self.dropout(actions)
#         return actions


class MultiInputBCPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        # 1) Feature extractor(s) for "pallet" and "upcoming_items"
        #    Each is simply Flatten for you, so we replicate that here.

        # Suppose we know the shapes:
        #   pallet.shape = (some_dim1, ...)
        #   upcoming_items.shape = (some_dim2, ...)
        # Flatten + concat => 192 dimension total
        # For BC, we just do the flatten + cat ourselves in forward().

        # 2) The policy MLP (matching MlpExtractor.policy_net).
        #    Input: 192, hidden layers: [64, 64, 64], ReLU, Output: 64.

        # - % val accuracy:
        # self.policy_net = nn.Sequential(
        #     nn.Linear(160, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 96),
        #     nn.ReLU(),
        # )

        # - % val accuracy:
        # self.policy_net = nn.Sequential(
        #     nn.Linear(160, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 96),
        #     nn.ReLU(),
        # )

        # - % val accuracy:
        # self.policy_net = nn.Sequential(
        #     nn.Linear(160, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 96),
        #     nn.ReLU(),
        # )

        # - % val accuracy:
        self.policy_net = nn.Sequential(
            nn.Linear(160, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 96),
            nn.ReLU(),
        )

        # 3) Final action layer: 64 -> 96
        self.action_net = nn.Linear(96, 96)

    def forward(self, pallet_obs, upcoming_items_obs):
        # Flatten & concatenate
        # e.g. pallet_obs.shape = (batch_size, dim1...)
        # For safety, do .reshape(batch_size, -1).

        pallet_flat = pallet_obs.view(pallet_obs.size(0), -1)
        upcoming_flat = upcoming_items_obs.view(upcoming_items_obs.size(0), -1)
        combined = th.cat([pallet_flat, upcoming_flat], dim=1)  # => shape [batch_size, 192]

        # pass through policy MLP
        hidden = self.policy_net(combined)  # => [batch_size, 64]
        # final logits (for discrete action space of size 96)
        action_logits = self.action_net(hidden)  # => [batch_size, 96]

        return action_logits


def train(obs, act, num_epochs=1000, batch_size=128, lr=1e-4, device='cpu'):

    pallet_data = obs['pallet']
    upcoming_data = obs['upcoming_items']

    pallet_tensor = th.from_numpy(pallet_data).float().to(device)
    upcoming_tensor = th.from_numpy(upcoming_data).float().to(device)
    action_tensor = th.from_numpy(act).long().to(device)

    dataset = TensorDataset(pallet_tensor, upcoming_tensor, action_tensor)
    total_size = len(action_tensor)
    validation_ratio = 0.25
    validation_size = int(total_size * validation_ratio)
    training_size = total_size - validation_size
    training_set, validation_set = random_split(dataset, [training_size, validation_size])
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    bc_model = MultiInputBCPolicy().to(device)
    optimizer = optim.Adam(bc_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # loss_fn = th.nn.BCEWithLogitsLoss()
    loss_fn = th.nn.CrossEntropyLoss()
    # loss_fn = th.nn.MSELoss()

    for epoch in tqdm(range(num_epochs)):
        bc_model.train()  # training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for pallet_batch, upcoming_batch, action_batch in train_loader:
            optimizer.zero_grad()
            outputs = bc_model(pallet_batch, upcoming_batch)
            # loss = loss_fn(outputs, F.one_hot(action_batch, num_classes=96).float())
            loss = loss_fn(outputs, action_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate training accuracy
            predicted = outputs.argmax(dim=1)
            correct_train += (predicted == action_batch).sum().item()
            total_train += action_batch.size(0)

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * correct_train / total_train
        print(f"Epoch [{epoch+1}/{num_epochs}], Train loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        # Validation
        bc_model.eval()  # evaluation mode
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with th.no_grad():
            for val_pallet_batch, val_upcoming_batch, val_action_batch in val_loader:
                outputs = bc_model(val_pallet_batch, val_upcoming_batch)
                # loss = loss_fn(outputs, F.one_hot(val_action_batch, num_classes=96).float())
                loss = loss_fn(outputs, val_action_batch)
                running_val_loss += loss.item()

                # Calculate validation accuracy
                predicted_val = outputs.argmax(dim=1)
                correct_val += (predicted_val == val_action_batch).sum().item()
                total_val += val_action_batch.size(0)

        avg_val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100.0 * correct_val / total_val
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

        scheduler.step()

    return bc_model, training_set, validation_set


def test(model, obs, act, batch_size=128, device='cpu'):
    pallet_data = obs['pallet']
    upcoming_data = obs['upcoming_items']

    pallet_tensor = th.from_numpy(pallet_data).float().to(device)
    upcoming_tensor = th.from_numpy(upcoming_data).float().to(device)
    action_tensor = th.from_numpy(act).long().to(device)

    test_set = TensorDataset(pallet_tensor, upcoming_tensor, action_tensor)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    loss_fn = th.nn.CrossEntropyLoss()

    model.eval()  # evaluation mode
    running_test_loss = 0.0
    correct_test = 0
    total_test = 0
    with th.no_grad():
        for test_pallet_batch, test_upcoming_batch, test_action_batch in test_loader:
            outputs = model(test_pallet_batch, test_upcoming_batch)
            loss = loss_fn(outputs, test_action_batch)
            running_test_loss += loss.item()

            # Calculate validation accuracy
            predicted = outputs.argmax(dim=1)
            correct_test += (predicted == test_action_batch).sum().item()
            total_test += test_action_batch.size(0)

        avg_val_loss = running_test_loss / len(test_loader)
        test_accuracy = 100.0 * correct_test / total_test
    print(f'Test loss: {avg_val_loss:.4f}, Accuracy: {test_accuracy:.2f}%')


if __name__ == '__main__':
    NUM_DATA_FILES = 18
    TRAIN_VAL_FILES = 15
    TEST_FILES = 3
    th.manual_seed(829)
    act = []
    pallet_obs = []
    upcoming_items_obs = []
    for i in range(0, TRAIN_VAL_FILES):
        data_file = f'3d_spp_random_data_16_items_10000_instances_12_8_grid_{i}'
        data_path = path_to_root + f'/data/cp_sat_data/{data_file}.json'
        with open(data_path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        act.append(data['actions'])
        pallet_obs.append(data['observations']['pallet'])
        upcoming_items_obs.append(data['observations']['upcoming_items'])

    pallet_obs = np.concatenate(np.asarray(pallet_obs))
    upcoming_items_obs = np.concatenate(np.asarray(upcoming_items_obs))
    act = np.hstack(act)
    print('Steps:', len(act))
    print('Unique act', len(np.unique(act, axis=0)))
    print('Unique upcoming_items_obs', len(np.unique(upcoming_items_obs, axis=0)))
    print('Unique pallet_obs', len(np.unique(pallet_obs, axis=0)))

    obs = {
        'pallet': pallet_obs,
        'upcoming_items': upcoming_items_obs,
    }
    trained_model, train_set, validation_set = train(
        obs, act, num_epochs=100, batch_size=512, lr=1e-3, device='cuda'
    )
    name = '3d_spp_random_data_16_items_10000_instances_12_8_grid_wide_net'
    save_path = path_to_root + f'/behavior_cloning/saved_models/{name}.pt'
    th.save(trained_model.state_dict(), save_path)

    act = []
    pallet_obs = []
    upcoming_items_obs = []
    for k in range(TRAIN_VAL_FILES, NUM_DATA_FILES):
        data_file = f'3d_spp_random_data_16_items_10000_instances_12_8_grid_{k}'
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
    test(trained_model, obs, act, batch_size=128, device='cuda')

    TRAIN_ENV_KWARGS = {
        'sim': True,
        'gravity': 0,
        'render_mode': 'human',
        'renderer': 'Tiny',
        'size': np.asarray([1.2, 0.8]),  # Y, X
        'lookahead': 16,
        'discretisation': 10,
        'max_height': 10_000,
        # 'item_getter_function': random_sampling.random_sample,
        'item_getter_function': cut.layer_cut_4_pieces,
        'i_g_f_kwargs': {
            'grid_shape': np.asarray([12, 8]),  # Y, X
            'num_items': 16,
            'mass': 10,
            'min_': [2, 2, 2],
            'max_': [6, 4, 4],
        },
    }
    exit()
    env = StackingWorld(**TRAIN_ENV_KWARGS)
    trained_model.eval()

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
        print(env.stack.compactness())
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
