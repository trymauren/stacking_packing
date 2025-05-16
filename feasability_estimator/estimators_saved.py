import os
import sys
import git
import torch as th
import numpy as np
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
sys.path.append(path_to_root + '/code')

import data.feasible_positions_data as fpd


# training is influenced by https://pytorch.org/tutorials/beginner/introyt/trainingyt.html


class MaskEstimator(nn.Module):

    """
    Estimates feasible placements with respect to stability and
    locational feasability
    """

    def __init__(self, grid_size):
        super().__init__()

        self.grid_size = grid_size
        # self.layers_grid = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=4, stride=4, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=2),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )

        # with th.no_grad():
        #     dummy_grid = th.zeros(size=(1, *grid_size))
        #     flattened_output = self.layers_grid(dummy_grid).shape[1]

        # # self.linear = nn.Sequential(nn.Linear(flattened_output, features_dim), nn.ReLU())
        # self.layers_item = nn.Sequential(
        #     nn.Linear(ITEM_SIZE, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, flattened_output)
        # )
        flattened_dim = th.zeros(grid_size).flatten().shape[0]
        half_flattened_dim = int(flattened_dim//2)

        self.layers_grid = nn.Sequential(
            nn.Linear(flattened_dim, half_flattened_dim),
            nn.ReLU(),
            nn.Linear(half_flattened_dim, half_flattened_dim),
            nn.ReLU(),
            nn.Linear(half_flattened_dim, half_flattened_dim),
            nn.ReLU(),
            nn.Linear(half_flattened_dim, half_flattened_dim),
            nn.ReLU(),
            nn.Linear(half_flattened_dim, half_flattened_dim),
            nn.ReLU(),
            nn.Linear(half_flattened_dim, 24)
        )

        self.layers_item = nn.Sequential(
            nn.Linear(3, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 24)
        )

        self.output_layers = nn.Sequential(
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.Linear(48, 96),
            nn.ReLU(),
            nn.Linear(96, flattened_dim),
        )

    def forward(self, grid, item):

        # # interpolation through replication
        # factor1 = max(self.grid_size[0]//grid.shape[-1], 1)
        # factor2 = max(self.grid_size[1]//grid.shape[-2], 1)

        # grid = th.kron(grid, th.ones((factor1, factor2)))
        grid = grid.reshape(-1, 1, grid.shape[-1] * grid.shape[-2])
        output_grid = self.layers_grid(grid)
        output_item = self.layers_item(item)
        cat = th.cat((output_grid, output_item), -1)
        output = self.output_layers(cat)
        output = output.reshape(-1, *self.grid_size)
        return output

    def train_one_epoch(
            self,
            epoch_index,
            tb_writer,
            loss_fn,
            optimiser,
            training_loader
            ):

        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair

            input1, input2, labels = data

            # Zero your gradients for every batch!
            optimiser.zero_grad()

            # Make predictions for this batch
            outputs = self(input1, input2)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimiser.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    def train_epochs(self, epochs, training_loader, validation_loader):
        loss_fn = th.nn.BCEWithLogitsLoss()
        optimiser = th.optim.Adam(self.parameters(), lr=0.001)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        best_vloss = 1_000_000.
        epoch_number = 0

        for epoch in range(epochs):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.train(True)
            avg_loss = self.train_one_epoch(
                epoch_number,
                writer,
                loss_fn,
                optimiser,
                training_loader,
            )

            self.eval()
            running_vloss = 0.0
            correct = 0
            total = 0

            with th.no_grad():
                for i, vdata in enumerate(validation_loader):
                    vinputs1, vinputs2, vlabels = vdata
                    voutputs = self(vinputs1, vinputs2)
                    predictions = (voutputs > 0.5).float()

                    correct_elements = (predictions == vlabels).float()

                    sample_accuracies = correct_elements.mean(dim=(-2, -1))

                    correct += sample_accuracies.sum().item()
                    total += vlabels.size(0)

                    vloss = loss_fn(voutputs, vlabels)
                    running_vloss += vloss.item()

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            print(f'Accuracy: {100 * correct // total} %')
            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                               {
                                'Training': avg_loss,
                                'Validation': avg_vloss
                               }, epoch_number + 1)
            writer.flush()

            # # Track best performance, and save the model's state
            # if avg_vloss < best_vloss:
            #     best_vloss = avg_vloss
            #     model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            #     th.save(est.state_dict(), model_path)

            epoch_number += 1

    def predict(self, validation_loader):

        self.eval()

        with th.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs1, vinputs2, vlabels = vdata
                voutputs = self(vinputs1, vinputs2)

        return voutputs, vinputs1, vinputs2, vlabels


class CustomDataset(Dataset):
    def __init__(self, x1, x2, y, device='cpu'):
        """
        Initializes the dataset with inputs x1, x2 and the true values y.

        Parameters:
            x1 (torch.Tensor): First input tensor.
            x2 (torch.Tensor): Second input tensor.
            y (torch.Tensor): True labels or targets.
        """
        self.x1 = x1.to(device)
        self.x2 = x2.to(device)
        self.y = y.to(device)

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.y)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Parameters:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing (x1[idx], x2[idx], y[idx]).
        """
        return self.x1[idx], self.x2[idx], self.y[idx]


if __name__ == '__main__':
    rng = np.random.default_rng(seed=37910)
    num_maps = 100
    num_items = 10
    grid_size = (20, 20)
    max_item_size = (10, 10, 10)
    min_item_size = (2, 2, 2)

    height_maps = fpd.generate_height_maps(
        rng,
        num_maps=num_maps,
        grid_size=grid_size,
        num_items=num_items,
        min_item_size=min_item_size,
        max_item_size=max_item_size,
    )
    items = fpd.random_items((2, 2, 2), (10, 10, 10), rng, num_items=100)

    masks = fpd.masks_from_shapes(
        [height_map.shape for height_map in height_maps],
        [item for item in items]
    )
    x1 = th.from_numpy(height_maps).float()
    x2 = th.from_numpy(items.reshape(-1, 1, items.shape[-1])).float()
    y = th.from_numpy(masks).float()

    model = MaskEstimator((20, 20))

    # Total dataset size
    total_size = len(y)

    # Define split ratio
    validation_ratio = 0.2
    validation_size = int(total_size * validation_ratio)
    training_size = total_size - validation_size

    # Create datasets
    dataset = CustomDataset(x1, x2, y)
    training_set, validation_set = random_split(dataset, [training_size, validation_size])

    training_set = CustomDataset(x1, x2, y)
    validation_set = CustomDataset(x1, x2, y)

    training_loader = th.utils.data.DataLoader(training_set, batch_size=16, shuffle=False)
    validation_loader = th.utils.data.DataLoader(validation_set, batch_size=16, shuffle=False)

    model.train_epochs(1000, training_loader, validation_loader)
