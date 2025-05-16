import sys
import git
import utility
import torch as th
import numpy as np

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
sys.path.append(path_to_root + '/code')

import estimators
import data.feasible_positions_data as fpd


if __name__ == '__main__':
    device = 'mps'
    rng = np.random.default_rng(seed=37910)
    data_size = 7680
    num_maps = data_size
    max_items = 10
    grid_size = (100, 100)
    max_item_size = (50, 50, 50)
    min_item_size = (5, 5, 5)

    height_maps = fpd.generate_height_maps(
        rng,
        num_maps=num_maps,
        grid_size=grid_size,
        max_items=max_items,
        min_item_size=min_item_size,
        max_item_size=max_item_size,
    )

    items = fpd.random_items((5, 5, 5), (50, 50, 50), rng, num_items=data_size)

    masks = fpd.ob_masks_from_shapes(
        [height_map.shape for height_map in height_maps],
        [item for item in items]
    )
    x1 = th.from_numpy(height_maps).float()
    x2 = th.from_numpy(items.reshape(-1, 1, items.shape[-1])).float()
    y = th.from_numpy(masks).float()

    model = estimators.MaskEstimator((grid_size))

    model.to(device)

    # Total dataset size
    total_size = len(y)

    # Define split ratio
    validation_ratio = 0.2
    validation_size = int(total_size * validation_ratio)
    training_size = total_size - validation_size

    # Create datasets
    dataset = estimators.CustomDataset(x1, x2, y, device=device)
    training_set, validation_set = th.utils.data.random_split(
        dataset, [training_size, validation_size])

    training_loader = th.utils.data.DataLoader(
        training_set, batch_size=128, shuffle=True)
    validation_loader = th.utils.data.DataLoader(
        validation_set, batch_size=128, shuffle=False)
    model.train_epochs(2000, training_loader, validation_loader)
    voutputs, vinputs1, vinputs2, vlabels = model.predict(validation_loader)
    print(np.where(voutputs[0].cpu() > 0.5, 1, 0))
    print(voutputs[0])
    print(vinputs1[0])
    print(vinputs2[0])
    print(vlabels[0])