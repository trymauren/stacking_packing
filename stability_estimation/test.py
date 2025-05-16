import sys
import git
import time
# import utility
import torch as th
import numpy as np

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
sys.path.append(path_to_root)

# import estimators
import data.feasible_positions_data as fpd
import feasability_estimator.utility as utils


if __name__ == '__main__':
    device = 'mps'
    rng = np.random.default_rng(seed=37910)
    data_size = 100
    num_stacks = data_size
    max_stack_height = 1000
    max_items = 10
    grid_size = (5, 5)
    max_item_size = (5, 5, 5)
    min_item_size = (2, 2, 2)

    stacks = fpd.generate_stacks(
        rng,
        num_stacks=num_stacks,
        grid_size=grid_size,
        max_stack_height=max_stack_height,
        max_items=max_items,
        min_item_size=min_item_size,
        max_item_size=max_item_size,
    )

    items = fpd.random_items((2, 2, 2), (5, 5, 5), rng, num_items=data_size)

    masks = fpd.ob_masks_from_shapes(
        [stack.height_map.shape for stack in stacks],
        [item.get_dimensions() for item in items]
    )

    item1 = items[0]
    mask1 = masks[0]
    stack1 = stacks[0]
    print('Stack:')
    print(stack1.height_map)
    print('Item:')
    print(item1.get_dimensions())
    print('Mask:')
    print(mask1)
    support = utils.check_support(
        np.array([[0, 0]]), item1.get_dimensions(), stack1)
    start = time.time()

    for stack, mask, item in zip(stacks, masks, items):
        ob_feasible_ixs = np.argwhere(mask == 1)
        z_pos_height_map = stack.height_map[ob_feasible_ixs]
        utils.simulate_placements(
            stack, item, ob_feasible_ixs,
            z_coordinates=z_pos_height_map.flatten(),
            render_mode='rgb_array'
        )

    end = time.time()
    print(end-start)
