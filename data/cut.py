import sys
import git
import numpy as np
import random
from copy import copy
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
sys.path.append(path_to_root)

from stacking_objects.item import Item
from stacking_objects.stack import Stack


def layer_cut_4_pieces(
    *,
    grid_shape,
    num_items,
    mass,
    rng,
    min_,
    **kwargs
):
    num_layers = int(np.ceil(num_items / 4))

    ret = np.zeros((num_layers * 4, 4), dtype=int)

    for i in range(0, num_layers * 4, 4):

        x_ix = rng.integers(min_[1], grid_shape[1]-min_[1])
        y_ix = rng.integers(min_[0], grid_shape[0]-min_[0])

        item0 = np.asarray([(x_ix, y_ix, 2, mass)])
        item1 = np.asarray([(grid_shape[1] - x_ix, y_ix, 2, mass)])
        item2 = np.asarray([(x_ix, grid_shape[0] - y_ix, 2, mass)])
        item3 = np.asarray(
            [(grid_shape[1] - x_ix, grid_shape[0] - y_ix, 2, mass)]
        )

        ret[i+0] = item0
        ret[i+1] = item1
        ret[i+2] = item2
        ret[i+3] = item3

    return np.asarray(ret, dtype=int)


def layer_cut_multiple_pieces(
    *,
    grid_shape,
    # num_items,
    mass,
    rng,
    min_,
    max_,
    sort=True,
    **kwargs
):
    num_layers = int(min(grid_shape) // min_[2])
    grid_shape_c = copy(grid_shape)
    grid_shape_c = grid_shape_c[::-1]  # function needs [x, y] instead of [y, x]
    max_c = copy(max_)
    min_c = copy(min_)
    max_c[0], max_c[1] = max_c[1], max_c[0]  # function needs [x, y] instead of [y, x]
    min_c[0], min_c[1] = min_c[1], min_c[0]  # function needs [x, y] instead of [y, x]
    stock_to_cut = [*grid_shape_c, min(grid_shape_c)]

    layer_height = int(min(grid_shape_c) // num_layers + 1)

    valid = []
    valid_flbs = []

    invalid = [[*stock_to_cut[:2], layer_height] for _ in range(num_layers)]
    invalid_flbs = [[0, 0, i*layer_height] for i in range(num_layers)]

    while len(invalid):
        item = invalid.pop(-1)  # pop an item to split (the last one)
        flb = invalid_flbs.pop(-1)  # pop the front-left-bottom of the item
        axes = [0, 1, 2]
        # this loop can be made more elegant
        for ix, (dim, max_c_) in enumerate(zip(item, max_c)):
            if dim <= max_c_:
                axes.remove(ix)

        if len(axes) == 0:  # no axes above max_c, the item is valid
            valid.append(item)

        else:  # item is still invalid. e.g., some dimension is too large
            axis = rng.choice(axes)  # sample an axis for cut
            lower = min_c[axis]
            upper = item[axis]
            cut_int = rng.integers(lower, upper - lower)
            item1 = item.copy()
            item2 = item.copy()

            # if-else because want item 1 to have the lowest xyz
            if upper - cut_int > cut_int:
                item1[axis] = cut_int
                item2[axis] = upper - cut_int
            else:
                item1[axis] = upper - cut_int
                item2[axis] = cut_int

            flb_copy = flb.copy()
            if np.greater(item1, max_c).any():
                invalid.append(item1)
                invalid_flbs.append(flb_copy)  # no changes to flb for item1
            else:
                valid.append(item1 + [mass])
                valid_flbs.append(flb_copy)  # no changes to flb for item1

            flb_copy = flb.copy()
            if np.greater(item2, max_c).any():
                invalid.append(item2)
                # must update the flb of item2,
                # as it is offset from original item
                flb_copy[axis] += item1[axis] + 1
                invalid_flbs.append(flb_copy)
            else:
                valid.append(item2 + [mass])
                # must update the flb of item2,
                # as it is offset from original item
                flb_copy[axis] += item1[axis] + 1
                valid_flbs.append(flb_copy)

    items = np.asarray(valid)

    if sort:
        # Sorting on z dim, then y dim, then x dim
        valid_flbs = np.asarray(valid_flbs)
        sorted_indices = np.lexsort(
            (valid_flbs[:, 0], valid_flbs[:, 1], valid_flbs[:, 2])
        )
        items = items[sorted_indices]

    # if num_items is not None:
    #     items = items[:num_items]
    return items


def cutting_stock(
    *,
    grid_shape,
    # num_items,
    mass,
    rng,
    min_,
    max_,
    sort=True,
    **kwargs
):
    grid_shape_c = copy(grid_shape)
    grid_shape_c = grid_shape_c[::-1]  # function needs [x, y] instead of [y, x]
    max_c = copy(max_)
    min_c = copy(min_)
    max_c[0], max_c[1] = max_c[1], max_c[0]  # function needs [x, y] instead of [y, x]
    min_c[0], min_c[1] = min_c[1], min_c[0]  # function needs [x, y] instead of [y, x]

    stock_to_cut = [*grid_shape_c, min(grid_shape_c)]

    valid = []
    valid_flbs = []

    invalid = [stock_to_cut]
    invalid_flbs = [[0, 0, 0]]

    while len(invalid):
        item = invalid.pop(-1)  # pop an item to split (the last one)
        flb = invalid_flbs.pop(-1)  # pop the front-left-bottom of the item
        axes = [0, 1, 2]
        # this loop can be made more elegant
        for ix, (dim, max_c_) in enumerate(zip(item, max_c)):
            if dim <= max_c_:
                axes.remove(ix)

        if len(axes) == 0:  # no axes above max_c, the item is valid
            valid.append(item)

        else:  # item is still invalid. e.g., some dimension is too large
            axis = rng.choice(axes)  # sample an axis for cut
            lower = min_c[axis]
            upper = item[axis]
            cut_int = rng.integers(lower, upper - lower)  # should add +1 here

            item1 = item.copy()
            item2 = item.copy()

            item1[axis] = cut_int
            item2[axis] = upper - cut_int

            flb_copy = flb.copy()
            if np.greater(item1, max_c).any():
                invalid.append(item1)
                invalid_flbs.append(flb_copy)  # no changes to flb for item1
            else:
                valid.append(item1 + [mass])
                valid_flbs.append(flb_copy)  # no changes to flb for item1

            flb_copy = flb.copy()
            if np.greater(item2, max_c).any():
                invalid.append(item2)
                # must update the flb of item2,
                # as it is offset from original item
                flb_copy[axis] += item1[axis] + 1
                invalid_flbs.append(flb_copy)
            else:
                valid.append(item2 + [mass])
                # must update the flb of item2,
                # as it is offset from original item
                flb_copy[axis] += item1[axis] + 1
                valid_flbs.append(flb_copy)

    items = np.asarray(valid)

    # volumes = items[:, 0] * items[:, 1] * items[:, 2]
    # items[:, 3] = volumes / 10.0

    if sort:
        # Sorting on z dim
        valid_flbs = np.asarray(valid_flbs)
        sorted_indices = np.argsort(valid_flbs[:, -1])
        items = items[sorted_indices]

    # if num_items is not None:
    #     items = items[:num_items]
    return items


if __name__ == '__main__':
    from heuristic_algorithms.heuristics import flb_heuristic
    from time import time

    grid_shape = [12, 8]
    num_items = 16
    min_ = [3, 2, 2]
    max_ = [6, 4, 4]
    mass = 10
    rng = np.random.default_rng(829)
    print_steps = True

    start = time()
    for _ in range(10000):
        items = layer_cut_4_pieces(
            grid_shape=grid_shape, num_items=num_items,
            mass=mass, rng=rng, min_=min_
        )
        # print(items)
    end = time()
    print(end-start)
    exit()
    # # hm = flb_heuristic(
    # #     grid_shape, np.asarray(items, dtype=int), print_steps=print_steps
    # # )
    # print(f'layer_cut_4_pieces items generated in {end - start} seconds:')
    # print(items)
    # print(len(items))
    # # print('layer_cut_4_pieces solution according to flb_heuristic:')
    # # print(hm)

    # start = time()
    # items = layer_cut_multiple_pieces(
    #     grid_shape=grid_shape, mass=mass, rng=rng,
    #     min_=min_, max_=max_, sort=True
    # )
    # end = time()
    # # hm = flb_heuristic(
    # #     grid_shape, np.asarray(items, dtype=int), print_steps=print_steps
    # # )
    # print(f'layer_cut_multiple_pieces items generated in {end - start} seconds:')
    # print(items)
    # print(len(items))
    # # print('layer_cut_multiple_pieces solution according to flb_heuristic:')
    # # print(hm)

    start = time()
    unique_items = set()
    from tqdm import tqdm
    for _ in tqdm(range(1_000_000)):
        items = cutting_stock(
            grid_shape=grid_shape, mass=mass, rng=rng,
            min_=min_, max_=max_, num_items=16, sort=True
        )
        for item in items:
            unique_items.add(tuple(item))

    print(f"Total distinct items: {len(unique_items)}")
    end = time()
    # hm = flb_heuristic(
    #     grid_shape, np.asarray(items, dtype=int), print_steps=print_steps
    # )
    print(f'cutting_stock items generated in {end - start} seconds:')
    print(items)
    print(len(items))
    # print('cutting_stock solution according to flb_heuristic:')
    # print(hm)
