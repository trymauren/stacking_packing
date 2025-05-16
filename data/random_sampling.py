import sys
import git
import numpy as np
import random
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
sys.path.append(path_to_root)


def random_sample(
    *,
    grid_shape,
    num_items,
    mass,
    min_,
    max_,
    rng,
):
    """
    Produce items with static mass.
    """
    low = np.array(min_, dtype=int)[[1, 0, 2]]
    high = np.array(max_, dtype=int)[[1, 0, 2]] + 1
    coords = rng.integers(low, high, size=(num_items, low.size))
    mass_col = np.full((num_items, 1), mass, dtype=int)
    return np.hstack((coords, mass_col))


def descending_volume(
    *,
    grid_shape,
    num_items,
    mass,
    min_,
    max_,
    rng,
):
    """
    Produce items with static mass, sorted by descending volume.
    """
    items = random_sample(
        grid_shape=grid_shape, num_items=num_items,
        mass=mass, min_=min_, max_=max_, rng=rng
    )
    volumes = items[:, :3].prod(axis=1)
    return items[np.argsort(-volumes, kind='stable')]


def mass_var(
    *,
    grid_shape,
    num_items,
    mass,
    min_,
    max_,
    rng,
):
    """
    Produce items with mass proportional to volume.
    """
    low = np.array(min_, dtype=int)[[1, 0, 2]]
    high = np.array(max_, dtype=int)[[1, 0, 2]] + 1
    coords = rng.integers(low, high, size=(num_items, low.size))
    volumes = np.ceil(coords.prod(axis=1, dtype=float) / 10).astype(int)
    return np.column_stack((coords, volumes))


def descending_mass_var(
    *,
    grid_shape,
    num_items,
    mass,
    min_,
    max_,
    rng,
):
    """
    Produce items with mass prop. to volume, sorted by descending volume.
    """
    items = mass_var(
        grid_shape=grid_shape, num_items=num_items,
        mass=mass, min_=min_, max_=max_, rng=rng
    )
    volumes = items[:, :3].prod(axis=1)
    return items[np.argsort(-volumes, kind='stable')]


if __name__ == '__main__':
    from time import time
    rng = np.random.default_rng()
    start = time()
    for _ in range(10000):
        items = random_sample(
            grid_shape=(12, 8), num_items=16, mass=10, min_=[3, 2, 2],
            max_=[6, 4, 4], rng=rng
        )
    end = time()
    print(end-start)
    print(items)
