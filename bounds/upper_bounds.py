import numpy as np
from operator import itemgetter
from typing import Any


def get_height_UB_tower(items: np.ndarray):
    """
    The stack cannot be higher than all items when both any x AND y
    coordinate of all items overlap.
    """
    tower_height = np.sum(items[:, 2])
    return tower_height


def get_height_upper_bounds(
    items: np.ndarray,
    plane_size: list[int] | tuple[int]
):

    UB_tower = get_height_UB_tower(items)
    return np.min(UB_tower)


def get_compact_upper_bounds(
    items: np.ndarray,
    plane_size: list[int] | tuple[int]
):
    pass


if __name__ == '__main__':
    rng = np.random.default_rng(829)
    grid_shape = (12, 8)
    items = [
                [rng.integers(2, grid_shape[0]//2 + 1),
                 rng.integers(2, grid_shape[1]//2 + 1),
                 rng.integers(2, int(grid_shape[1]//2 + 1))]
                for _ in range(16)
            ]

    result = get_height_upper_bounds(np.asarray(items), grid_shape)
    print(result)
