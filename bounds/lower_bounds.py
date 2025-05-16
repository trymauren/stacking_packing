import numpy as np
from operator import itemgetter
from typing import Any


def get_height_LB_highest_item(items: np.ndarray[int]):
    """
    The highest item lower bound says that the stack/bin cannot be lower
    than the highest item.
    """
    highest_item_height = np.max(items[:, 2])
    return highest_item_height


def get_height_LB_volume(
    items: np.ndarray[int],
    plane_size: list[int] | tuple[int]
):
    """
    The volume lower bound says that the stack/bin cannot be lower than
    the total volume of the items to stack/pack divided by the bottom
    area.
    """
    item_volume_sum = np.sum(np.prod(items[:, :3], axis=-1))
    plane_area = np.prod(plane_size)
    return item_volume_sum/plane_area


def get_height_lower_bounds(
    items: np.ndarray[int],
    plane_size: list[int] | tuple[int]
):

    LB_highest_item = get_height_LB_highest_item(items)
    LB_volume = get_height_LB_volume(items, plane_size)
    return max(LB_highest_item, LB_volume)


def get_compact_LB_volume(
    items: np.ndarray[int],
    plane_size: list[int] | tuple[int]
):
    pass


def get_compact_lower_bounds(
    items: np.ndarray[int],
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

    result = get_compact_lower_bounds(np.asarray(items), grid_shape)
    print(result)
