from typing import Sequence
import numpy as np


def flb_heuristic(
    grid_shape: Sequence[int],
    items: np.ndarray,
    return_height=False,
    print_steps=False,
):

    height_map = np.zeros(shape=grid_shape)
    max_y, max_x = height_map.shape

    for item in items:
        current_best_z = np.inf
        current_best_ix = False
        if len(item) == 4:
            len_x, len_y, len_z, _ = item
        if len(item) == 3:
            len_x, len_y, len_z = item
        max_col_pos = max_x - len_x
        max_row_pos = max_y - len_y

        it = np.nditer(
            height_map[:max_row_pos + 1, :max_col_pos + 1],
            flags=['multi_index']
        )

        for i in it:
            ix = it.multi_index
            start_y, start_x = ix
            end_y, end_x = start_y + len_y, start_x + len_x
            slice_ = height_map[start_y:end_y, start_x:end_x]
            if len(np.unique(slice_)) == 1:
                z = slice_[0, 0]
                if z < current_best_z:
                    current_best_z = z
                    current_best_ix = ix

        if not current_best_ix:
            current_best_ix = (0, 0)
            current_best_z = height_map[0, 0]

        y, x = current_best_ix
        height_map[y:y+len_y, x:x+len_x] = current_best_z + len_z

        if print_steps:
            print(height_map)

    if return_height:
        return height_map, int(np.max(height_map))
    return height_map
