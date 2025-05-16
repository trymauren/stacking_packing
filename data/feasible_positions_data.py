import sys
import git
import numpy as np

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
sys.path.append(path_to_root)

from stacking_objects.item import Item
from stacking_objects.stack import Stack


def generate_random_height_map(
        rng: np.random.Generator,
        grid_size=(100, 100),
        max_items=10,
        min_item_size=(2, 2, 2),
        max_item_size=(20, 20, 20),
        ):
    """
    Generate a random height map for a pallet with items of varying shapes.

    Args:
    grid_size: Tuple of (rows, cols) for the grid size
    num_items: Number of items to place on the pallet
    max_item_size: Maximum size of an item (rows, cols)
    min_item_size: Minimum size of an item (rows, cols)

    Returns:
    height_map: A 2d np-array representing the pallet's height map
    """

    if not all(mi <= ma for mi, ma in zip(min_item_size, max_item_size)):
        raise ValueError(('All elements in min_values must be less '
                          'than or equal to their corresponding '
                          'elements in max_values.'))
    elif not all(ma <= gs for ma, gs in zip(max_item_size[:2], grid_size)):
        raise ValueError(('All x, y element in max_item_size must be '
                          'less than or equal to their corresponding '
                          'elements in grid_size.'))

    height_map = np.zeros(grid_size)

    num_items = rng.choice(max_items)

    for _ in range(num_items):

        # Random item size
        len_z = rng.integers(min_item_size[2], max_item_size[2] + 1)
        len_y = rng.integers(min_item_size[0], max_item_size[0] + 1)
        len_x = rng.integers(min_item_size[1], max_item_size[1] + 1)
        item_shape = (len_x, len_y)

        # Random position
        max_row_pos = grid_size[0] - len_y
        max_col_pos = grid_size[1] - len_x
        if max_row_pos < 0 or max_col_pos < 0:
            continue  # Skip if the item is larger than the grid
        y = rng.integers(0, max_row_pos + 1)
        x = rng.integers(0, max_col_pos + 1)

        highest = np.amax(height_map[
                y:(y + len_y),
                x:(x + len_x)]
                )

        z = highest

        height_map[y:(y + len_y),
                   x:(x + len_x)] = z + len_z

    return height_map


def generate_height_maps(
        rng: np.random.Generator,
        num_maps=1000,
        **kwargs
        ):
    """
    Generate (possibly) many height maps

    Args:
    rng: Random number generator for reproducibility
    num_maps: Number of height maps to generate
    **kwargs: passed on to generate_random_height_map()

    Returns:
    list of np.ndarrays created in generate_random_height_map()
    """
    return np.stack([generate_random_height_map(rng, **kwargs) for _ in range(num_maps)])


def ob_mask_from_shape(mask_shape, item_shape):
    """
    Generate an out-of-bounds mask with shape equal to mask_shape, dependent
    on item_shape.

    Args:
    mask_shape: Shape of the mask to create
    item_shape: Shape of the item to place

    Returns:
    mask with shape equal to mask_shape
    """

    if not all(m_s >= i_s for m_s, i_s, in zip(mask_shape, item_shape[:2])):
        raise ValueError(('x, y elements in item_shape must be less '
                          'than or equal to their corresponding '
                          'elements in mask_shape.'))

    len_y, len_x, len_z = item_shape
    max_y, max_x = mask_shape
    max_col_pos = max_x - len_x
    max_row_pos = max_y - len_y
    mask = np.zeros(shape=(mask_shape))
    mask[:max_col_pos + 1, :max_row_pos + 1] = 1
    return mask


def ob_masks_from_shapes(mask_shapes, item_shapes):
    """
    Generate (possibly) many out-of-boundsmasks using ob_mask_from_shape().

    Args:
    mask_shape: Shape of the mask to create
    item_shape: Shape of the item to place

    Returns:
    mask with shape equal to mask_shape
    """

    return np.stack([ob_mask_from_shape(m_s, i_s) for m_s, i_s in zip(mask_shapes, item_shapes)])


def random_items(min_values, max_values, rng, num_items=100):
    """
    Generates a number of random 3D-items by creating a
    random length, width and height, dependent on args.

    Args:
    min_values: inclusive
    max_values: inclusive

    Returns:
    2d np-array with items

    Raises:
        ValueError: If any element in min_values is greater than its
        corresponding element in max_values.
    """

    if not all(min_ <= max_ for min_, max_ in zip(min_values, max_values)):
        raise ValueError(('All elements in min_values must be less '
                          'than or equal to their corresponding '
                          'elements in max_values.'))

    return [
        Item((rng.choice(range(min_values[1], max_values[1] + 1)),
              rng.choice(range(min_values[0], max_values[0] + 1)),
              rng.choice(range(min_values[2], max_values[2] + 1))), 1)
        for _ in range(num_items)
        ]


def random_item_dimensions(min_values, max_values, rng, num_items=100):
    """
    Generates a number of random 3D-items by creating a
    random length, width and height, dependent on args.

    Args:
    min_values: inclusive
    max_values: inclusive

    Returns:
    2d np-array with items

    Raises:
        ValueError: If any element in min_values is greater than its
        corresponding element in max_values.
    """

    if not all(min_ <= max_ for min_, max_ in zip(min_values, max_values)):
        raise ValueError(('All elements in min_values must be less '
                          'than or equal to their corresponding '
                          'elements in max_values.'))

    return np.column_stack([
        rng.integers(min_values[1], max_values[1] + 1, num_items),
        rng.integers(min_values[0], max_values[0] + 1, num_items),
        rng.integers(min_values[2], max_values[2] + 1, num_items)
    ])


# def generate_height_maps(
def generate_stacks(
        rng: np.random.Generator,
        num_stacks=1000,
        **kwargs
        ):
    """
    Generate (possibly) many height maps

    Args:
    rng: Random number generator for reproducibility
    num_stacks: Number of stacks to generate
    **kwargs: passed on to generate_stack()

    Returns:
    list of stacks created in generate_stack()
    """

    # Note: np.stack() has nothing to do with other mentions of stack
    return np.stack([generate_stack(rng, **kwargs) for _ in range(num_stacks)])


def generate_stack(
        rng: np.random.Generator,
        grid_size=(100, 100),
        max_stack_height=np.inf,
        max_items=10,
        min_item_size=(2, 2, 2),
        max_item_size=(20, 20, 20),
):
    """
    Generate a random stack with items of varying shapes.

    Args:
    grid_size: Tuple of (rows, cols) for the grid size.
    num_items: Number of items to stack.
    max_item_size: Maximum size of an item.
    min_item_size: Minimum size of an item.

    Returns:
    stack: A Stack() object with the stacked items
    """

    if not all(mi <= ma for mi, ma in zip(min_item_size, max_item_size)):
        raise ValueError(('All elements in min_values must be less '
                          'than or equal to their corresponding '
                          'elements in max_values.'))
    elif not all(ma <= gs for ma, gs in zip(max_item_size[:2], grid_size)):
        raise ValueError(('All x, y element in max_item_size must be '
                          'less than or equal to their corresponding '
                          'elements in grid_size.'))

    stack = Stack(grid_size, max_stack_height)

    num_items = rng.choice(max_items)

    for _ in range(num_items):

        # Random item size
        len_z = rng.integers(min_item_size[2], max_item_size[2] + 1)
        len_y = rng.integers(min_item_size[0], max_item_size[0] + 1)
        len_x = rng.integers(min_item_size[1], max_item_size[1] + 1)
        item_shape = (len_x, len_y)

        # Random position
        max_row_pos = grid_size[0] - len_y
        max_col_pos = grid_size[1] - len_x
        if max_row_pos < 0 or max_col_pos < 0:
            continue  # Skip if the item is larger than the grid
        y = rng.integers(0, max_row_pos + 1)
        x = rng.integers(0, max_col_pos + 1)

        item = Item((len_x, len_y, len_z), 1)
        item.y = y
        item.x = x
        stack.place(item)

    return stack


if __name__ == '__main__':

    import time

    rng = np.random.default_rng(seed=37910)
    num_maps = 100
    num_items = 10
    grid_size = (20, 20)
    max_item_size = (10, 10, 10)
    min_item_size = (2, 2, 2)

    height_maps = generate_height_maps(
        rng,
        num_maps=num_maps,
        grid_size=grid_size,
        num_items=num_items,
        min_item_size=min_item_size,
        max_item_size=max_item_size,
    )
    items = random_items((2, 2, 2), (10, 10, 10), rng, num_items=100)
    mask_shapes = [height_map.shape for height_map in height_maps]
    item_shapes = [item for item in items]
    start = time.time()
    masks = ob_masks_from_shapes(mask_shapes, item_shapes)
    end = time.time()

