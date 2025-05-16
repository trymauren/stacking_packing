import numpy as np
from stacking_objects.bullet_env import PyBulletEnv

# def flb_to_mid(item):


# def mid_to_flb(item):


def get_pretrained_mask_estimator(path):
    return torch.load('models/mask_estimator.pt')


def get_simulator(render_mode):
    return PyBulletEnv(render_mode=render_mode)


def simulate_placements(
        stack,
        item,
        candidate_coordinates,
        z_coordinates=None,
        render_mode: str = 'human',
        ):

    if z_coordinates is None:
        raise (NotImplementedError)

    stable_arr = np.zeros_like(candidate_coordinates)

    for xy, z in zip(candidate_coordinates, z_coordinates):
        ret = simulate_one_placement(stack, item, xy, z, render_mode)
        stable_arr[xy] = ret
    return stable_arr


def simulate_one_placement(stack, item, xy, z, render_mode):

    simulator = PyBulletEnv(render_mode=render_mode)
    simulator.loadURDF('plane', fileName='plane.urdf')
    simulator.set_lateral_friction('plane', 0, 1)

    half_plane_sizes = np.array([stack.x_lim, stack.y_lim, 0.01])

    simulator.place_visualizer(
        target_position=(stack.x_lim/2, stack.y_lim/2, 0),
        distance=15,
        yaw=0,
        pitch=-10,
            )

    item.x = xy[0]
    item.y = xy[1]
    item.z = z

    mid_pos = item.get_position_mid()

    simulator.create_stack_from_list(stack.items)
    simulator.create_box(
        body_name=999,
        half_extents=np.array(item.get_dimensions())/2,
        mass=item.mass,
        position=np.array(mid_pos),
        rgba_color=np.array([100, 100, 100, 1]),
        lateral_friction=None,
        spinning_friction=None,
        )

    simulator.step()
    stable = simulator.stability(999, theoretic_pos=mid_pos)

    simulator.close()

    return stable


def placement_feasible(
        coordinate,
        item_dimensions,
        stack,
        ):

    x, y = coordinate
    len_x, len_y, len_z = item_dimensions

    if x + len_x > stack.x_lim:
        return 0

    if y + len_y > stack.y_lim:
        return 0

    highest = np.amax(stack.height_map[
        y:(y + len_y),
        x:(x + len_x)]
        )

    z = highest

    if z + len_z >= stack.max_height:
        return 0

    return 1


def placement_supported(
        coordinate,
        item_dimensions,
        stack
        ):

    x, y = coordinate
    len_x, len_y, len_z = item_dimensions

    if x + len_x > stack.x_lim:
        raise IndexError('Attempted to place item outside stack')

    if y + len_y > stack.y_lim:
        raise IndexError('Attempted to place item outside stack')

    def get_z(x, y):
        return stack.height_map[y, x]

    flb = x, y, get_z(x, y)  # front-left-bottom
    rlb = x, y + len_y-1, get_z(x, y + len_y-1)  # rear-left-bottom
    frb = x + len_x-1, y, get_z(x + len_x-1, y)  # front-right-bottom
    rrb = x + len_x-1, y + len_y-1, get_z(x + len_x-1, y + len_y-1)  # rear-right-bottom

    highest = np.amax(stack.height_map[
        y:(y + len_y),
        x:(x + len_x)]
        )

    corners = flb, rlb, frb, rrb

    corners_supported = 0

    for corner in corners:
        corners_supported += int(corner[-1] == highest)

    return corners_supported


def check_feasability(
        ixs_to_check,
        item_dimensions,
        stack
        ):

    feasible = np.zeros_like(ixs_to_check)

    for ix in ixs_to_check:
        feasible[ix] = placement_feasible(ix, item_dimensions, stack)


def get_mask(observation, estimator) -> np.ndarray:
    """

    """
    mask, mask_probs = estimator.forward(observation)
    unconfident_ixs = (mask_probs < 0.9).nonzero()
    feasible_ixs = check_feasability(unconfident_ixs)
    mask[feasible_ixs] = simulate_placements(feasible_ixs)

    return mask


def check_support(
        ixs_to_check,
        item_dimensions,
        stack
        ):

    support = np.zeros(ixs_to_check.shape[:-1])

    for i, ix in enumerate(ixs_to_check):
        support[i] = placement_supported(ix, item_dimensions, stack)
        # placement_supported return the number of corners supported

    return support