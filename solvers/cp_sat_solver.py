import sys
import numpy as np
import matplotlib.pyplot as plt
import git
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ortools.sat.python import cp_model
from random import shuffle
from tqdm import tqdm

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root + '/code')

from stacking_environment.environment import StackingWorld
from stacking_objects.stack import Stack
from stacking_objects.item import Item
from data import cut, random_sampling
from heuristic_algorithms import heuristics
from plotting.plotly_plot_result import plotly_plot_stack


def spp_3d(
    items: list(dict[str:int]),
    container_depth: int,
    container_width: int,
    max_time_to_solve: int = 10_000,
    num_workers: int = 8,
    enable_logging: bool = False,
    random_seed: bool = False,
):

    def print_bool(var, solver):
        return bool(solver.Value(var))

    num_items = len(items)
    items_as_array = [
        [
            item[k] for k, v in item.items()
        ] for item in items
    ]
    # Lower and upper bounds
    max_height = 0
    volume = 0
    for item in items:
        max_height += item['height']
        volume += item['width'] * item['depth'] * item['height']
    min_height = volume / (container_depth*container_width)
    min_height = round(min_height)
    _, h = heuristics.flb_heuristic(
        [container_depth, container_width], items_as_array, return_height=True
    )
    max_height = min(max_height, h)

    model = cp_model.CpModel()

    stack_height = model.new_int_var(min_height, max_height, 'stack_height')

    # Allowed item positions
    positions = []
    for i in range(num_items):
        item = items[i]
        x = model.new_int_var(0, container_width - item['width'], f'x_{i}')
        y = model.new_int_var(0, container_depth - item['depth'], f'y_{i}')
        z = model.new_int_var(0, max_height - item['height'], f'z_{i}')
        positions.append((x, y, z))
        model.add(stack_height >= z + item['height'])

    xy_overlap_container = {}
    x_overlap_container = {}
    y_overlap_container = {}
    bool_z_contacts_container = {}
    has_ground_support_container = {}
    bool_same_z_container = {}

    for i in range(num_items):
        for j in range(i + 1, num_items):
            item1 = items[i]
            item2 = items[j]
            x1, y1, z1 = positions[i]
            x2, y2, z2 = positions[j]

            # No overlap constraints
            no_overlap_left = model.new_bool_var(f'no_overlap_left_{i}_{j}')
            no_overlap_right = model.new_bool_var(f'no_overlap_right_{i}_{j}')
            no_overlap_front = model.new_bool_var(f'no_overlap_front_{i}_{j}')
            no_overlap_back = model.new_bool_var(f'no_overlap_back_{i}_{j}')
            no_overlap_below = model.new_bool_var(f'no_overlap_below_{i}_{j}')
            no_overlap_above = model.new_bool_var(f'no_overlap_above_{i}_{j}')

            model.add(x1 + item1['width'] <= x2).only_enforce_if(no_overlap_left)
            model.add(x1 + item1['width'] > x2).only_enforce_if(no_overlap_left.Not())
            model.add(x2 + item2['width'] <= x1).only_enforce_if(no_overlap_right)
            model.add(x2 + item2['width'] > x1).only_enforce_if(no_overlap_right.Not())
            model.add(y1 + item1['depth'] <= y2).only_enforce_if(no_overlap_front)
            model.add(y1 + item1['depth'] > y2).only_enforce_if(no_overlap_front.Not())
            model.add(y2 + item2['depth'] <= y1).only_enforce_if(no_overlap_back)
            model.add(y2 + item2['depth'] > y1).only_enforce_if(no_overlap_back.Not())
            model.add(z1 + item1['height'] <= z2).only_enforce_if(no_overlap_below)
            model.add(z1 + item1['height'] > z2).only_enforce_if(no_overlap_below.Not())
            model.add(z2 + item2['height'] <= z1).only_enforce_if(no_overlap_above)
            model.add(z2 + item2['height'] > z1).only_enforce_if(no_overlap_above.Not())

            model.add_bool_or([
                no_overlap_left,
                no_overlap_right,
                no_overlap_front,
                no_overlap_back,
                no_overlap_below,
                no_overlap_above
            ])

            # Ordering constraint:
            x_overlap = model.new_bool_var(f'x_overlap_{i}_{j}')
            y_overlap = model.new_bool_var(f'y_overlap_{i}_{j}')
            xy_overlap = model.new_bool_var(f'xy_overlap_{i}_{j}')

            xy_overlap_container[f'xy_overlap_{i}_{j}'] = xy_overlap
            x_overlap_container[f'x_overlap_{i}_{j}'] = x_overlap
            y_overlap_container[f'y_overlap_{i}_{j}'] = y_overlap

            model.add_implication(x_overlap, no_overlap_left.Not())
            model.add_implication(x_overlap, no_overlap_right.Not())
            model.add_bool_or(no_overlap_left, no_overlap_right, x_overlap)

            model.add_implication(y_overlap, no_overlap_front.Not())
            model.add_implication(y_overlap, no_overlap_back.Not())
            model.add_bool_or(no_overlap_front, no_overlap_back, y_overlap)

            model.add_implication(xy_overlap, x_overlap)
            model.add_implication(xy_overlap, y_overlap)
            model.add_bool_or(x_overlap.Not(), y_overlap.Not(), xy_overlap)

            # forall (i, j) in items (I) with i < j: xy_overlap(i, j) -> (z_j >= z_i + h_i)
            model.add(z2 >= z1 + item1['height']).only_enforce_if(xy_overlap)

    for i in reversed(range(num_items)):
        # print(i)
        bool_z_contacts = []
        item2 = items[i]
        x2, y2, z2 = positions[i]
        for k in range(i):
            # print(k)
            if i != k:
                item1 = items[k]
                x1, y1, z1 = positions[k]

                # (z(i)+h_i == z(j) <-> same_z
                same_z = model.new_bool_var(f'z_{i} = z_{k}+h_{k}')
                model.add(z2 == z1 + item1['height']).only_enforce_if(same_z)
                model.add(z2 != z1 + item1['height']).only_enforce_if(same_z.Not())  # reduces solution quality?

                # z-contact constraint:
                # (overlap_xy(i, j) /\ (z(i)+h_i == z(j)) <-> z_contact(i, j)
                # equivalent: overlap_xy /\ same_z <-> z_contact
                xy_overlap = xy_overlap_container[f'xy_overlap_{k}_{i}']
                z_contact = model.new_bool_var(f'z_contact_{i}_{k}')
                model.add_implication(z_contact, same_z)
                model.add_implication(z_contact, xy_overlap)
                model.add_bool_or(same_z.Not(), xy_overlap.Not(), z_contact)

                bool_z_contacts.append(z_contact)
                bool_z_contacts_container[f'z_contact_{i}_{k}'] = z_contact
                bool_same_z_container[f'z_{i} = z_{k}+h_{k}'] = same_z

        # (sum(z_contact(i, j) | j in items) >= 1 \/ has_ground_support(i))
        has_ground_support = model.new_bool_var(f'has_ground_support_{i}')
        model.add(z2 == 0).only_enforce_if(has_ground_support)
        model.add(z2 != 0).only_enforce_if(has_ground_support.Not())

        supported_by_others = model.new_bool_var(f'supported_by_others{i}')
        model.add(sum(bool_z_contacts) >= 1).only_enforce_if(supported_by_others)
        model.add(sum(bool_z_contacts) < 1).only_enforce_if(supported_by_others.Not())

        model.add_exactly_one([supported_by_others, has_ground_support])
        has_ground_support_container[f'has_ground_support_{i}'] = has_ground_support

    # x0, y0, _ = positions[0]
    # model.add(x0 <= int(container_width/2))
    # model.add(y0 <= int(container_depth/2))
    # print(int(container_width/2))
    # print(int(container_depth/2))

    model.minimize(stack_height)
    solver = cp_model.CpSolver()
    if random_seed:
        solver.parameters.random_seed = random_seed
        solver.parameters.interleave_search = True
    solver.parameters.num_search_workers = num_workers

    solver.parameters.max_time_in_seconds = max_time_to_solve
    solver.parameters.log_search_progress = enable_logging  # Enable logging
    status = solver.solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        if status == cp_model.FEASIBLE:
            print('ONLY FEASIBLE SOLUTION FOUND!')
            print(items)
        stack_height_val = solver.value(stack_height)

        pos = []
        for i in range(num_items):
            x_var, y_var, z_var = positions[i]
            pos.append([solver.value(x_var), solver.value(y_var), solver.value(z_var)])

        return stack_height_val, pos

    else:
        print(f"No solution found. Code: {solver.StatusName(status=status)}")
        return 0, []


def test_spp_3d():

    def no_items():
        stack_height, pos = spp_3d([], 1, 1)
        assert pos == []
        assert stack_height == 0

    def one_item_doesnt_fit():
        items = [{'width': 10, 'depth': 10, 'height': 10}]
        stack_height, pos = spp_3d(items, container_depth=5, container_width=5)
        assert stack_height == 0
        assert len(pos) == 0

    def no_items_fit():
        items = [{'width': 10, 'depth': 10, 'height': 10} for _ in range(10)]
        stack_height, pos = spp_3d(items, container_depth=5, container_width=5)
        assert stack_height == 0
        assert len(pos) == 0

    def some_items_doesnt_fit():
        items = [{'width': 10, 'depth': 10, 'height': 10} for _ in range(5)]
        items += [{'width': 2, 'depth': 2, 'height': 2} for _ in range(5)]
        stack_height, pos = spp_3d(items, container_depth=5, container_width=5)
        assert stack_height == 0
        assert len(pos) == 0

    def optimal_solution():
        items = [{'width': 2, 'depth': 2, 'height': 2} for _ in range(8)]
        stack_height, pos = spp_3d(items, container_depth=4, container_width=4)
        assert stack_height == 4
        assert len(pos) == 8

    def within_stack_bounds():
        depth = 10
        width = 10
        max_item_height = 10
        iter_ = (range(1, width + 1), range(1, depth + 1), range(1, max_item_height + 1))
        items = [
            {
                'width': i,
                'depth': j,
                'height': k,
            }
            for i, j, k in zip(*iter_)
        ]
        _, pos = spp_3d(items, depth, width)
        max_height = sum(item['height'] for item in items)
        for p, dim in zip(pos, items):
            assert p[0] >= 0
            assert p[1] >= 0
            assert p[2] >= 0
            assert p[0] + dim['width'] <= width
            assert p[1] + dim['depth'] <= depth
            assert p[2] + dim['height'] <= max_height

    def no_overlap_between_items():
        depth = 10
        width = 10
        max_item_height = 10
        zs = list(range(1, max_item_height + 1))
        shuffle(zs)
        iter_ = (
            range(1, width + 1),
            reversed(range(1, depth + 1)),
            zs
        )
        items = [
            {
                'width': i,
                'depth': j,
                'height': k,
            }
            for i, j, k in zip(*iter_)
        ]
        num_items = len(items)
        _, pos = spp_3d(items, depth, width)
        for i in range(num_items):
            item1 = items[i]
            x1, y1, z1 = pos[i]
            for j in range(i + 1, num_items):
                item2 = items[j]
                x2, y2, z2 = pos[j]
                assert any([
                    x1 + item1['width'] <= x2,
                    x2 + item2['width'] <= x1,
                    y1 + item1['depth'] <= y2,
                    y2 + item2['depth'] <= y1,
                    z1 + item1['height'] <= z2,
                    z2 + item2['height'] <= z1
                ])

    def reordering():
        depth = 10
        width = 10
        max_item_height = 10
        zs = list(range(1, max_item_height + 1))
        shuffle(zs)
        iter_ = (
            range(1, width + 1),
            reversed(range(1, depth + 1)),
            zs
        )
        items = [
            {
                'width': i,
                'depth': j,
                'height': k,
            }
            for i, j, k in zip(*iter_)
        ]
        num_items = len(items)
        _, pos = spp_3d(items, depth, width)
        for i in range(num_items):
            item1 = items[i]
            x1, y1, z1 = pos[i]
            for j in range(i + 1, num_items):
                item2 = items[j]
                x2, y2, z2 = pos[j]
                overlap_left = (x1 + item1['width'] > x2)
                overlap_right = (x2 + item2['width'] > x1)
                overlap_front = (y1 + item1['depth'] > y2)
                overlap_back = (y2 + item2['depth'] > y1)
                x_overlap = overlap_left and overlap_right
                y_overlap = overlap_front and overlap_back
                xy_overlap = x_overlap and y_overlap
                if xy_overlap:
                    assert z2 >= z1 + item1['height']

    def items_supported():
        depth = 10
        width = 10
        max_item_height = 10
        zs = list(range(1, max_item_height + 1))
        shuffle(zs)
        iter_ = (
            range(1, width + 1),
            reversed(range(1, depth + 1)),
            zs
        )
        items = [
            {
                'width': i,
                'depth': j,
                'height': k,
            }
            for i, j, k in zip(*iter_)
        ]
        num_items = len(items)
        _, pos = spp_3d(items, depth, width)
        height_map = np.zeros(shape=(depth, width))
        for i in range(num_items):
            item = items[i]
            x, y, z = pos[i]
            len_x, len_y, len_z = item['width'], item['depth'], item['height']
            highest = np.amax(height_map[y:(y + len_y), x:(x + len_x)])
            assert z == highest
            height_map[y:(y + len_y), x:(x + len_x)] = z + len_z

    no_items()
    one_item_doesnt_fit()
    no_items_fit()
    some_items_doesnt_fit()
    optimal_solution()
    within_stack_bounds()
    no_overlap_between_items()
    reordering()
    items_supported()


def run_spp_3d(
    instances: int = 100,
    grid_shape: tuple[int] = (12, 8),
    num_items: int = 16,
    rng: np.random.Generator | int = 829,
    data_type: 'str' = 'random',
    min_=[2, 2, 2],
    max_=[6, 4, 4],
    mass=10,
    plot=True,
    verbose=False,
    disable_tqdm=True,
    **solver_kwargs,
):

    total_gap = 0
    solutions = []
    solutions_positions = []
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    for i in tqdm(range(instances), disable=disable_tqdm):
        stack = Stack(size=grid_shape, max_height=10_000)

        if data_type == 'random':
            items_as_array = random_sampling.random_sample(
                grid_shape=grid_shape, num_items=num_items, mass=10,
                rng=rng, min_=min_, max_=max_).tolist()
            items_as_dict = items_as_dict_from_array(items_as_array)

        elif data_type == 'layer_cut_4_pieces':
            items_as_array = cut.layer_cut_4_pieces(
                grid_shape=grid_shape, mass=mass, num_items=num_items,
                rng=rng, min_=min_).tolist()
            items_as_dict = items_as_dict_from_array(items_as_array)

        elif data_type == 'layer_cut_multiple_pieces':
            items_as_array = cut.layer_cut_multiple_pieces(
                grid_shape=grid_shape, mass=mass, rng=rng, min_=min_,
                max_=max_, sort=True).tolist()
            items_as_dict = items_as_dict_from_array(items_as_array)

        elif data_type == 'cutting_stock':
            items_as_array = cut.cutting_stock(
                grid_shape=grid_shape, mass=mass, rng=rng, min_=min_,
                max_=max_, sort=True).tolist()

        elif callable(data_type):
            items_as_array = data_type(
                grid_shape=grid_shape, num_items=num_items, rng=rng, min_=min_,
                max_=max_, mass=mass
            )

        items_as_dict = items_as_dict_from_array(items_as_array)

        stack_height, positions = spp_3d(items_as_dict, *grid_shape, **solver_kwargs)

        items_volume = sum(
            item['width'] * item['depth'] * item['height']
            for item in items_as_dict
        )

        # if stack_height is None:
        #     print('oops')
        #     continue

        item_container = []
        item_as_list_container = []
        yx_positions_container = []
        for i in range(len(positions)):
            len_x = items_as_dict[i]['width']
            len_y = items_as_dict[i]['depth']
            len_z = items_as_dict[i]['height']
            item_as_list_container.append([len_x, len_y, len_z, mass])
            x, y, z = positions[i]
            yx_positions_container.append((y, x))
            # print(x, y, z)
            item = Item([len_x, len_y, len_z], mass)
            item.x = x
            item.y = y
            item.z = z
            item_container.append(item)
            # stack.place(item)
            # print(stack.get_height_map())
            # print('item_num:', i)
            # print(f'len_x: {len_x}, len_y: {len_y}, len_z: {len_z}')
            # print(stack.height_map)
            # input()
        if plot:
            plotly_plot_stack(item_container, *grid_shape[::-1], show=True)
        stack_volume = grid_shape[0] * grid_shape[1] * stack_height
        gap = 1 - (items_volume / stack_volume)
        # print(gap)
        total_gap += gap
        solutions.append(item_as_list_container)
        solutions_positions.append(yx_positions_container)

    # Print average gap
    if instances > 0 and verbose:
        print("Average Gap:", total_gap / instances)

    return solutions, solutions_positions


def items_as_array_from_dict(items_as_dict):
    items_as_array = [
        [
            item['width'],
            item['depth'],
            item['height'],
            item['mass']
        ] for item in items_as_dict
    ]
    return items_as_array


def items_as_dict_from_array(items_as_array):
    items_as_dict = [
        {
            'width': item[0],
            'depth': item[1],
            'height': item[2],
            'mass': item[3],
        }
        for item in items_as_array
    ]
    return items_as_dict


# TODO: DOESNT WORK FOR OTHER DATA TYPES THAN RANDOM AND 4CUT
def get_3d_spp_solutions(
    instances: int = 100,
    grid_shape: tuple[int] = (12, 8),
    num_items: int = 16,
    rng: np.random.Generator | int = 829,
    data_type: 'str' = 'random',
    min_=[2, 2, 2],
    max_=[6, 4, 4],
    mass=10,
    disable_tqdm=False,
    **solver_kwargs,
):
    pallet = np.zeros(shape=(instances*num_items, *grid_shape))
    upcoming_items = np.zeros(shape=(instances*num_items, num_items, 4))
    act = np.zeros(shape=(instances*num_items))
    obs = {'pallet': pallet, 'upcoming_items': upcoming_items}

    count = 0
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    for i in tqdm(range(instances), disable=disable_tqdm):
        stack = Stack(size=grid_shape, max_height=10_000)

        if data_type == 'random':
            items_as_array = random_sampling.random_sample(
                grid_shape=grid_shape, num_items=num_items, mass=10,
                rng=rng, min_=min_, max_=max_).tolist()
            items_as_dict = items_as_dict_from_array(items_as_array)

        elif data_type == 'layer_cut_4_pieces':
            items_as_array = cut.layer_cut_4_pieces(
                grid_shape=grid_shape, mass=mass, rng=rng, min_=min_).tolist()
            items_as_dict = items_as_dict_from_array(items_as_array)

        elif data_type == 'layer_cut_multiple_pieces':
            items_as_array = cut.layer_cut_multiple_pieces(
                grid_shape=grid_shape, mass=mass, rng=rng, min_=min_,
                max_=max_, sort=True).tolist()
            items_as_dict = items_as_dict_from_array(items_as_array)

        elif data_type == 'cutting_stock':
            items_as_array = cut.cutting_stock(
                grid_shape=grid_shape, mass=mass, rng=rng, min_=min_,
                max_=max_, sort=True).tolist()
            items_as_dict = items_as_dict_from_array(items_as_array)

        stack_height, positions = spp_3d(
            items_as_dict, *grid_shape, **solver_kwargs
        )
        for k in range(len(positions)):
            obs['pallet'][count] = stack.height_map
            upcoming = items_as_array[k:]
            padding = [[0, 0, 0, 0] for _ in items_as_dict[:k]]
            upcoming += padding
            obs['upcoming_items'][count] = np.asarray(upcoming)
            len_x = items_as_dict[k]['width']
            len_y = items_as_dict[k]['depth']
            len_z = items_as_dict[k]['height']
            x, y, z = positions[k]
            act[count] = np.ravel_multi_index(
                (y, x), dims=(stack.height_map.shape)
            )
            item = Item([len_x, len_y, len_z], 10)
            item.x = x
            item.y = y
            item.z = z
            stack.place(item)
            count += 1
    return obs, act


if __name__ == "__main__":
    # get_3d_spp_solutions(5)
    # test_spp_3d()

    # run_spp_3d(
    #     num_items=28,
    #     data_type='layer_cut_4_pieces',
    #     instances=1000,
    #     grid_shape=(12, 8),
    #     rng=0,
    #     min_=[3, 2, 2],
    #     max_=[6, 4, 4],
    #     mass=10,
    #     plot=False,
    #     max_time_to_solve=10000
    # )

    # run_spp_3d(
    #     num_items=16,
    #     data_type='cutting_stock',
    #     instances=10,
    #     grid_shape=(10, 10),
    #     rng=0,
    #     min_=[2, 2, 2],
    #     max_=[5, 4, 4],
    #     mass=10,
    #     plot=True,
    #     disable_tqdm=False,
    #     max_time_to_solve=10000,
    #     num_workers=32,
    # )

    # items = [{'width': 2, 'depth': 3, 'height': 2, 'mass': 10}, {'width': 2, 'depth': 5, 'height': 3, 'mass': 10}, {'width': 2, 'depth': 3, 'height': 4, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 2, 'mass': 10}, {'width': 2, 'depth': 5, 'height': 3, 'mass': 10}, {'width': 4, 'depth': 3, 'height': 2, 'mass': 10}, {'width': 4, 'depth': 6, 'height': 2, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 4, 'depth': 3, 'height': 2, 'mass': 10}, {'width': 4, 'depth': 5, 'height': 2, 'mass': 10}, {'width': 4, 'depth': 3, 'height': 2, 'mass': 10}, {'width': 4, 'depth': 4, 'height': 2, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 2, 'depth': 3, 'height': 2, 'mass': 10}, {'width': 2, 'depth': 5, 'height': 2, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 2, 'mass': 10}, {'width': 2, 'depth': 5, 'height': 2, 'mass': 10}, {'width': 2, 'depth': 3, 'height': 4, 'mass': 10}, {'width': 2, 'depth': 3, 'height': 4, 'mass': 10}, {'width': 4, 'depth': 4, 'height': 4, 'mass': 10}, {'width': 4, 'depth': 3, 'height': 4, 'mass': 10}, {'width': 4, 'depth': 5, 'height': 4, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 2, 'depth': 5, 'height': 3, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 2, 'depth': 5, 'height': 3, 'mass': 10}]
    # # items = [{'width': 3, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 3, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 3, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 3, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 3, 'depth': 4, 'height': 2, 'mass': 10}, {'width': 3, 'depth': 4, 'height': 2, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 2, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 2, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 2, 'mass': 10}, {'width': 3, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 3, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 3, 'depth': 5, 'height': 2, 'mass': 10}, {'width': 3, 'depth': 3, 'height': 2, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 2, 'mass': 10}, {'width': 4, 'depth': 4, 'height': 2, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 2, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 4, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 3, 'depth': 3, 'height': 3, 'mass': 10}, {'width': 3, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 3, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 4, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 3, 'depth': 5, 'height': 3, 'mass': 10}]
    # # items = [{'width': 3, 'depth': 5, 'height': 2, 'mass': 10}, {'width': 3, 'depth': 4, 'height': 2, 'mass': 10}, {'width': 3, 'depth': 4, 'height': 2, 'mass': 10}, {'width': 4, 'depth': 3, 'height': 2, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 4, 'mass': 10}, {'width': 2, 'depth': 5, 'height': 2, 'mass': 10}, {'width': 3, 'depth': 5, 'height': 2, 'mass': 10}, {'width': 4, 'depth': 3, 'height': 2, 'mass': 10}, {'width': 4, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 4, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 3, 'depth': 4, 'height': 2, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 2, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 2, 'mass': 10}, {'width': 3, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 4, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 4, 'mass': 10}, {'width': 3, 'depth': 4, 'height': 4, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 4, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 3, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 4, 'depth': 5, 'height': 3, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 3, 'mass': 10}, {'width': 4, 'depth': 3, 'height': 3, 'mass': 10}]
    items = [{'width': 2, 'depth': 3, 'height': 2, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 2, 'mass': 10}, {'width': 2, 'depth': 5, 'height': 2, 'mass': 10}, {'width': 2, 'depth': 5, 'height': 2, 'mass': 10}, {'width': 4, 'depth': 5, 'height': 2, 'mass': 10}, {'width': 2, 'depth': 5, 'height': 2, 'mass': 10}, {'width': 3, 'depth': 5, 'height': 2, 'mass': 10}, {'width': 3, 'depth': 5, 'height': 2, 'mass': 10}, {'width': 2, 'depth': 3, 'height': 2, 'mass': 10}, {'width': 2, 'depth': 3, 'height': 4, 'mass': 10}, {'width': 2, 'depth': 3, 'height': 4, 'mass': 10}, {'width': 2, 'depth': 4, 'height': 4, 'mass': 10}, {'width': 4, 'depth': 2, 'height': 4, 'mass': 10}, {'width': 2, 'depth': 2, 'height': 4, 'mass': 10}, {'width': 3, 'depth': 5, 'height': 4, 'mass': 10}, {'width': 3, 'depth': 3, 'height': 4, 'mass': 10}, {'width': 5, 'depth': 5, 'height': 4, 'mass': 10}, {'width': 5, 'depth': 3, 'height': 4, 'mass': 10}, {'width': 2, 'depth': 2, 'height': 4, 'mass': 10}, {'width': 4, 'depth': 3, 'height': 4, 'mass': 10}, {'width': 4, 'depth': 3, 'height': 4, 'mass': 10}, {'width': 4, 'depth': 4, 'height': 4, 'mass': 10}, {'width': 3, 'depth': 5, 'height': 4, 'mass': 10}, {'width': 3, 'depth': 5, 'height': 4, 'mass': 10}, {'width': 4, 'depth': 5, 'height': 4, 'mass': 10}, {'width': 2, 'depth': 5, 'height': 4, 'mass': 10}]
    spp_3d(items, 12, 8, num_workers=32, max_time_to_solve=5, enable_logging=True)