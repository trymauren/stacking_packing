import sys
import git
import numpy as np

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
sys.path.append(path_to_root + '/code')

from stacking_objects.stack import Stack
from stacking_objects.bullet_env import PyBulletEnv
from stacking_objects.item import Item


def flb_heuristic(observations, flat_action_space=True):
    pallets = observations['pallet']
    upcoming_items_s = observations['upcoming_items']

    def flb_heuristic_helper(observation, flat_action_space=True):
        # THIS WILL FAIL FOR LOOKAHEAD > 1
        pallet = observation['pallet']
        upcoming_items = observation['upcoming_items']

        height_map = pallet
        item = upcoming_items[0]

        current_best_z = np.inf
        current_best_ix = False

        len_x, len_y, len_z, mass = item

        max_y, max_x = height_map.shape
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
            # print('Heuristic couldnt find a spot with full support')
            current_best_ix = (0, 0)

        if flat_action_space:
            ret = np.ravel_multi_index(
                (current_best_ix), dims=(height_map.shape)
            )
        else:
            ret = current_best_ix
        return np.asarray(current_best_ix)

    actions = []
    for pallet, upcoming_items in zip(pallets, upcoming_items_s):
        obs = {'pallet': pallet, 'upcoming_items': upcoming_items}
        actions.append(flb_heuristic_helper(
            obs, flat_action_space=flat_action_space)
        )

    return np.asarray(actions)


def random_heuristic(observations, **kwargs):
    pallets = observations['pallet']
    upcoming_items_s = observations['upcoming_items']

    def random_helper(observation, i=None, rng=None, flat_action_space=True):
        # THIS WILL FAIL FOR LOOKAHEAD > 1
        pallet = observation['pallet']
        upcoming_items = observation['upcoming_items']

        height_map = pallet
        item = upcoming_items[0]

        len_x, len_y, len_z, mass = item

        max_y, max_x = height_map.shape
        max_col_pos = max_x - len_x
        max_row_pos = max_y - len_y

        if rng is None:
            y = np.random.randint(0, max_row_pos + 1)
            x = np.random.randint(0, max_col_pos + 1)

        else:
            # rng = rng[i]
            y = rng.integers(0, max_row_pos + 1)
            x = rng.integers(0, max_col_pos + 1)

        if flat_action_space:
            ret = np.ravel_multi_index(
                (y, x), dims=(height_map.shape)
            )
        else:
            ret = (y, x)
        return np.asarray(ret)

    actions = []

    for i, (pallet, upcoming_items) in enumerate(zip(pallets, upcoming_items_s)):
        obs = {'pallet': pallet, 'upcoming_items': upcoming_items}
        actions.append(random_helper(
            obs, i, **kwargs
        ))

    return np.asarray(actions)


# def or_tools_bin_packing():

#     def create_data_model():
#         """Create the data for the example."""
#         data = {}
#         weights = [48, 30, 19, 36, 36, 27, 42, 42, 36, 24, 30]
#         data["weights"] = weights
#         data["items"] = list(range(len(weights)))
#         data["bins"] = data["items"]
#         data["bin_capacity"] = 10_000  # 100
#         return data

#     data = create_data_model()

#     # Create the mip solver with the SCIP backend.
#     solver = pywraplp.Solver.CreateSolver("SCIP")

#     if not solver:
#         return

#     # Variables
#     # x[i, j] = 1 if item i is packed in bin j.
#     x = {}
#     for i in data["items"]:
#         for j in data["bins"]:
#             x[(i, j)] = solver.IntVar(0, 1, "x_%i_%i" % (i, j))

#     # y[j] = 1 if bin j is used.
#     y = {}
#     for j in data["bins"]:
#         y[j] = solver.IntVar(0, 1, "y[%i]" % j)

#     # The amount packed in each bin cannot exceed its capacity.
#     for j in data["bins"]:
#         solver.Add(
#             sum(x[(i, j)] * data["weights"][i] for i in data["items"])
#             <= y[j] * data["bin_capacity"]
#         )

#     # Objective: minimize the number of bins used.
#     solver.Minimize(solver.Sum([y[j] for j in data["bins"]]))

#     print(f"Solving with {solver.SolverVersion()}")
#     status = solver.Solve()

#     if status == pywraplp.Solver.OPTIMAL:
#         num_bins = 0
#         for j in data["bins"]:
#             if y[j].solution_value() == 1:
#                 bin_items = []
#                 bin_weight = 0
#                 for i in data["items"]:
#                     if x[i, j].solution_value() > 0:
#                         bin_items.append(i)
#                         bin_weight += data["weights"][i]
#                 if bin_items:
#                     num_bins += 1
#                     print("Bin number", j)
#                     print("  Items packed:", bin_items)
#                     print("  Total weight:", bin_weight)
#                     print()
#         print()
#         print("Number of bins used:", num_bins)
#         print("Time = ", solver.WallTime(), " milliseconds")
#     else:
#         print("The problem does not have an optimal solution.")


def main():
    # stack = Stack([5, 5], max_height=10_000, verbose=True)
    # items = [Item([2, 2, 2], 10) for _ in range(8)]
    # for item in items:
    #     find_flb_pos(stack.height_map, item)
    #     stack.place(item)
    #     # print(stack.height_map)

    # print(stack.height_map)
    pass


if __name__ == '__main__':
    main()

