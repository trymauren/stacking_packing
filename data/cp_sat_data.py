import os
import sys
import git
import json
import numpy

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root + '/code')

from solvers import cp_sat_solver


def generate_data(file_ix, solution_generator_kwargs, solver_kwargs) -> None:
    """
    Generates data using the google cp-sat solver program in
    code/solvers/cp_sat_solver.py, and writes them to file

    Args:
    solution_generator_kwargs: keyword arguments passed to the solution
      generator code/solvers/cp_sat_solver.get_3d_spp_solutions()
    solver_kwargs: keyword arguments passed to the cp-sat solver
    Returns:
    None
    """
    obs, act = cp_sat_solver.get_3d_spp_solutions(
        **solution_generator_kwargs,
        **solver_kwargs
    )

    obs_as_list = {}
    for key, value in obs.items():
        obs_as_list[key] = value.tolist()

    act_as_list = act.tolist()

    data = {'observations': obs_as_list, 'actions': act_as_list}
    path_to_dir = path_to_root + '/code/data/cp_sat_data'
    data_path = f'{path_to_dir}/3d_spp_{data_type}_data_{num_items}_items_'
    data_path += f'{instances}_instances_{grid_shape[0]}_{grid_shape[1]}'
    data_path += f'_grid_{file_ix}.json'

    with open(data_path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, sort_keys=True, indent=4)


if __name__ == '__main__':

    print('REMEMBER TO CHANGE SETTINGS TO AVOID OVERRIDING DATAFILES!')

    num_items = 16
    data_type = 'random'
    instances = 10_000
    grid_shape = (12, 8)
    seed = 829
    min_ = [2, 2, 2]
    max_ = [6, 4, 4]
    mass = 10
    disable_tqdm = False
    solver_logging = False
    max_time_to_solve = 300

    solution_generator_kwargs = {
        'instances': instances,
        'grid_shape': grid_shape,
        'num_items': num_items,
        'rng': numpy.random.default_rng(seed),
        'data_type': data_type,
        'min_': min_,
        'max_': max_,
        'mass': mass,
        'disable_tqdm': disable_tqdm,
    }

    solver_kwargs = {
        'max_time_to_solve': max_time_to_solve,
        'enable_logging': solver_logging,
    }

    for i in range(30):
        generate_data(i, solution_generator_kwargs, solver_kwargs)
