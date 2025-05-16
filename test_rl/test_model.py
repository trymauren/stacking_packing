import os
import sys
import git
import pandas as pd
import gymnasium as gym
import torch as th
from torch import nn
import numpy as np
from time import time
from copy import deepcopy
from stable_baselines3.common.vec_env import (
    SubprocVecEnv, DummyVecEnv, VecNormalize
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.ppo_mask import MaskablePPO

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from importlib import import_module

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)

from stacking_environment.environment import StackingWorld
from callback.callbacks import (
    SummaryWriterCallback, EvalSummaryWriterCallback
)
from utils import rng_utils
from plotting.plotly_plot_result import plotly_plot_stack
from plotting.plt_plot_result import plt_plot_stack
from plotting.pyvista_plot_result import pyvista_plot_stack, pyvista_plot_many_stacks
from plotting.plot_test_stats import plot_individual_stats, plot_inference_time, plot_combined_stats
from heuristic_algorithms.obs_act_heuristics import flb_heuristic, random_heuristic
from heuristic_algorithms.evaluate_heuristic import evaluate_heuristic
from solvers.cp_sat_solver import run_spp_3d


OmegaConf.register_new_resolver(
    'scale_array',
    lambda size, factor: [int(x * factor) for x in size]
)
OmegaConf.register_new_resolver(
    'multiply',
    lambda x, y: x * y
)
# OmegaConf.register_new_resolver(
#     'get_item_bound',
#     lambda size, factor: [*[x / factor for x in size], min(size)/factor]
# )
OmegaConf.register_new_resolver(
    'get_item_bound',
    lambda size, factor: [*[x // factor for x in size], min(size) // factor]
)
OmegaConf.register_new_resolver(  # legacy
    'get_lower',
    lambda size, factor: [*[x / factor for x in size], min(size)/factor]
)
OmegaConf.register_new_resolver(  # legacy
    'get_upper',
    lambda size, factor: [*[x / factor for x in size], min(size)/factor]
)


# A helper to convert a dotted path string to a callable.
def load_callable(dotted_path: str):
    module_path, callable_name = dotted_path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, callable_name)


@hydra.main(
    config_path=path_to_root + '/test_rl/test_conf',
    config_name='config',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    LOG_DIR = path_to_root + '/log/' + cfg.log_dir

    train_cfg = OmegaConf.load(LOG_DIR + '/.hydra/config.yaml')

    item_getter_function = load_callable(train_cfg.env.item_getter_function)
    # activation_fn = load_callable(train_cfg.model.policy_kwargs.activation_fn)
    env_id = load_callable(train_cfg.vec_env.env_id)
    vec_env_cls = load_callable(train_cfg.vec_env.vec_env_cls)
    # wrapper_class = load_callable(train_cfg.vec_env.wrapper_class)

    # learn_config = OmegaConf.to_container(train_cfg.learn, resolve=True)
    # eval_config = OmegaConf.to_container(train_cfg.eval, resolve=True)
    # model_config = OmegaConf.to_container(train_cfg.model, resolve=True)
    vec_env_config = OmegaConf.to_container(train_cfg.vec_env, resolve=True)
    i_g_f_kwargs = OmegaConf.to_container(train_cfg.env.i_g_f_kwargs, resolve=True)

    # model_config['policy_kwargs']['activation_fn'] = activation_fn
    vec_env_config['env_id'] = env_id
    vec_env_config['vec_env_cls'] = vec_env_cls
    # vec_env_config['vec_env_cls'] = DummyVecEnv
    # vec_env_config['wrapper_class'] = wrapper_class
    vec_env_config['env_kwargs']['item_getter_function'] = item_getter_function
    vec_env_config['env_kwargs']['i_g_f_kwargs'] = i_g_f_kwargs

    print('i_g_f_kwargs: ')
    print(i_g_f_kwargs)

    evaluate_policy_kwargs = OmegaConf.to_container(cfg.evaluate_policy_kwargs, resolve=True)
    override_train_env = OmegaConf.to_container(cfg.override_train_env, resolve=True)
    for key in override_train_env:
        vec_env_config['env_kwargs'][key] = override_train_env[key]
    vec_env_config['n_envs'] = cfg.override_make_vec_env['n_envs']

    # Setup -----------------------------------------------------------

    BEST_MODEL_PATH = LOG_DIR
    ENV_NORMALIZE_PATH = LOG_DIR
    LAST_MODEL_PATH = LOG_DIR

    if cfg.model_dir:
        BEST_MODEL_PATH += f'/saved_model/{cfg.model_dir}/best_model'
        ENV_NORMALIZE_PATH += f'/saved_model/{cfg.model_dir}/best_vec_normalize.pkl'
        LAST_MODEL_PATH += f'/saved_model/{cfg.model_dir}/final_model'
        STACK_PLOT_PATH = LOG_DIR + f'/stack_plots/{cfg.model_dir}'
        PLOT_PATH = LOG_DIR + f'/plots/{cfg.model_dir}'
        STATS_PATH = LOG_DIR + f'/stats/{cfg.model_dir}'
    else:
        BEST_MODEL_PATH += '/saved_model/best_model'
        ENV_NORMALIZE_PATH += '/saved_model/best_vec_normalize.pkl'
        LAST_MODEL_PATH += '/saved_model/final_model'
        STACK_PLOT_PATH = LOG_DIR + '/stack_plots'
        PLOT_PATH = LOG_DIR + '/plots'
        STATS_PATH = LOG_DIR + '/stats'

    if not os.path.exists(STACK_PLOT_PATH):
        os.makedirs(STACK_PLOT_PATH)
    if not os.path.exists(PLOT_PATH):
        os.makedirs(PLOT_PATH)
    if not os.path.exists(STATS_PATH):
        os.makedirs(STATS_PATH)

    rl_stats = []
    random_stats = []
    flb_stats = []
    cp_sat_stats = []

    rl_time = []
    random_time = []
    flb_time = []
    cp_sat_time = []

    for seed in cfg.seeds:
        if cfg.test_rl:
            # rng_utils.set_global_seed(seed)
            vec_env_config['seed'] = seed
            env = make_vec_env(**vec_env_config)
            if train_cfg.norm:
                env = VecNormalize.load(ENV_NORMALIZE_PATH, env)
                env.training = False
                env.norm_reward = False

            try:
                print('Load best model')
                loaded_model = MaskablePPO.load(BEST_MODEL_PATH)
            except FileNotFoundError:
                print('Didnt find best model, loading last model')
                loaded_model = MaskablePPO.load(LAST_MODEL_PATH)
            finally:
                pass

            # RL ==========================================================
            print(f'Running RL with seed {seed}')
            rl_test_callback = EvalSummaryWriterCallback()
            start_time = time()

            evaluate_policy(
                loaded_model,
                env,
                callback=rl_test_callback,
                **evaluate_policy_kwargs,
            )
            elapsed = time() - start_time
            num_instances = cfg.evaluate_policy_kwargs.n_eval_episodes
            rl_stats.append(rl_test_callback.get_stats())
            rl_time.append(elapsed/num_instances)

        # =============================================================

        flat_action_space = vec_env_config['env_kwargs']['flat_action_space']

        # RANDOM ======================================================
        if cfg.test_random:
            print(f'Running RANDOM with seed {seed}')
            # rng_utils.set_global_seed(seed)
            env = make_vec_env(**vec_env_config)
            random_test_callback = EvalSummaryWriterCallback()
            start_time = time()
            evaluate_heuristic(
                random_heuristic,
                env,
                callback=random_test_callback,
                **evaluate_policy_kwargs,
                heuristic_kwargs={
                    # 'rng': env.env_method('get_np_random'),
                    'rng': np.random.default_rng(seed),
                    'flat_action_space': flat_action_space
                }
            )
            elapsed = time() - start_time
            num_instances = cfg.evaluate_policy_kwargs.n_eval_episodes
            random_stats.append(random_test_callback.get_stats())
            random_time.append(elapsed/num_instances)
        # =============================================================

        # FLB =========================================================
        if cfg.test_flb:
            print(f'Running FLB with seed {seed}')
            # rng_utils.set_global_seed(seed)
            env = make_vec_env(**vec_env_config)
            flb_test_callback = EvalSummaryWriterCallback()
            start_time = time()
            evaluate_heuristic(
                flb_heuristic,
                env,
                callback=flb_test_callback,
                **evaluate_policy_kwargs,
                heuristic_kwargs={
                    'flat_action_space': flat_action_space
                }
            )
            elapsed = time() - start_time
            num_instances = cfg.evaluate_policy_kwargs.n_eval_episodes
            flb_stats.append(flb_test_callback.get_stats())
            flb_time.append(elapsed/num_instances)
        # =============================================================

        # CP-SAT ======================================================
        if cfg.test_cp_sat:
            print(f'Running CP-SAT with seed {seed}')
            # rng_utils.set_global_seed(seed)
            start_time = time()
            solutions, solutions_positions = run_spp_3d(
                instances=cfg.evaluate_policy_kwargs.n_eval_episodes,
                rng=seed,
                data_type=item_getter_function,
                **i_g_f_kwargs,
                plot=False,
                max_time_to_solve=5,
                num_workers=32,
                disable_tqdm=True,
            )
            cp_sat_stats_temp = []
            env = StackingWorld(**vec_env_config['env_kwargs'])
            env.reset(seed=seed)
            for solution, positions in zip(solutions, solutions_positions):
                env.items = np.asarray(solution)
                for pos in positions:
                    grid_shape = i_g_f_kwargs['grid_shape']
                    if flat_action_space:
                        loc = np.ravel_multi_index(
                            pos, dims=grid_shape)
                    else:
                        loc = pos
                    obs, rew, term, trun, infos = env.step(loc)

                    if term or trun:
                        cp_sat_stats_temp.append(infos)
                        env.reset()
                        break

                if not term or trun:
                    print('ERROR!')
                    exit()

            elapsed = time() - start_time
            num_instances = cfg.evaluate_policy_kwargs.n_eval_episodes
            cp_sat_stats.append(pd.DataFrame(cp_sat_stats_temp))
            cp_sat_time.append(elapsed/num_instances)
        # =============================================================

    # Write stats to file =============================================
    seeds = cfg.seeds

    if cfg.test_rl:
        rl_stats_with_seed = [
            df.assign(seed=s, infer_time=t) for df, s, t in zip(
                rl_stats, seeds, rl_time)]
        rl_stats_cat = pd.concat(rl_stats_with_seed, ignore_index=True)

        rl_stats_mean = rl_stats_cat.mean(numeric_only=True)
        rl_stats_mean['time'] = np.mean(rl_time)
        if cfg.write_stats:
            rl_stats_cat.to_csv(STATS_PATH + '/rl_stats.csv')
            rl_stats_mean.to_csv(STATS_PATH + '/mean_rl_stats.csv')

    if cfg.test_flb:
        flb_stats_with_seed = [
            df.assign(seed=s, infer_time=t) for df, s, t in zip(
                flb_stats, seeds, flb_time)]
        flb_stats_cat = pd.concat(flb_stats_with_seed, ignore_index=True)

        flb_stats_mean = flb_stats_cat.mean(numeric_only=True)
        flb_stats_mean['time'] = np.mean(flb_time)
        if cfg.write_stats:
            flb_stats_cat.to_csv(STATS_PATH + '/flb_stats.csv')
            flb_stats_mean.to_csv(STATS_PATH + '/mean_flb_stats.csv')

    if cfg.test_random:
        random_stats_with_seed = [
            df.assign(seed=s, infer_time=t) for df, s, t in zip(
                random_stats, seeds, random_time)]
        random_stats_cat = pd.concat(random_stats_with_seed, ignore_index=True)

        random_stats_mean = random_stats_cat.mean(numeric_only=True)
        random_stats_mean['time'] = np.mean(random_time)
        if cfg.write_stats:
            random_stats_cat.to_csv(STATS_PATH + '/random_stats.csv')
            random_stats_mean.to_csv(STATS_PATH + '/mean_random_stats.csv')

    if cfg.test_cp_sat:
        cp_sat_stats_with_seed = [
            df.assign(seed=s, infer_time=t) for df, s, t in zip(
                cp_sat_stats, seeds, cp_sat_time)]
        cp_sat_stats_cat = pd.concat(cp_sat_stats_with_seed, ignore_index=True)

        cp_sat_stats_mean = cp_sat_stats_cat.mean(numeric_only=True)
        cp_sat_stats_mean['time'] = np.mean(cp_sat_time)
        if cfg.write_stats:
            cp_sat_stats_cat.to_csv(STATS_PATH + '/cp_sat_stats.csv')
            cp_sat_stats_mean.to_csv(STATS_PATH + '/mean_cp_sat_stats.csv')

    # =================================================================

    # Plotting ========================================================
    if cfg.make_plots:
        # stats = [
        #     rl_stats_cat, cp_sat_stats_cat, flb_stats_cat, random_stats_cat
        # ]
        # algorithms = ['RL', 'CP-SAT', 'Close-to-origin', 'Random']
        # for stat, alg in zip(stats, algorithms):
        #     stat['algorithm'] = alg
        # metrics = [
        #     'stack_compactness', 'stack_stability_bool', 'completed_act_ratio'
        # ]
        # names = [
        #     'Compactness', 'Stable stacks ratio', 'Stable actions ratio'
        # ]
        # all_data = pd.concat(
        #     [rl_stats_cat, cp_sat_stats_cat, flb_stats_cat, random_stats_cat],
        #     ignore_index=True
        # )
        # plot_individual_stats(all_data, metrics, names, algorithms, PLOT_PATH)
        # plot_combined_stats(all_data, metrics, names, algorithms, PLOT_PATH, seed)

        # time_df = {
        #     'RL': rl_time, 'CP-SAT': cp_sat_time,
        #     'Close-to-origin': flb_time, 'Random': random_time,
        # }
        # plot_inference_time(time_df, algorithms, PLOT_PATH)

        # # Stack plots =====================================================

        discretisation = vec_env_config['env_kwargs']['discretisation']
        size = vec_env_config['env_kwargs']['size']

        D, W = int(size[0] * discretisation), int(size[1] * discretisation)

        if cfg.plot_ind_stacks:
            IND_PATH = STACK_PLOT_PATH + '/ind_stacks'
            if not os.path.exists(IND_PATH):
                os.makedirs(IND_PATH)

            for i in range(cfg.ind_num_stacks):
                if cfg.test_rl:
                    pyvista_plot_stack(
                        rl_stats_cat['stacked_items'].iloc[i],
                        W,
                        D,
                        save_path=f'{IND_PATH}/rl_stack_{i}.pdf'
                    )
                if cfg.test_random:
                    pyvista_plot_stack(
                        random_stats_cat['stacked_items'].iloc[i],
                        W,
                        D,
                        save_path=f'{IND_PATH}/random_stack_{i}.pdf'
                    )
                if cfg.test_flb:
                    pyvista_plot_stack(
                        flb_stats_cat['stacked_items'].iloc[i],
                        W,
                        D,
                        save_path=f'{IND_PATH}/flb_stack_{i}.pdf'
                    )
                if cfg.test_cp_sat:
                    pyvista_plot_stack(
                        cp_sat_stats_cat['stacked_items'].iloc[i],
                        W,
                        D,
                        save_path=f'{IND_PATH}/cp_sat_stack_{i}.pdf'
                    )

        if cfg.plot_collection:
            COL_PATH = STACK_PLOT_PATH + '/col_stacks'
            if not os.path.exists(COL_PATH):
                os.makedirs(COL_PATH)

            n = cfg.coll_num_stacks

            if cfg.test_rl:
                c = rl_test_callback.get_stats()['stack_compactness'].round(3)[:n]
                pyvista_plot_many_stacks(
                    rl_stats_cat['stacked_items'].iloc[:n].tolist(),
                    W,
                    D,
                    compactness=c,
                    heights=(rl_stats_cat['stack_height'].iloc[:n]*discretisation).tolist(),
                    lbs=(rl_stats_cat['height_lower_bound'].iloc[:n]*discretisation).tolist(),
                    n_to_stack=rl_stats_cat['n_to_stack'].iloc[:n].tolist(),
                    n_stacked=rl_stats_cat['n_stacked'].iloc[:n].tolist(),
                    shape=(1, n),
                    save_path=f'{COL_PATH}/rl_stacks.pdf'
                )

            if cfg.test_random:
                c = random_test_callback.get_stats()['stack_compactness'].round(3)[:n]
                pyvista_plot_many_stacks(
                    random_stats_cat['stacked_items'].iloc[:n].tolist(),
                    W,
                    D,
                    compactness=c,
                    heights=(random_stats_cat['stack_height'].iloc[:n]*discretisation).tolist(),
                    lbs=(random_stats_cat['height_lower_bound'].iloc[:n]*discretisation).tolist(),
                    n_to_stack=random_stats_cat['n_to_stack'].iloc[:n].tolist(),
                    n_stacked=random_stats_cat['n_stacked'].iloc[:n].tolist(),
                    shape=(1, n),
                    save_path=f'{COL_PATH}/random_stacks.pdf'
                )

            if cfg.test_flb:
                c = flb_test_callback.get_stats()['stack_compactness'].round(3)[:n]
                pyvista_plot_many_stacks(
                    flb_stats_cat['stacked_items'].iloc[:n].tolist(),
                    W,
                    D,
                    compactness=c,
                    heights=(flb_stats_cat['stack_height'].iloc[:n]*discretisation).tolist(),
                    lbs=(flb_stats_cat['height_lower_bound'].iloc[:n]*discretisation).tolist(),
                    n_to_stack=flb_stats_cat['n_to_stack'].iloc[:n].tolist(),
                    n_stacked=flb_stats_cat['n_stacked'].iloc[:n].tolist(),
                    shape=(1, n),
                    save_path=f'{COL_PATH}/flb_stacks.pdf'
                )

            if cfg.test_cp_sat:
                c = cp_sat_stats_cat['stack_compactness'].round(3)[:n]
                pyvista_plot_many_stacks(
                    cp_sat_stats_cat['stacked_items'].iloc[:n].tolist(),
                    W,
                    D,
                    compactness=c,
                    heights=(cp_sat_stats_cat['stack_height'].iloc[:n]*discretisation).tolist(),
                    lbs=(cp_sat_stats_cat['height_lower_bound'].iloc[:n]*discretisation).tolist(),
                    n_to_stack=cp_sat_stats_cat['n_to_stack'].iloc[:n].tolist(),
                    n_stacked=cp_sat_stats_cat['n_stacked'].iloc[:n].tolist(),
                    shape=(1, n),
                    save_path=f'{COL_PATH}/cp_sat_stacks.pdf'
                )
        temp = (cfg.test_rl, cfg.test_random, cfg.test_flb, cfg.test_cp_sat)
        if cfg.plot_comparison and all(temp):
            COMP_PATH = STACK_PLOT_PATH + '/comp_stacks'
            if not os.path.exists(COMP_PATH):
                os.makedirs(COMP_PATH)

            rows = 4
            cols = 3

            c1 = rl_test_callback.get_stats()['stack_compactness'].round(3)[:cols]
            c2 = random_test_callback.get_stats()['stack_compactness'].round(3)[:cols]
            c3 = flb_test_callback.get_stats()['stack_compactness'].round(3)[:cols]
            c4 = cp_sat_stats_cat['stack_compactness'].round(3)[:cols]

            rl_instances = rl_stats_cat['stacked_items'].iloc[:cols].tolist()
            random_instances = random_stats_cat['stacked_items'].iloc[:cols].tolist()
            flb_instances = flb_stats_cat['stacked_items'].iloc[:cols].tolist()
            cp_sat_instances = cp_sat_stats_cat['stacked_items'].iloc[:cols].tolist()

            instances = (
                list(rl_instances) +
                list(random_instances) +
                list(flb_instances) +
                list(cp_sat_instances)
            )

            c = list(c1) + list(c2) + list(c3) + list(c4)

            pyvista_plot_many_stacks(
                instances,
                W,
                D,
                compactness=c,
                shape=(rows, cols),
                save_path=f'{COMP_PATH}/comparison.pdf'
            )


if __name__ == '__main__':
    main()
