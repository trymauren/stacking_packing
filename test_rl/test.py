import gymnasium as gym
import torch as th
from torch import nn
import numpy as np

import sys
import git
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3.common.vec_env import (
    SubprocVecEnv, VecNormalize
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.ppo_mask import MaskablePPO
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root + '/code')

from stacking_environment.environment import StackingWorld
from callback.callbacks import (
    SummaryWriterCallback, EvalSummaryWriterCallback
)
from data import cut, random_sampling
from run_scripts import logging_utils, rng_utils
from heuristic_algorithms.evaluate_heuristic import evaluate_heuristic
from heuristic_algorithms.obs_act_heuristics import flb_heuristic, random_heuristic
from plotting.plotly_plot_result import plotly_plot_stack
from plotting.plt_plot_result import plt_plot_stack


def main():

    META_CONFIG = {
        'seed': 829,
    }

    TRAIN_ENV_KWARGS = {
        'sim': False,
        'gravity': 0,
        'render_mode': 'rgb_array',
        'renderer': 'Tiny',
        'size': np.asarray([1.2, 0.8]),  # Y, X
        'lookahead': 8,
        'discretisation': 10,
        'max_height': 10_000,
        'item_getter_function': cut.cutting_stock,
        'i_g_f_kwargs': {
            'grid_shape': [12, 8],
            'mass': 10,
            'num_items': 16,
            'min_': [2, 2, 2],
            'max_': [6, 4, 4],
            'sort': True,
        },
        'step_compact_rew_coef': 0,
        'term_compact_rew_coef': 0,
        'num_hetero_instance_samples': -1,
        'flat_action_space': False
    }

    MAKE_TRAIN_ENV_KWARGS = {
        'env_id': StackingWorld,
        'seed': META_CONFIG['seed'],
        'n_envs': 1,
        'vec_env_cls': SubprocVecEnv,
        'env_kwargs': TRAIN_ENV_KWARGS,
    }

    TEST_KWARGS = {
        'n_eval_episodes': 160,
        'render': False,
        'deterministic': True,
    }

    """ Setup ----------------------------------------------------- """
    # LOG_DIR = path_to_root + '/code/log/data_randomness_term_reward/2025-02-26_11-47-33/'
    # LOG_DIR += f'data_func=random_sampling.random_sample,env.num_hetero_instance_samples=-1,env.step_compact_rew_coef=0,env.term_compact_rew_coef=1,log_header=data_randomness_term_reward,model.device=cpu,n_envs=128,num_items={lookahead}'
    LOG_DIR = path_to_root + '/code/log/decoupled_action_performance/2025-04-14_18-49-02/'
    LOG_DIR += f'data_func=cut.cutting_stock,env.flat_action_space=False,env.lookahead=8,env.term_compact_rew_coef=1,learn.total_timesteps=10_000_000,model.ent_coef=0.0,model.learning_rate=0.001,model.n_epochs=15,model.policy_kwargs.net_arch=[64, 64],n_envs=4'
    BEST_MODEL_PATH = LOG_DIR + '/saved_model/best_model'
    LAST_MODEL_PATH = LOG_DIR + '/saved_model/final_model'
    ENV_NORMALIZE_PATH = LOG_DIR + '/saved_model/vec_normalize.pkl'
    VIDEO_DIR = LOG_DIR + '/recordings'
    # VIDEO_LENGTH = TEST_KWARGS['n_eval_episodes'] * (
    #     TRAIN_ENV_KWARGS['num_items'] / MAKE_TRAIN_ENV_KWARGS['n_envs'])

    rng_utils.set_global_seed(META_CONFIG['seed'])

    env = make_vec_env(**MAKE_TRAIN_ENV_KWARGS)
    env = VecNormalize.load(ENV_NORMALIZE_PATH, env)
    env.training = False
    env.norm_reward = False

    try:
        print('Load best model')
        loaded_model = MaskablePPO.load(BEST_MODEL_PATH, env=env, print_system_info=True)
    except FileNotFoundError:
        print('Didnt find best model, loading last model')
        loaded_model = MaskablePPO.load(LAST_MODEL_PATH, env=env, print_system_info=True)
    finally:
        pass

    test_callback = EvalSummaryWriterCallback()
    rew, _ = evaluate_policy(
        loaded_model,
        # video_env,
        loaded_model.get_env(),
        callback=test_callback,
        return_episode_rewards=True,
        **TEST_KWARGS,
    )

    print(rew)
    print(np.mean(rew))
    ppo_stats = test_callback.get_stats()
    ppo_mean_stats = test_callback.get_mean_stats()
    print(ppo_stats)
    print(ppo_mean_stats)
    # discretisation = TRAIN_ENV_KWARGS['discretisation']
    # size = TRAIN_ENV_KWARGS['size']

    # for i in range(10):
    #     plotly_plot_stack(
    #         test_callback.get_stacked_instances()[i],
    #         *np.asarray(discretisation * size, dtype=int)[::-1],
    #         show=True
    #     )


#     env = make_vec_env(**MAKE_TRAIN_ENV_KWARGS, wrapper_class=Monitor)

#     try:
#         loaded_model = MaskablePPO.load(BEST_MODEL_PATH, env=env)
#     except FileNotFoundError:
#         print('Didnt find best model, loading last model')
#         loaded_model = MaskablePPO.load(LAST_MODEL_PATH, env=env)
#     finally:
#         pass

#     env = loaded_model.get_env()

#     video_env = VecVideoRecorder(
#         env,
#         video_folder=VIDEO_DIR,
#         record_video_trigger=lambda x: True,
#         name_prefix='random'
#     )

#     test_callback = EvalSummaryWriterCallback()
#     evaluate_heuristic(
#         random_heuristic,
#         video_env,
#         # env,
#         **TEST_KWARGS,
#         callback=test_callback,
#         heuristic_kwargs={'rng': env.env_method('get_np_random')}
#     )
#     random_stats = test_callback.get_stats()
#     random_mean_stats = test_callback.get_mean_stats()

#     video_env.close()
#     env.close()

#     env = make_vec_env(**MAKE_TRAIN_ENV_KWARGS, wrapper_class=Monitor)

#     try:
#         loaded_model = MaskablePPO.load(BEST_MODEL_PATH, env=env)
#     except FileNotFoundError:
#         print('Didnt find best model, loading last model')
#         loaded_model = MaskablePPO.load(LAST_MODEL_PATH, env=env)
#     finally:
#         pass

#     env = loaded_model.get_env()

#     video_env = VecVideoRecorder(
#         env,
#         video_folder=VIDEO_DIR,
#         record_video_trigger=lambda x: True,
#         name_prefix='heuristic'
#     )

#     test_callback = EvalSummaryWriterCallback()
#     evaluate_heuristic(
#         flb_heuristic,
#         video_env,
#         # env,
#         callback=test_callback,
#         **TEST_KWARGS,
#     )
#     heuristic_stats = test_callback.get_stats()
#     heuristic_mean_stats = test_callback.get_mean_stats()

#     video_env.close()
#     env.close()

#     ppo_mean_stats['stack_height'] = ppo_mean_stats['stack_height']/32
#     random_mean_stats['stack_height'] = random_mean_stats['stack_height']/32
#     heuristic_mean_stats['stack_height'] = heuristic_mean_stats['stack_height']/32

#     data = pd.DataFrame({
#         'PPO': ppo_mean_stats,
#         'Random': random_mean_stats,
#         'Heuristic': heuristic_mean_stats
#     })

#     # Create a grouped bar chart
#     ax = data.plot(kind='bar', figsize=(8, 5))
#     for container in ax.containers:
#         ax.bar_label(container)
#     ax.set_title('3_000_000 steps, with physics sim and area mask (0.5)')
#     ax.set_ylabel('Value')
#     plt.xticks(rotation=0)  # Keep x-labels horizontal for readability
#     plt.legend(loc='best')
#     plt.tight_layout()
#     plt.savefig(LOG_DIR + 'results.png', dpi=600)


if __name__ == '__main__':
    main()

    # # Alternatively, you can use the MuJoCo equivalent "HalfCheetah-v4"
    # vec_env = make_vec_env("HalfCheetahBulletEnv-v0", n_envs=1)
    # # Automatically normalize the input features and reward
    # vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # model = PPO("MlpPolicy", vec_env)
    # model.learn(total_timesteps=2000)

    # # Don't forget to save the VecNormalize statistics when saving the agent
    # log_dir = Path("/tmp/")
    # model.save(log_dir / "ppo_halfcheetah")
    # stats_path = log_dir / "vec_normalize.pkl"
    # vec_env.save(stats_path)

    # # To demonstrate loading
    # del model, vec_env

    # # Load the saved statistics
    # vec_env = make_vec_env("HalfCheetahBulletEnv-v0", n_envs=1)
    # vec_env = VecNormalize.load(stats_path, vec_env)
    # #  do not update them at test time
    # vec_env.training = False
    # # reward normalization is not needed at test time
    # vec_env.norm_reward = False
