from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecVideoRecorder, VecNormalize
from torch import nn
from torch.distributions import Distribution
import git
import gymnasium as gym
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import sys
import torch as th
from typing import Callable

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root + '/code')

from callback.callbacks import SummaryWriterCallback, SaveVecNormalizeCallback
from run_scripts import logging_utils, rng_utils
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from importlib import import_module
from feature_extraction.feature_extractors import FlattenDictExtractor


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value:
    :return: current learning rate depending on remaining progress
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func


Distribution.set_default_validate_args(False)  # fix for something?

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


# A helper to convert a dotted path string to a callable.
def load_callable(dotted_path: str):
    module_path, callable_name = dotted_path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, callable_name)


@hydra.main(
    config_path=path_to_root + '/code/train_rl/train_conf',
    config_name='stability_curriculum',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # policy = load_callable(cfg.model.policy)
    item_getter_function = load_callable(cfg.env.item_getter_function)
    activation_fn = load_callable(cfg.model.policy_kwargs.activation_fn)
    ft_ext_class = load_callable(
        cfg.model.policy_kwargs.features_extractor_class)
    ft_activation_fn = load_callable(cfg.model.policy_kwargs.features_extractor_kwargs.activation_fn)
    env_id = load_callable(cfg.vec_env.env_id)
    vec_env_cls = load_callable(cfg.vec_env.vec_env_cls)
    # wrapper_class = load_callable(cfg.vec_env.wrapper_class)

    eval_config_compact = OmegaConf.to_container(cfg.eval_compact, resolve=True)
    eval_config_support = OmegaConf.to_container(cfg.eval_support, resolve=True)
    eval_config_stable = OmegaConf.to_container(cfg.eval_stable, resolve=True)
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    vec_env_config = OmegaConf.to_container(cfg.vec_env, resolve=True)
    i_g_f_kwargs = OmegaConf.to_container(cfg.env.i_g_f_kwargs, resolve=True)

    print('i_g_f_kwargs: ')
    print(i_g_f_kwargs)

    model_config['policy_kwargs']['activation_fn'] = activation_fn
    model_config['policy_kwargs']['features_extractor_class'] = ft_ext_class
    model_config['policy_kwargs']['features_extractor_kwargs']['activation_fn'] = ft_activation_fn
    vec_env_config['env_id'] = env_id
    vec_env_config['vec_env_cls'] = vec_env_cls
    # vec_env_config['wrapper_class'] = wrapper_class
    vec_env_config['env_kwargs']['item_getter_function'] = item_getter_function
    vec_env_config['env_kwargs']['i_g_f_kwargs'] = i_g_f_kwargs

    # Setup -----------------------------------------------------------
    LOG_DIR = path_to_root + '/code/' + HydraConfig.get().run.dir
    CONFIG_PATH = LOG_DIR + '/saved_config'
    SAVE_EVAL_PATH_COMPACT = LOG_DIR + '/saved_model/compact/'
    SAVE_EVAL_PATH_SUPPORT = LOG_DIR + '/saved_model/support/'
    SAVE_EVAL_PATH_STABLE = LOG_DIR + '/saved_model/stable/'
    SAVE_FINAL_MODEL_PATH_COMPACT = LOG_DIR + '/saved_model/compact/final_model.zip'
    SAVE_FINAL_MODEL_PATH_SUPPORT = LOG_DIR + '/saved_model/support/final_model.zip'
    SAVE_FINAL_MODEL_PATH_STABLE = LOG_DIR + '/saved_model/stable/final_model.zip'
    SAVE_FINAL_ENV_PATH_COMPACT = LOG_DIR + '/saved_model/compact/final_vec_normalize.pkl'
    SAVE_FINAL_ENV_PATH_SUPPORT = LOG_DIR + '/saved_model/support/final_vec_normalize.pkl'
    SAVE_FINAL_ENV_PATH_STABLE = LOG_DIR + '/saved_model/stable/final_vec_normalize.pkl'

    rng_utils.set_global_seed(cfg.seed)

    # Train -----------------------------------------------------------
    eval_env_config = vec_env_config.copy()
    eval_env_config['seed'] += 100 * eval_env_config['n_envs']

    if cfg.norm:
        train_env = VecNormalize(
            make_vec_env(**vec_env_config),
            gamma=1, norm_reward=cfg.norm_rew,
            norm_obs=cfg.norm_obs, clip_obs=100
        )
        eval_env = VecNormalize(
            make_vec_env(**eval_env_config),
            gamma=1, norm_reward=cfg.norm_rew,
            norm_obs=cfg.norm_obs, clip_obs=100
        )
        eval_env.training = False

    else:
        train_env = make_vec_env(**vec_env_config)
        eval_env = make_vec_env(**eval_env_config)

    logger = configure(LOG_DIR, cfg.log_types)

    model = MaskablePPO(
        env=train_env,
        **model_config,
    )

    model.set_logger(logger)

    train_callback = SummaryWriterCallback()

    # Compactness =============================================================

    eval_config_compact['eval_freq'] = max(
        eval_config_compact['eval_freq'] //
        vec_env_config['n_envs'],
        1
    )

    rew_coef_compact = OmegaConf.to_container(cfg.reward_compact, resolve=True)
    train_env.env_method('set_reward_coefs', **rew_coef_compact)
    eval_env.env_method('set_reward_coefs', **rew_coef_compact)

    learn_compact = OmegaConf.to_container(cfg.learn_compact, resolve=True)

    if learn_compact['total_timesteps'] > 0:
        save_vec_norm = None
        if cfg.norm:
            save_vec_norm = SaveVecNormalizeCallback(
                train_env, SAVE_EVAL_PATH_COMPACT
            )

        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=SAVE_EVAL_PATH_COMPACT,  # save best model
            **eval_config_compact,
            callback_on_new_best=save_vec_norm  # corresponding VecNormalize
        )

        callbacks = [train_callback, eval_callback]
        model.learn(
            **learn_compact, callback=callbacks, tb_log_name='tb_logger'
        )
        model.save(SAVE_FINAL_MODEL_PATH_COMPACT)
        if cfg.norm:
            train_env.save(SAVE_FINAL_ENV_PATH_COMPACT)  # save VecNormalize

    # Support =============================================================

    eval_config_support['eval_freq'] = max(
        eval_config_support['eval_freq'] //
        vec_env_config['n_envs'],
        1
    )

    rew_coef_support = OmegaConf.to_container(cfg.reward_support, resolve=True)
    train_env.env_method('set_reward_coefs', **rew_coef_support)
    eval_env.env_method('set_reward_coefs', **rew_coef_support)

    learn_support = OmegaConf.to_container(cfg.learn_support, resolve=True)

    if learn_support['total_timesteps'] > 0:

        train_env.env_method('set_minimal_support', 0.5)
        eval_env.env_method('set_minimal_support', 0.5)
        model.set_env(train_env, force_reset=True)

        save_vec_norm = None
        if cfg.norm:
            save_vec_norm = SaveVecNormalizeCallback(
                train_env, SAVE_EVAL_PATH_SUPPORT
            )

        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=SAVE_EVAL_PATH_SUPPORT,  # save best model
            **eval_config_support,
            callback_on_new_best=save_vec_norm  # corresponding VecNormalize
        )

        callbacks = [train_callback, eval_callback]
        model.learn(
            **learn_support, callback=callbacks, tb_log_name='tb_logger',
            reset_num_timesteps=False
        )
        model.save(SAVE_FINAL_MODEL_PATH_SUPPORT)
        if cfg.norm:
            train_env.save(SAVE_FINAL_ENV_PATH_SUPPORT)  # save VecNormalize

    # Stability ===============================================================

    eval_config_stable['eval_freq'] = max(
        eval_config_stable['eval_freq'] //
        vec_env_config['n_envs'],
        1
    )

    rew_coef_stable = OmegaConf.to_container(cfg.reward_stable, resolve=True)
    train_env.env_method('set_reward_coefs', **rew_coef_stable)
    eval_env.env_method('set_reward_coefs', **rew_coef_stable)

    learn_stable = OmegaConf.to_container(cfg.learn_stable, resolve=True)

    if learn_stable['total_timesteps'] > 0:

        train_env.env_method('set_minimal_support', 0)
        eval_env.env_method('set_minimal_support', 0)

        train_env.env_method(
            'set_sim',
            sim=True,
            gravity=1,
            render_mode='rgb_array',
            renderer='Tiny',
            item_lateral_friction=0.9,
            item_spin_friction=0.9,
            item_restitution=1,
            item_collision_margin=0.0,
            move_pallet=0,
        )

        eval_env.env_method(
            'set_sim',
            sim=True,
            gravity=1,
            render_mode='rgb_array',
            renderer='Tiny',
            item_lateral_friction=0.9,
            item_spin_friction=0.9,
            item_restitution=1,
            item_collision_margin=0.0,
            move_pallet=0,
        )

        model.set_env(train_env, force_reset=True)

        save_vec_norm = None
        if cfg.norm:
            save_vec_norm = SaveVecNormalizeCallback(
                train_env, SAVE_EVAL_PATH_STABLE
            )

        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=SAVE_EVAL_PATH_STABLE,  # save best model
            **eval_config_stable,
            callback_on_new_best=save_vec_norm  # corresponding VecNormalize
        )

        callbacks = [train_callback, eval_callback]
        model.learn(
            **learn_stable, callback=callbacks, tb_log_name='tb_logger',
            reset_num_timesteps=False
        )
        model.save(SAVE_FINAL_MODEL_PATH_STABLE)
        if cfg.norm:
            train_env.save(SAVE_FINAL_ENV_PATH_STABLE)  # save VecNormalize


if __name__ == '__main__':
    main()
