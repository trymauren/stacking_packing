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
    config_name='config',
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

    learn_config = OmegaConf.to_container(cfg.learn, resolve=True)
    eval_config = OmegaConf.to_container(cfg.eval, resolve=True)
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

    if cfg.save_header:
        SAVE_EVAL_PATH = LOG_DIR + f'/saved_model/{cfg.save_header}/'
        SAVE_FINAL_MODEL_PATH = LOG_DIR + f'/saved_model/{cfg.save_header}/final_model.zip'
        SAVE_FINAL_ENV_PATH = LOG_DIR + f'/saved_model/{cfg.save_header}/final_vec_normalize.pkl'
        logger = configure(LOG_DIR + f'/{cfg.save_header}_logs/', cfg.log_types)
    else:
        SAVE_EVAL_PATH = LOG_DIR + f'/saved_model/'
        SAVE_FINAL_MODEL_PATH = LOG_DIR + f'/saved_model/final_model.zip'
        SAVE_FINAL_ENV_PATH = LOG_DIR + f'/saved_model/final_vec_normalize.pkl'
        logger = configure(LOG_DIR + f'/train_logs/', cfg.log_types)

    rng_utils.set_global_seed(cfg.seed)

    # Train -----------------------------------------------------------
    eval_env_config = vec_env_config.copy()
    eval_env_config['seed'] += 100 * eval_env_config['n_envs']
    if cfg.norm:
        train_env = VecNormalize(make_vec_env(**vec_env_config), norm_reward=cfg.norm_rew)
        eval_env = VecNormalize(make_vec_env(**eval_env_config), norm_reward=cfg.norm_rew)
        eval_env.training = False
    else:
        train_env = make_vec_env(**vec_env_config)
        eval_env = make_vec_env(**eval_env_config)

    model = MaskablePPO(
        env=train_env,
        **model_config,
    )

    model.set_logger(logger)

    eval_config['eval_freq'] = max(  # will not override saved config, which is good
        eval_config['eval_freq'] //
        vec_env_config['n_envs'],
        1
    )

    save_vec_norm = None
    if cfg.norm:
        save_vec_norm = SaveVecNormalizeCallback(train_env, SAVE_EVAL_PATH)

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=SAVE_EVAL_PATH,  # save best model
        **eval_config,
        callback_on_new_best=save_vec_norm  # save corresponding Vec_Normalize
    )

    train_callback = SummaryWriterCallback()

    callback = [train_callback, eval_callback]

    model.learn(**learn_config, callback=callback)

    model.save(SAVE_FINAL_MODEL_PATH)

    if cfg.norm:
        train_env.save(SAVE_FINAL_ENV_PATH)  # save the VecNormalize for test


if __name__ == '__main__':
    main()
