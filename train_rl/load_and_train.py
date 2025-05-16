from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecVideoRecorder, VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv
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
    config_path=path_to_root + '/code/train_rl/train_conf/',
    config_name='continue_learn_config',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    LOG_DIR = path_to_root + '/code/log/' + cfg.log_dir
    train_cfg = OmegaConf.load(LOG_DIR + '/.hydra/config.yaml')

    item_getter_function = load_callable(train_cfg.env.item_getter_function)
    activation_fn = load_callable(train_cfg.model.policy_kwargs.activation_fn)
    ft_ext_class = load_callable(
        train_cfg.model.policy_kwargs.features_extractor_class)
    ft_activation_fn = load_callable(train_cfg.model.policy_kwargs.features_extractor_kwargs.activation_fn)
    env_id = load_callable(train_cfg.vec_env.env_id)
    vec_env_cls = load_callable(train_cfg.vec_env.vec_env_cls)
    # wrapper_class = load_callable(train_cfg.vec_env.wrapper_class)

    model_config = OmegaConf.to_container(train_cfg.model, resolve=True)
    vec_env_config = OmegaConf.to_container(train_cfg.vec_env, resolve=True)
    i_g_f_kwargs = OmegaConf.to_container(train_cfg.env.i_g_f_kwargs, resolve=True)

    print('i_g_f_kwargs: ')
    print(i_g_f_kwargs)

    load_model_kwargs = {}
    if cfg.load_model_kwargs is not None:
        load_model_kwargs = OmegaConf.to_container(cfg.load_model_kwargs, resolve=True)

    model_config['policy_kwargs']['activation_fn'] = activation_fn
    model_config['policy_kwargs']['features_extractor_class'] = ft_ext_class
    model_config['policy_kwargs']['features_extractor_kwargs']['activation_fn'] = ft_activation_fn
    vec_env_config['env_id'] = env_id
    vec_env_config['vec_env_cls'] = vec_env_cls
    # vec_env_config['vec_env_cls'] = DummyVecEnv
    vec_env_config['wrapper_class'] = None
    vec_env_config['env_kwargs']['item_getter_function'] = item_getter_function
    vec_env_config['env_kwargs']['i_g_f_kwargs'] = i_g_f_kwargs

    if cfg.env is not None:
        override_env_kwargs = OmegaConf.to_container(cfg.env, resolve=True)
        for key in override_env_kwargs:
            vec_env_config['env_kwargs'][key] = override_env_kwargs[key]

    continue_learn_config = OmegaConf.to_container(cfg.learn, resolve=True)
    eval_config = OmegaConf.to_container(cfg.eval, resolve=True)

    # Setup -----------------------------------------------------------

    LOAD_BEST_MODEL_PATH = LOG_DIR
    LOAD_BEST_ENV_PATH = LOG_DIR
    LOAD_FINAL_MODEL_PATH = LOG_DIR
    LOAD_FINAL_ENV_PATH = LOG_DIR
    if cfg.model_dir:
        LOAD_BEST_MODEL_PATH += f'/saved_model/{cfg.model_dir}/best_model'
        LOAD_BEST_ENV_PATH += f'/saved_model/{cfg.model_dir}/best_vec_normalize.pkl'
        LOAD_FINAL_MODEL_PATH += f'/saved_model/{cfg.model_dir}/final_model'
        LOAD_FINAL_ENV_PATH += f'/saved_model/{cfg.model_dir}/final_vec_normalize.pkl'
    else:
        LOAD_BEST_MODEL_PATH += '/saved_model/best_model'
        LOAD_BEST_ENV_PATH += '/saved_model/best_vec_normalize.pkl'
        LOAD_FINAL_MODEL_PATH += '/saved_model/final_model'
        LOAD_FINAL_ENV_PATH += '/saved_model/final_vec_normalize.pkl'

    SAVE_EVAL_PATH = f'{LOG_DIR}/saved_model/{cfg.save_header}/'
    SAVE_FINAL_MODEL_PATH = f'{LOG_DIR}/saved_model/{cfg.save_header}/final_model.zip'
    SAVE_FINAL_ENV_PATH = f'{LOG_DIR}/saved_model/{cfg.save_header}/final_vec_normalize.pkl'

    rng_utils.set_global_seed(train_cfg.seed)

    # Train -----------------------------------------------------------
    # vec_env_config['seed'] = train_cfg.seed
    eval_env_config = vec_env_config.copy()
    eval_env_config['seed'] += 100 * eval_env_config['n_envs']

    if train_cfg.norm:

        if cfg.load_norm:
            train_env = VecNormalize.load(LOAD_BEST_ENV_PATH, make_vec_env(**vec_env_config))
            eval_env = VecNormalize.load(LOAD_BEST_ENV_PATH, make_vec_env(**eval_env_config))
            train_env.training = True
            eval_env.training = False
            print('Running with loaded normalisation')

        elif cfg.make_norm:
            train_env = VecNormalize(make_vec_env(**vec_env_config), norm_obs=cfg.norm_obs)
            eval_env = VecNormalize(make_vec_env(**eval_env_config), norm_obs=cfg.norm_obs)
            train_env.norm_obs = False
            eval_env.training = False
            print('Running with fresh normalisation')

        else:
            train_env = make_vec_env(**vec_env_config)
            eval_env = make_vec_env(**eval_env_config)
            print('Running without normalisation')

    else:
        train_env = make_vec_env(**vec_env_config)
        eval_env = make_vec_env(**eval_env_config)
        print('Running without normalisation')

    try:
        print('Load best model')
        # loaded_model = MaskablePPO.load(LOAD_BEST_MODEL_PATH, env=train_env, print_system_info=False, **load_model_kwargs)
        loaded_model = MaskablePPO.load(LOAD_BEST_MODEL_PATH, print_system_info=False, **load_model_kwargs)
    except FileNotFoundError:
        print('Didnt find best model, loading last model')
        # loaded_model = MaskablePPO.load(LOAD_FINAL_MODEL_PATH, env=train_env, print_system_info=False, **load_model_kwargs)
        loaded_model = MaskablePPO.load(LOAD_FINAL_MODEL_PATH, print_system_info=False, **load_model_kwargs)
    finally:
        pass

    loaded_model.set_env(train_env)

    print(train_env.env_method('get_reward_coefs'))

    logger = configure(f'{LOG_DIR}/{cfg.save_header}_logs/', train_cfg.log_types)

    loaded_model.set_logger(logger)

    eval_config['eval_freq'] = max(
        eval_config['eval_freq'] //
        vec_env_config['n_envs'],
        1
    )

    save_vec_norm = None
    if (train_cfg.norm and cfg.load_norm) or cfg.make_norm:
        save_vec_norm = SaveVecNormalizeCallback(train_env, SAVE_EVAL_PATH)

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=SAVE_EVAL_PATH,  # save best model
        **eval_config,
        callback_on_new_best=save_vec_norm  # save corresponding Vec_Normalize
    )

    train_callback = SummaryWriterCallback()

    callback = [train_callback, eval_callback]

    loaded_model.learn(**continue_learn_config, callback=callback, reset_num_timesteps=True)

    loaded_model.save(SAVE_FINAL_MODEL_PATH)

    if train_cfg.norm:
        train_env.save(SAVE_FINAL_ENV_PATH)  # save VecNormalize


if __name__ == '__main__':
    main()
