from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import VecNormalize

import pandas as pd
import numpy as np
from statistics import fmean
from copy import copy


# https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#directly-accessing-the-summary-writer
class SummaryWriterCallback(BaseCallback):
    def __init__(self, verbose=0, log_freq=1000, **kwargs):
        super(SummaryWriterCallback, self).__init__(
            verbose=verbose, **kwargs)
        # self.episode_rewards = []
        self._log_freq = log_freq
        # self.custom_counter = 0

    def _on_training_start(self):
        output_formats = self.logger.output_formats
        try:
            self.tb_formatter = next(
                formatter for formatter in output_formats if isinstance(
                    formatter, TensorBoardOutputFormat)
            )
        except StopIteration:
            raise ValueError(('No TensorBoardOutputFormat found.'
                              'Ensure tensorboard is enabled in the logger.'))

    def _on_step(self) -> bool:

        rewards = self.locals['rewards']
        infos = self.locals['infos']
        dones = self.locals['dones']

        for ix, (info, done) in enumerate(zip(infos, dones)):
            # if self.n_calls % self._log_freq == 0:
            if done:
                reward = [
                    f'train-reward/reward_env_{ix}',
                    rewards[ix],
                    self.n_calls
                ]
                stack_stability = [
                    f'train-stability/env_{ix}_stability_bool',
                    info['stack_stability_bool'],
                    self.n_calls
                ]
                self.logger.record(*reward)
                self.logger.record(*stack_stability)

                stack_height = [
                    f'train-height/env_{ix}_stack_height',
                    info['stack_height'],
                    self.n_calls
                ]
                stack_gap_ratio = [
                    f'train-gap-ratio/env_{ix}_stack_gap_ratio',
                    info['stack_gap_ratio'],
                    self.n_calls
                ]
                stack_compactness = [
                    f'train-compactness/env_{ix}_stack_compactness',
                    info['stack_compactness'],
                    self.n_calls
                ]
                height_upper_bound = [
                    f'train_height_bounds/env_{ix}_height_upper_bound',
                    info['height_upper_bound'],
                    self.n_calls
                ]
                height_lower_bound = [
                    f'train_height_bounds/env_{ix}_height_lower_bound',
                    info['height_lower_bound'],
                    self.n_calls
                ]
                # compact_upper_bound = [
                #     f'train_compact_bounds/env_{ix}_compactness_upper_bound',
                #     info['compact_upper_bound'],
                #     self.n_calls
                # ]
                # compact_lower_bound = [
                #     f'train_compact_bounds/env_{ix}_compactness_lower_bound',
                #     info['compact_lower_bound'],
                #     self.n_calls
                # ]

                self.logger.record(*stack_height)
                self.logger.record(*stack_gap_ratio)
                self.logger.record(*stack_compactness)
                self.logger.record(*height_upper_bound)
                self.logger.record(*height_lower_bound)
                # self.logger.record(*compact_upper_bound)
                # self.logger.record(*compact_lower_bound)

        return True

    def _on_training_end(self):
        pass


# Not used
class EvalCallback(EventCallback):
    def __init__(self, verbose=0, **kwargs):
        super(NewBestStatsCallback, self).__init__(
            verbose=verbose, **kwargs)

    def _on_step(self) -> bool:

        rewards = self.locals['rewards']
        infos = self.locals['infos']
        dones = self.locals['dones']

        for ix, (info, done) in enumerate(zip(infos, dones)):
            reward = [
                f'eval-reward/reward_env_{ix}',
                rewards[ix],
                self.n_calls
            ]
            stack_stability = [
                f'eval-stability/env_{ix}_stability_bool',
                info['stack_stability_bool'],
                self.n_calls
            ]
            if done:
                stack_height = [
                    f'eval-height/env_{ix}_stack_height',
                    info['stack_height'],
                    self.n_calls
                ]
                stack_gap_ratio = [
                    f'eval-gap-ratio/env_{ix}_stack_gap_ratio',
                    info['stack_gap_ratio'],
                    self.n_calls
                ]
                height_upper_bound = [
                    f'eval_bounds/env_{ix}_height_upper_bound',
                    info['height_upper_bound'],
                    self.n_calls
                ]
                height_lower_bound = [
                    f'eval_bounds/env_{ix}_height_lower_bound',
                    info['height_lower_bound'],
                    self.n_calls
                ]
                # compact_upper_bound = [
                #     f'eval_bounds/env_{ix}_compactness_upper_bound',
                #     info['compact_upper_bound'],
                #     self.n_calls
                # ]
                # compact_lower_bound = [
                #     f'eval_bounds/env_{ix}_compactness_lower_bound',
                #     info['compact_lower_bound'],
                #     self.n_calls
                # ]

                self.logger.record(*stack_height)
                self.logger.record(*stack_gap_ratio)
                self.logger.record(*height_upper_bound)
                self.logger.record(*height_lower_bound)
                # self.logger.record(*compact_upper_bound)
                # self.logger.record(*compact_lower_bound)
            self.logger.record(*reward)
            self.logger.record(*stack_stability)

        return True


class EvalSummaryWriterCallback():
    def __init__(self):
        self.df = None
        self.stats_ready = 0
        self.columns = [
            'stack_stability_bool',
            'stack_gap_ratio',
            'stack_compactness',
            'stack_height',
            'height_upper_bound',
            'height_lower_bound',
            'stacked_items',
            'n_to_stack',
            'n_stacked',
            'completed_act_ratio',
        ]
        self.logger = {col: [] for col in self.columns}
        self.stacked_instances = []

    def __call__(self, *args, **kwargs):

        def eval_callback(locals_, globals_, *args, **kwargs):
            infos = locals_['infos']
            dones = locals_['dones']
            for info, done in zip(infos, dones):
                if done:
                    for col in self.columns:
                        self.logger[col].append(info[col])
                    # self.stacked_instances.append(info['stacked_items'])
        return eval_callback(*args, **kwargs)

    def get_stats(self):
        if self.stats_ready:
            return self.df
        df = pd.DataFrame(data=self.logger, columns=self.columns)
        self.df = df
        self.stats_ready = 1
        return self.df

    def get_mean_stats(self):
        if self.stats_ready:
            return self.df.mean(numeric_only=True)
        df = pd.DataFrame(data=self.logger, columns=self.columns)
        self.df = df
        self.stats_ready = 1
        return self.df.mean(numeric_only=True)

    def get_stacked_instances(self):
        if self.stats_ready:
            return self.df['stacked_items']
        df = pd.DataFrame(data=self.logger, columns=self.columns)
        self.df = df
        self.stats_ready = 1
        return self.df['stacked_items']


class SaveVecNormalizeCallback(BaseCallback):

    def __init__(self, v_env: VecNormalize, save_path: str):
        super().__init__()
        self.v_env = v_env
        self.save_path = save_path + 'best_vec_normalize.pkl'

    def _on_step(self) -> bool:
        self.v_env.save(self.save_path)
        return True


# class SaveVecNormalizeCallback(BaseCallback):
#     """
#     Callback for saving a VecNormalize wrapper every ``save_freq`` steps

#     :param save_freq: (int)
#     :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
#     :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
#         only one file will be kept.
#     """

#     def __init__(self, save_freq: int, save_path: str, name_prefix: Optional[str] = None, verbose: int = 0):
#         super(SaveVecNormalizeCallback, self).__init__(verbose)
#         self.save_freq = save_freq
#         self.save_path = save_path
#         self.name_prefix = name_prefix

#     def _init_callback(self) -> None:
#         # Create folder if needed
#         if self.save_path is not None:
#             os.makedirs(self.save_path, exist_ok=True)

#     def _on_step(self) -> bool:
#         if self.n_calls % self.save_freq == 0:
#             if self.name_prefix is not None:
#                 path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
#             else:
#                 path = os.path.join(self.save_path, "vecnormalize.pkl")
#             if self.model.get_vec_normalize_env() is not None:
#                 self.model.get_vec_normalize_env().save(path)
#                 if self.verbose > 1:
#                     print(f"Saving VecNormalize to {path}")
#         return True
