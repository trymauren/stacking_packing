import gymnasium as gym
import numpy as np


class NormalizedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        highest = np.amax(obs['pallet'])
        if highest:
            obs['pallet'] = np.divide(obs['pallet'], highest)
            obs['upcoming_items'] = np.divide(obs['upcoming_items'], highest)
        return obs


class PalletItemToImageWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        new_observations = self.setup_new_obs(env)

        self.observation_space = gym.spaces.Box(
            low=0, high=10_000, shape=new_observations.shape
        )

    def setup_new_obs(self, env):

        pallet = np.expand_dims(
            env.observation_space['pallet'].sample(), axis=0)

        upcoming_items = np.expand_dims(
            env.observation_space['upcoming_items'].sample(), axis=0
        )

        upcoming_items = np.stack(
            [np.broadcast_to(upcoming_items[:, :, i], pallet.shape) for i in range(3)],
            axis=1
        )

        new_observations = np.concatenate(
            (upcoming_items, np.expand_dims(pallet, axis=0)),
            axis=1
        )
        return new_observations.squeeze(axis=0)

    def observation(self, obs):

        pallet = np.expand_dims(obs['pallet'], axis=0)

        upcoming_items = np.expand_dims(obs['upcoming_items'], axis=0)

        upcoming_items = np.stack(
            [np.broadcast_to(upcoming_items[:, :, i], pallet.shape) for i in range(3)],
            axis=1
        )

        new_observations = np.concatenate(
            (upcoming_items, np.expand_dims(pallet, axis=0)),
            axis=1
        )

        return np.asarray(new_observations.squeeze(axis=0), np.uint8)
