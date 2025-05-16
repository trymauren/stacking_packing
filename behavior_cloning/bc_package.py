import git
import sys
import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env


path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root + '/code')


from stacking_environment.environment import StackingWorld
from stacking_environment.wrappers import (
    NormalizedObservationWrapper, PalletItemToImageWrapper)
from feature_extraction.custom_combined_extractor import (
    CustomCombinedExtractor, CustomCNN)
from callback.callbacks import (
    SummaryWriterCallback, EvalSummaryWriterCallback, NewBestStatsCallback)
from data import cut_1
from run_scripts import logging_utils, rng_utils


rng = np.random.default_rng(0)
env = make_vec_env(
    env_id=StackingWorld,
    rng=rng,
    n_envs=1,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
)

# expert = load_policy(
#     "ppo-huggingface",
#     organization="HumanCompatibleAI",
#     env_name="seals-CartPole-v0",
#     venv=env,
# )

expert = 

rollouts = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_timesteps=None, min_episodes=50),
    rng=rng,
)
transitions = rollout.flatten_trajectories(rollouts)
print(transitions)
# bc_trainer = bc.BC(
#     observation_space=env.observation_space,
#     action_space=env.action_space,
#     demonstrations=transitions,
#     rng=rng,
# )
# bc_trainer.train(n_epochs=1)
# reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
# print("Reward:", reward)

# import numpy as np
# import gymnasium as gym
# from stable_baselines3.common.evaluation import evaluate_policy

# from imitation.algorithms import bc
# from imitation.data import rollout
# from imitation.data.wrappers import RolloutInfoWrapper
# from imitation.policies.serialize import load_policy
# from imitation.util.util import make_vec_env

# rng = np.random.default_rng(0)
# env = make_vec_env(
#     "seals:seals/CartPole-v0",
#     rng=rng,
#     n_envs=1,
#     post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
# )
# expert = load_policy(
#     "ppo-huggingface",
#     organization="HumanCompatibleAI",
#     env_name="seals-CartPole-v0",
#     venv=env,
# )
# rollouts = rollout.rollout(
#     expert,
#     env,
#     rollout.make_sample_until(min_timesteps=None, min_episodes=50),
#     rng=rng,
# )
# transitions = rollout.flatten_trajectories(rollouts)

# bc_trainer = bc.BC(
#     observation_space=env.observation_space,
#     action_space=env.action_space,
#     demonstrations=transitions,
#     rng=rng,
# )
# bc_trainer.train(n_epochs=1)
# reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
# print("Reward:", reward)