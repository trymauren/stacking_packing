import git
import sys
import time
import numpy as np
from collections import deque
from random import shuffle
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
sys.path.append(path_to_root + '/code')

from stacking_objects.stack import Stack
from stacking_objects.bullet_env import PyBulletEnv
from stacking_objects.item import Item
from bounds.upper_bounds import (
    get_height_upper_bounds,
    get_compact_upper_bounds
)
from bounds.lower_bounds import (
    get_height_lower_bounds,
    get_compact_lower_bounds
)
from data import data_loaders


SET1_CMAP = plt.cm.get_cmap('Set1')


class StackingWorld(gym.Env):

    def __init__(
        self,
        sim=False,
        gravity=0,
        render_mode='rgb_array',
        renderer='Tiny',
        size: np.array(int) = np.asarray((10, 10)),
        discretisation: int = 1,
        max_height=np.inf,
        lookahead=1,
        item_getter_function=False,
        i_g_f_kwargs=False,
        item_spin_friction=None,  # 0.9
        item_lateral_friction=None,  # 0.9
        item_restitution=None,  # 1
        item_collision_margin: float = 0.0,
        move_pallet=False,
        step_compact_rew_coef=0,
        term_compact_rew_coef=0,
        term_compact_rew_exp=0,
        stable_rew_coef=0,
        mask_non_supported_items=0,
        num_hetero_instance_samples=-1,
        flat_action_space=True,
        minimal_support=0,
    ):
        self.sim = sim
        self.render_mode = render_mode
        self.renderer = renderer
        self.size = size
        self.max_height = max_height
        self.lookahead = lookahead
        self.discretisation = discretisation
        self.gravity = gravity
        self.num_hetero_instance_samples = num_hetero_instance_samples
        self.set_item_loader(item_getter_function, i_g_f_kwargs)
        self.step_compact_rew_coef = step_compact_rew_coef
        self.term_compact_rew_coef = term_compact_rew_coef
        self.term_compact_rew_exp = term_compact_rew_exp
        self.stable_rew_coef = stable_rew_coef
        self.mask_non_supported_items = mask_non_supported_items
        self.flat_action_space = flat_action_space
        self.minimal_support = minimal_support

        self.sim_env = 0
        if self.sim:
            self.item_spin_friction = item_spin_friction
            self.item_lateral_friction = item_lateral_friction
            self.item_restitution = item_restitution
            self.item_collision_margin = item_collision_margin
            self.move_pallet = move_pallet
            self.stable_rew_coef = stable_rew_coef

        self.count = 0
        self.items = []

        self.grid_shape = grid_shape = np.ceil(
            np.asarray(size) * discretisation).astype(int)

        self.stack = Stack(grid_shape, self.max_height)

        low = [[*[0 for _ in range(3)], 0.0]] * self.lookahead
        high = [[*grid_shape, max_height, 10000]] * self.lookahead

        self.observation_space = spaces.Dict(
            {
                'pallet': spaces.Box(
                    low=0,
                    high=self.max_height,
                    shape=grid_shape,
                    dtype=np.float32),
                'upcoming_items': spaces.Box(
                    low=np.array(low),  # Minimums for [x, y, z, mass]
                    high=np.array(high),  # Maximums for [x, y, z, mass]
                    shape=(self.lookahead, 3 + 1),  # x +y + z + mass = 4
                    dtype=np.float32)
                }
            )

        if flat_action_space:
            flattened_size = 1
            for dim in grid_shape:
                flattened_size *= dim
            self.action_space = spaces.Discrete(flattened_size)
        else:
            self.action_space = spaces.MultiDiscrete(np.array([*grid_shape]))

    def _add_pallet_to_sim(self):

        half_extents = np.asarray((*(np.flip(self.grid_shape)/2), 0.495))
        mid_position = np.asarray((*(np.flip(self.grid_shape)/2), -0.5))

        self.sim_env.create_box(
            body_name='pallet',
            half_extents=half_extents,
            mass=0,
            position=mid_position,
            rgba_color=np.array((153, 102, 51)),  # wooden pallet color?
            lateral_friction=1,
            spinning_friction=1,
            restitution=0.1,
        )

    def _create_sim_env(self):
        sim_env = PyBulletEnv(
            render_mode=self.render_mode, renderer=self.renderer)
        sim_env.reset_sim(gravity=self.gravity)
        sim_env.loadURDF('plane', fileName='plane.urdf')
        # sim_env.loadURDF(
        #     'plane', fileName=path_to_root + '/code/urdf/custom_plane.urdf')
        # sim_env.loadURDF(
        #     'pallet', fileName=path_to_root + '/code/urdf/euro_pallet.urdf')

        # sim_env.place_visualizer(
        #     target_position=(*self.grid_shape/2, 0),
        #     distance=15,
        #     yaw=0,
        #     pitch=-30,
        # )
        self.sim_env = sim_env

        # pallet_id = self.sim_env._bodies_idx['pallet']
        # joint_type = self.sim_env.physics_client.JOINT_FIXED
        # constraint_id = self.sim_env.physics_client.createConstraint(
        #     pallet_id,
        #     parentLinkIndex=-1,
        #     childBodyUniqueId=-1,  # no body
        #     childLinkIndex=-1,  # the base
        #     jointType=joint_type,
        #     jointAxis=[0, 0, 0],
        #     parentFramePosition=[0, 0, 0],
        #     childFramePosition=[0, 0, 0]
        # )
        # self.sim_env.pallet_constraint = constraint_id

        # self._add_pallet_to_sim()

    def _reset_sim_env(self):
        self.sim_env.reset_sim(gravity=self.gravity)
        self.sim_env.loadURDF('plane', fileName='plane.urdf')
        # self.sim_env.loadURDF(
        #     'plane', fileName=path_to_root + '/code/urdf/custom_plane.urdf')
        # self.sim_env.loadURDF(
        #     'pallet', fileName=path_to_root + '/code/urdf/euro_pallet.urdf')

        # pallet_id = self.sim_env._bodies_idx['pallet']
        # joint_type = self.sim_env.physics_client.JOINT_FIXED
        # constraint_id = self.sim_env.physics_client.createConstraint(
        #     pallet_id,
        #     parentLinkIndex=-1,
        #     childBodyUniqueId=-1,  # no body
        #     childLinkIndex=-1,  # the base
        #     jointType=joint_type,
        #     jointAxis=[0, 0, 0],
        #     parentFramePosition=[0, 0, 0],
        #     childFramePosition=[0, 0, 0]
        # )
        # self.sim_env.pallet_constraint = constraint_id

    def schedule_items(self):
        del self.items
        items = self.item_getter_function()
        self.num_items_this_episode = len(items)
        n_rows_to_pad = max(self.lookahead - items.shape[0], 0)
        pad_width = [(0, n_rows_to_pad), (0, 0)]
        self.items = np.pad(items, pad_width, mode='constant')

    def step(self, action):

        terminations = False
        truncations = False
        infos = {}

        item = self.items[0]

        item = Item(item[:-1], item[-1])

        if self.flat_action_space:
            _y, _x, = np.unravel_index(action, self.grid_shape)
        else:
            _y, _x = action
        item.y, item.x = _y, _x

        feasible = self.stack.is_out_of_bounds(item)

        stable = 1
        previous_stable = 0
        current_stable = 0
        r = 0

        if feasible:
            previous_gap_ratio = self.stack.gap_ratio()
            if self.sim:
                self.stack.add_z_pos_to_item(item)
                self._add_item_to_sim(item)
                previous_item = self.stack.last_item_placed()
                previous_stable = self._check_stability(previous_item)
                if previous_stable:
                    current_stable = self._check_stability(item)
                    self.count += current_stable

                stable = int(previous_stable * current_stable)
                terminations = not stable
                if stable:
                    self.stack.place(item)
                    current_gap_ratio = self.stack.gap_ratio()
                    r = previous_gap_ratio - current_gap_ratio

            elif self.minimal_support:
                stable = self._heuristic_check_stability(item)
                terminations = not stable
                if stable:
                    self.stack.place(item)
                    current_gap_ratio = self.stack.gap_ratio()
                    r = previous_gap_ratio - current_gap_ratio
                    self.count += 1
            else:
                self.stack.place(item)
                current_gap_ratio = self.stack.gap_ratio()
                r = previous_gap_ratio - current_gap_ratio
                self.count += 1

        else:
            print('Received invalid action')

        self.items[:-1] = self.items[1:]  # shifting towards [0]
        self.items[-1] = np.zeros(4)  # replacing last element by zeros
        observations = self._gather_observations()

        compactness = self.stack.compactness()

        completed_stacking = False
        if (self.items[0] == 0).all():
            terminations = True
            completed_stacking = True

        failure = int(not stable)  # 0 if stable=True, 1 if stable=False
        terminal_success = int(stable and completed_stacking)
        rewards = (
            terminal_success * self.term_compact_rew_coef * (
                    compactness ** self.term_compact_rew_exp
            )
            + self.step_compact_rew_coef * r
            - failure * self.stable_rew_coef * (
                (self.num_items_this_episode - self.count) / (
                    self.num_items_this_episode
                )
            )
        )

        if terminations:
            infos['stack_height'] = self.stack.height() / self.discretisation
            infos['stack_gap_ratio'] = 1 - compactness
            infos['stack_compactness'] = compactness
            infos['height_upper_bound'] = self.height_ub / self.discretisation
            infos['height_lower_bound'] = self.height_lb / self.discretisation
            infos['stacked_items'] = self.stack.get_stacked_items()
            infos['n_to_stack'] = self.num_items_this_episode
            infos['n_stacked'] = self.count
            infos['completed_act_ratio'] = self.count / self.num_items_this_episode
            infos['is_success'] = stable

        infos['stack_stability_bool'] = stable

        return observations, rewards, terminations, truncations, infos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:  # will only be True for first call to reset
            self.item_getter_function.set_seed(seed)

        self.count = 0

        self.items = None
        self.stack = None

        self.stack = Stack(
            self.grid_shape,
            self.max_height
        )

        self.schedule_items()

        self.height_ub = get_height_upper_bounds(self.items, self.grid_shape)
        self.height_lb = get_height_lower_bounds(self.items, self.grid_shape)
        self.compact_ub = get_compact_upper_bounds(self.items, self.grid_shape)
        self.compact_lb = get_compact_lower_bounds(self.items, self.grid_shape)

        if self.sim:
            if self.sim_env:
                self._reset_sim_env()
            else:
                self._create_sim_env()

        observation = self._gather_observations()
        info = {}

        return observation, info

    def _gather_observations(self):

        stack_observation = self.stack.get_height_map()
        upcoming_item_observation = self.items[:self.lookahead]

        return {
                'pallet': stack_observation,
                'upcoming_items': upcoming_item_observation
                }

    def _add_item_to_sim(self, item):
        item.id = self.count
        self.sim_env.create_box(
            body_name=f'item{self.count}',
            half_extents=item.get_half_extents()/self.discretisation,
            mass=item.mass,
            position=item.get_position_mid()/self.discretisation,
            rgba_color=np.array(SET1_CMAP(self.np_random.uniform(0, 1))),
            lateral_friction=self.item_lateral_friction,
            spinning_friction=self.item_spin_friction,
            restitution=self.item_restitution,
            collision_margin=self.item_collision_margin,
        )
        self.sim_env.step()
        if self.move_pallet:
            self.sim_env.move_pallet()

    def _check_stability(self, previous_item):
        if previous_item:
            item_mid_pos = previous_item.get_position_mid()/self.discretisation
            item_mid_pos = np.asarray(item_mid_pos)
            sim_mid_pos = self.sim_env.get_item_pos(f'item{previous_item.id}')

            if (np.abs(sim_mid_pos - item_mid_pos) > 0.1).any():
                # print('FAILURE')
                return 0
            else:
                # print('SUCCESS')
                pass
            return 1
        return 1

    def _heuristic_check_stability(self, item):
        support = self.stack.get_footprint_support_ratio(item)
        return support >= self.minimal_support

    def action_masks(self):
        # Called when e.g., MaskablePPO is used
        item_shape = self.items[0][:3]
        masked_actions = self.stack.valid_placements(
            item_shape,
            flat_action_space=self.flat_action_space,
            min_support=self.mask_non_supported_items,
        )
        # # Symmetry breaking
        # half_grid = self.grid_shape/2
        # half_y, half_x = int(half_grid[0]), int(half_grid[1])
        # (performs a little better early, a little worse later)
        # if self.count == 0:
        #     masked_actions[half_y:] = False
        #     masked_actions[half_x:] = False
        self.mask = masked_actions
        return masked_actions

    def render(self):
        target_position = (*self.size/2, 0.5)
        # target_position = (0, 0, 0)
        if self.sim_env:
            return self.sim_env.render(
                target_position=(target_position)
            )
        else:
            return None

    def close(self):
        if self.sim:
            self.sim_env.close()

    def start_recording(self, file_name):
        self.sim_env.start_recording(file_name)

    def stop_recording(self):
        self.sim_env.stop_recording()

    def set_item_loader(self, item_getter_function, i_g_f_kwargs):
        self.item_getter_function = data_loaders.ItemLoader(
            item_getter_function,
            i_g_f_kwargs,
            reset_at_step=self.num_hetero_instance_samples
        )

    def set_sim(
        self,
        sim=True,
        gravity=1,
        render_mode='rgb_array',
        renderer='Tiny',
        item_spin_friction=1,
        item_lateral_friction=1,
        item_restitution=1,
        item_collision_margin=0,
        move_pallet=0,
    ):
        self.sim = sim
        self.gravity = gravity
        self.render_mode = render_mode
        self.renderer = renderer
        self.item_spin_friction = item_spin_friction
        self.item_lateral_friction = item_lateral_friction
        self.item_restitution = item_restitution
        self.item_collision_margin = item_collision_margin
        self.move_pallet = move_pallet
        self._create_sim_env()

    def set_reward_coefs(
        self,
        step_compact_rew_coef=None,
        term_compact_rew_coef=None,
        term_compact_rew_exp=None,
        stable_rew_coef=None,
        verbose=False,
    ):
        if step_compact_rew_coef is not None:
            self.step_compact_rew_coef = step_compact_rew_coef
        if term_compact_rew_coef is not None:
            self.term_compact_rew_coef = term_compact_rew_coef
        if term_compact_rew_exp is not None:
            self.term_compact_rew_exp = term_compact_rew_exp
        if stable_rew_coef is not None:
            self.stable_rew_coef = stable_rew_coef

        if verbose:
            print('self.step_compact_rew_coef=', step_compact_rew_coef)
            print('self.term_compact_rew_coef=', term_compact_rew_coef)
            print('self.term_compact_rew_exp=', term_compact_rew_exp)
            print('self.stable_rew_coef=', stable_rew_coef)

    def set_minimal_support(self, minimal_support):
        self.minimal_support = minimal_support

    def set_render_mode(self, render_mode: str):
        self.render_mode = render_mode

    def set_renderer(self, renderer: str):
        self.renderer = renderer

    def get_np_random(self):
        return self.np_random

    def get_reward_coefs(self):
        ret = {
            'step_compact_rew_coef': self.step_compact_rew_coef,
            'term_compact_rew_coef': self.term_compact_rew_coef,
            'term_compact_rew_exp': self.term_compact_rew_exp,
            'stable_rew_coef': self.stable_rew_coef,
        }
        return ret
