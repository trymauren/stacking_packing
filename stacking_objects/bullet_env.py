import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import numpy as np
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
import git
# from time import wait

path_to_root = git.Repo('.', search_parent_directories=True).working_dir

# inspiration: https://github.com/qgallouedec/panda-gym/blob/master/panda-gym
# /panda_gym/pybullet.py


def get_background_color():
    background_color = np.array([0, 255, 0]).astype(np.float32) / 255
    ret = '--background_color_red={} --background_color_green={} --background_color_blue={}'.format(
            *background_color
        )
    return ret


class PyBulletEnv:
    """Convenient class to use PyBullet physics engine.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        n_substeps (int, optional): Number of sim substep when step() is called. Defaults to 20.
        background_color (np.ndarray, optional): The background color as (red, green, blue).
            Defaults to np.array([223, 54, 45]).
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
    """

    def __init__(
        self,
        render_mode: str = 'rgb_array',
        renderer: str = 'Tiny',
    ) -> None:
        self.render_mode = render_mode

        if self.render_mode == 'human':
            self.connection_mode = p.GUI
        elif self.render_mode == 'rgb_array':
            if renderer == 'OpenGL':
                self.connection_mode = p.GUI
            elif renderer == 'Tiny':
                self.connection_mode = p.DIRECT
            else:
                raise ValueError('The "renderer" argument is must be in {"Tiny", "OpenGL"}')
        else:
            raise ValueError('The "render" argument is must be in {"rgb_array", "human"}')

        self.physics_client = bc.BulletClient(
            connection_mode=self.connection_mode)  # , options=get_background_color())

        # print(self.physics_client.getPhysicsEngineParameters())

        # Not related to the numSubSteps parameter of bullet:
        self.n_substeps = 240

        self.physics_client.setPhysicsEngineParameter(
            # Physics engine timestep in fraction of seconds, each time
            # you call 'stepSimulation' simulated time will progress
            # this amount.
            fixedTimeStep=1.0 / 240.0,  # should be 240 according to manual!

            # Subdivide the physics simulation step further
            # by 'numSubSteps'. This will trade performance over
            # accuracy.
            # numSubSteps=??,  # higher increase stability

            # Contact points with distance exceeding this threshold are
            # not processed by the LCP solver. In addition, AABBs are
            # extended by this number. Defaults to 0.02 in Bullet 2.x
            contactBreakingThreshold=0.02,

            # Choose the maximum number of constraint solver iterations.
            # If the solverResidualThreshold is reached, the solver may
            # terminate before the numSolverIterations.
            numSolverIterations=300,
        )

        # print(self.physics_client.getPhysicsEngineParameters())

        self.recording_id = -1
        self.reset_sim()
        self._bodies_idx = {}

    def step(self) -> None:
        for _ in range(self.n_substeps):
            self.physics_client.stepSimulation()

    def close(self) -> None:
        if self.physics_client.isConnected():
            self.physics_client.disconnect()

    def get_item_pos(self, item_id):
        sim_id = self._bodies_idx[item_id]
        sim_mid_pos, _ = self.physics_client.getBasePositionAndOrientation(
            sim_id
        )
        return np.asarray(sim_mid_pos)

    def place_visualizer(
        self,
        target_position: np.ndarray,
        distance: float,
        yaw: float,
        pitch: float
    ) -> None:
        """Orient the camera used for rendering.

        Args:
            target (np.ndarray): Target position, as (x, y, z).
            distance (float): Distance from the target position.
            yaw (float): Yaw.
            pitch (float): Pitch.
        """
        self.physics_client.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=target_position,
        )

    def loadURDF(self, body_name: str, **kwargs: Any) -> None:
        """Load URDF file.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
        """
        self._bodies_idx[body_name] = self.physics_client.loadURDF(**kwargs)

    def create_stack_from_list(self, stack) -> None:
        """
        Create a box for each item in the stack and place them.
        Args:
            stack (iterable with Item(s)) to be created.
        """
        for ix, item in enumerate(stack):
            self.create_box_from_item(item, ix)
            self.step()

    def create_box_from_item(self, item, i):
        mid_pos = item.get_position_mid()
        self.create_box(
            body_name=f'item{i}',
            half_extents=np.array(item.get_dimensions())/2,
            mass=item.mass,
            position=np.array(mid_pos),
            rgba_color=np.array([100, 100, 100, 1]),
            lateral_friction=None,
            spinning_friction=None,
        )

    def create_box(
        self,
        body_name: str,
        half_extents: np.ndarray,
        mass: float,
        position: np.ndarray,
        rgba_color: Optional[np.ndarray] = None,
        specular_color: Optional[np.ndarray] = None,
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
        restitution: Optional[float] = None,
        collision_margin: Optional[float] = 0,
    ) -> None:
        """Create a box.
        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            half_extents (np.ndarray): Half size of the box in meters, as (x, y, z).
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
            texture (str or None, optional): Texture file name. Defaults to None.
        """

        rgba_color = rgba_color if rgba_color is not None else np.zeros(4)
        specular_color = specular_color if specular_color is not None else np.zeros(3)
        modified_half_extents = np.array(half_extents) - collision_margin
        modified_half_extents[-1] = half_extents[-1]
        visual_kwargs = {
            'halfExtents': modified_half_extents,
            'specularColor': specular_color,
            'rgbaColor': rgba_color,
        }
        collision_kwargs = {'halfExtents': modified_half_extents}
        self._create_geometry(
            body_name,
            geom_type=self.physics_client.GEOM_BOX,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

        # position, orientation = self.physics_client.getBasePositionAndOrientation(int(body_name[-1]))
        # print("Position:", position)
        # print("Orientation:", orientation)

    def _create_geometry(
        self,
        body_name: str,
        geom_type: int,
        mass: float = 0.0,
        position: Optional[np.ndarray] = None,
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
        restitution: Optional[float] = None,
        visual_kwargs: Dict[str, Any] = {},
        collision_kwargs: Dict[str, Any] = {},
    ) -> None:
        """Create a geometry.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            geom_type (int): The geometry type. See self.physics_client.GEOM_<shape>.
            mass (float, optional): The mass in kg. Defaults to 0.
            position (np.ndarray, optional): The position, as (x, y, z). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
            visual_kwargs (dict, optional): Visual kwargs. Defaults to {}.
            collision_kwargs (dict, optional): Collision kwargs. Defaults to {}.
        """
        position = position if position is not None else np.zeros(3)
        baseVisualShapeIndex = self.physics_client.createVisualShape(
            geom_type, **visual_kwargs)
        if not ghost:
            baseCollisionShapeIndex = self.physics_client.createCollisionShape(
                geom_type, **collision_kwargs)
        else:
            baseCollisionShapeIndex = -1
        self._bodies_idx[body_name] = self.physics_client.createMultiBody(
            baseVisualShapeIndex=baseVisualShapeIndex,
            baseCollisionShapeIndex=baseCollisionShapeIndex,
            baseMass=mass,
            basePosition=position,
        )

        if lateral_friction is not None:
            self.set_lateral_friction(
                body=body_name, link=-1, lateral_friction=lateral_friction
            )
        if spinning_friction is not None:
            self.set_spinning_friction(
                body=body_name, link=-1, spinning_friction=spinning_friction
            )
        if restitution is not None:
            self.set_restitution(
                body=body_name, link=-1, restitution=restitution
            )

    def set_lateral_friction(
        self,
        body: str,
        link: int,
        lateral_friction: float
    ) -> None:
        """Set the lateral friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            lateral_friction (float): Lateral friction.
        """
        self.physics_client.changeDynamics(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            lateralFriction=lateral_friction,
            activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING,
        )

    def set_spinning_friction(
        self,
        body: str,
        link: int,
        spinning_friction: float
    ) -> None:
        """Set the spinning friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            spinning_friction (float): Spinning friction.
        """
        self.physics_client.changeDynamics(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            spinningFriction=spinning_friction,
            activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING,
        )

    def set_restitution(
        self,
        body: str,
        link: int,
        restitution: float
    ) -> None:
        """Set the bounciness of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            restitution (float): Bounciness.
        """
        self.physics_client.changeDynamics(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            restitution=restitution,
            activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING,
        )

    def move_pallet(self):
        meters_to_drive = 0.5
        seconds = 2
        n_substeps = 240 * seconds
        split = meters_to_drive / n_substeps

        def pivot_pallet_move(pivot):
            self.physics_client.changeConstraint(
                self.pallet_constraint, jointChildPivot=pivot, maxForce=50000
            )
            self.physics_client.stepSimulation()

        x_pivot = 0
        y_pivot = 0

        for _ in range(n_substeps):
            pivot = [x_pivot, y_pivot, 0]
            pivot_pallet_move(pivot)
            x_pivot += split

        for _ in range(n_substeps):
            pivot = [x_pivot, y_pivot, 0]
            pivot_pallet_move(pivot)
            y_pivot += split

        for _ in range(n_substeps):
            pivot = [x_pivot, y_pivot, 0]
            pivot_pallet_move(pivot)
            x_pivot -= split

        for _ in range(n_substeps):
            pivot = [x_pivot, y_pivot, 0]
            pivot_pallet_move(pivot)
            y_pivot -= split

        for _ in range(n_substeps):
            self.physics_client.stepSimulation()

    def render(
        self,
        width: int = 3024,
        height: int = 1964,
        target_position: Optional[np.ndarray] = None,
        distance: float = 3.5,
        yaw: float = 45,
        pitch: float = -30,
        roll: float = 0,
    ) -> Optional[np.ndarray]:
        """Render.

        If render mode is "rgb_array", return an RGB array of the scene. Else, do nothing and return None.

        Args:
            width (int, optional): Image width. Defaults to 720.
            height (int, optional): Image height. Defaults to 480.
            target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
                Defaults to [0., 0., 0.].
            distance (float, optional): Distance of the camera. Defaults to 1.4.
            yaw (float, optional): Yaw of the camera. Defaults to 45.
            pitch (float, optional): Pitch of the camera. Defaults to -30.
            roll (int, optional): Roll of the camera. Defaults to 0.
            mode (str, optional): Deprecated: This argument is deprecated and will be removed in a future
                version. Use the render_mode argument of the constructor instead.

        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        if self.render_mode == 'rgb_array':
            target_position = target_position if target_position is not None else np.zeros(3)
            view_matrix = self.physics_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=target_position,
                distance=distance,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                upAxisIndex=2,
            )
            proj_matrix = self.physics_client.computeProjectionMatrixFOV(
                fov=60, aspect=float(width) / height, nearVal=0.1, farVal=100.0
            )
            (_, _, rgba, _, _) = self.physics_client.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                shadow=True,
                renderer=self.physics_client.ER_BULLET_HARDWARE_OPENGL,
            )
            # With Python3.10, pybullet return flat tuple instead of array. So we need to build create the array.
            rgba = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
            return rgba[..., :3]

    def start_recording(self, file_name):
        logging_type = self.physics_client.STATE_LOGGING_VIDEO_MP4
        self.recording_id = self.physics_client.startStateLogging(
            loggingType=logging_type,
            fileName=file_name
        )

    def stop_recording(self):
        if self.recording_id == -1:
            print('No logger to stop')
            return
        self.physics_client.stopStateLogging(
            self.recording_id
        )

    def reset_sim(self, gravity=0):
        self.physics_client.resetSimulation()

        # if self.recording_id != -1:
        #     self.physics_client.stopStateLogging(self.recording_id)
        # self.recording_id = -1
        # logging_type = self.physics_client.STATE_LOGGING_VIDEO_MP4
        # path = path_to_root + '/code/recordings/first.mov'
        # path = path.replace(' ', '\\ ')
        # self.recording_id = self.physics_client.startStateLogging(
        #     loggingType=logging_type,
        #     fileName=path
        # )

        self.physics_client.setAdditionalSearchPath(
            pybullet_data.getDataPath())
        self._bodies_idx = {}

        if gravity:
            self.physics_client.setGravity(0, 0, -9.81)
