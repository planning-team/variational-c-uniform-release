from __future__ import annotations

import pickle
import numpy as np

from typing import Tuple, Any
from pathlib import Path
from enum import Enum, StrEnum
from functools import partial
from tqdm import tqdm
from dataclasses import dataclass
from pyminisim import robot
from pyminisim.core import Simulation
from pyminisim.world_map import EmptyWorld
from pyminisim.robot import BicycleRobotModel, UnicycleRobotModel
from pyminisim.sensors import PedestrianDetectorConfig, LidarSensor, LidarSensorConfig, PedestrianDetector, SemanticDetector, SemanticDetectorConfig
from pyminisim.visual import Renderer, CircleDrawing
from pyminisim.pedestrians import (ORCAParams, 
                                   ORCAPedestriansModel,
                                   HeadedSocialForceModelPolicy,
                                   HSFMParams,
                                   ReplayPedestriansPolicy,
                                   RandomWaypointTracker,
                                   FixedWaypointTracker)
from mppi_plus_proto.eval.traj_helpers import (
    LinearGTTrackerFactory, to_relative_frame)


@dataclass
class SocialCase:
    pedestrians_poses: np.ndarray
    pedestrians_goals: np.ndarray
    robot_pose: np.ndarray
    robot_goal: np.ndarray

    @staticmethod
    def load(data: dict[str, Any]) -> SocialCase:
        return SocialCase(
            pedestrians_poses=np.array(data["pedestrians_poses"]),
            pedestrians_goals=np.array(data["pedestrians_goals"]),
            robot_pose=np.array(data["robot_pose"]),
            robot_goal=np.array(data["robot_goal"])
        )

    def dump(self) -> dict[str, Any]:
        return {
            "pedestrians_poses": self.pedestrians_poses.tolist(),
            "pedestrians_goals": self.pedestrians_goals.tolist(),
            "robot_pose": self.robot_pose.tolist(),
            "robot_goal": self.robot_goal.tolist()
        }


@dataclass
class ExteriorConfig:
        
    # Speed limits for pedestrians (bounds for uniform sampling)
    max_speed_limits: tuple[float, float] = (1., 2.)
    
    # Robot perception
    detector_range: float = 4.
    detector_fov: float = 2 * np.pi
    
    ped_model: str = "orca"


@dataclass
class KinematicsConfig:
    kinematics: str  # "unicycle" or "bicycle"
    u_lb: tuple[float, ...]
    u_ub: tuple[float, ...]
    spec_args: dict[str, Any]
    robot_radius: float

    @staticmethod
    def unicycle(linear_vel_lb: float = 0,
                 linear_vel_ub: float = 1.,
                 angular_vel_abs: float = float(np.deg2rad(45.)),
                 robot_radius: float = 0.35) -> KinematicsConfig:
        return KinematicsConfig(
            kinematics="unicycle",
            u_lb=(linear_vel_lb, -angular_vel_abs),
            u_ub=(linear_vel_ub, angular_vel_abs),
            robot_radius=robot_radius,
            spec_args={}
        )

    @staticmethod
    def bicycle(linear_vel_lb: float = 0.,
                linear_vel_ub: float = 1.,
                angle_abs: float = float(np.deg2rad(30.)),
                l: float = 0.324,
                robot_radius: float = 0.35) -> KinematicsConfig:
        return KinematicsConfig(
            kinematics="bicycle",
            u_lb=(linear_vel_lb, -angle_abs),
            u_ub=(linear_vel_ub, angle_abs),
            spec_args={"l": l},
            robot_radius=robot_radius
        )


@dataclass
class SuccessConfig:
    goal_threshold: float = 0.3
    max_steps: int = 250


@dataclass
class PedPredictionConfig:
    horizon: int
    max_peds: int = 7
    filter_length: int = 4
    history: int = 5
    max_ghost_tracking_time: int = 5


@dataclass
class TimeConfig:
    sim_dt: float = 0.01
    policy_dt: float = 0.2


@dataclass
class SocialNavObs:
    predictions: np.ndarray
    goal: np.ndarray


class SocialNavOutcome(StrEnum):
    SUCCESS = "SUCCESS"
    COLLISION = "COLLISION"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"
    STEP = "STEP"


class SocialNavEnv:

    def __init__(self,
                 cases: list[SocialCase] | str | Path,
                 kinematics: KinematicsConfig,
                 prediction_config: PedPredictionConfig,
                 visualize: bool = False,
                 exterior_config: ExteriorConfig | None = None,
                 success_config: SuccessConfig | None = None,
                 time_config: TimeConfig | None = None):
        if exterior_config is None:
            exterior_config = ExteriorConfig()
        if success_config is None:
            success_config = SuccessConfig()
        if time_config is None:
            time_config = TimeConfig()

        self._exterior_config = exterior_config
        self._success_config = success_config
        self._time_config = time_config
        self._kinematics_config = kinematics
        self._prediction_config = prediction_config

        self._visualize = visualize

        if isinstance(cases, str | Path):
            self._cases = self._load_benchmarks(Path(cases))
        else:
            self._cases = cases
        self._case_idx = -1

        self._sim: Simulation = None
        self._renderer: Renderer = None

        tracker_factory = LinearGTTrackerFactory(
            horizon=prediction_config.horizon,
            history=prediction_config.history,
            dt=time_config.policy_dt,
            max_ghost_tracking_time=prediction_config.max_ghost_tracking_time
        )

        self._tracker = tracker_factory(None)

        self._current_goal: np.ndarray = None

        self._step_count = 0

    @property
    def num_cases(self) -> int:
        return len(self._cases)

    @property
    def case_idx(self) -> int:
        return self._case_idx

    @property
    def case_scenario(self) -> str:
        return self._cases[self._case_idx][0]

    @property
    def renderer(self) -> Renderer | None:
        return self._renderer

    def reset(self) -> SocialNavObs:
        # Increment case index and check if we have more cases to evaluate
        case_idx = self._case_idx + 1
        if case_idx >= len(self._cases):
            raise RuntimeError("No more cases to evaluate")
        self._case_idx = case_idx

        # Init simulation and renderer
        self._sim, self._renderer = self._create_sim()

        # Reset tracker and get initial predictions
        # Predictions in relative frame
        self._tracker.reset()
        predictions = self._update_tracker()
        if predictions is None:
            predictions = np.ones((self._prediction_config.max_peds, 
                                   self._prediction_config.horizon, 2)) * np.inf

        # Get goal in relative frame
        goal = self._get_goal()

        # Reset step count
        self._step_count = 0

        # First step to initialize the simulation
        self._sim.step()

        return SocialNavObs(predictions=predictions, goal=goal)

    def step(self, action: np.ndarray) -> tuple[SocialNavObs | None, SocialNavOutcome | None]:
        if self._renderer is not None:
            self._renderer.render()

        hold_time = 0.
        has_collision = False
        goal_reached = False
        
        # Sample and hold loop
        while hold_time < self._time_config.policy_dt:
            try:
                self._sim.step(action)
            except Exception as e:
                print(f"Error in simulation step: {e}")
                return None, SocialNavOutcome.ERROR

            hold_time += self._time_config.sim_dt
            if self._renderer is not None:
                self._renderer.render()

            collisions = self._sim.current_state.world.robot_to_pedestrians_collisions
            has_collision = collisions is not None and len(collisions) > 0
            if has_collision:
                break

            robot_pose = self._sim.current_state.world.robot.pose
            if np.linalg.norm(robot_pose[:2] - self._current_goal[:2]) <= self._success_config.goal_threshold:
                goal_reached = True
                break
        
        self._step_count += 1

        if has_collision:
            return None, SocialNavOutcome.COLLISION
        
        if self._step_count >= self._success_config.max_steps:
            return None, SocialNavOutcome.TIMEOUT

        if goal_reached:
            return None, SocialNavOutcome.SUCCESS

        goal = self._get_goal()
        predictions = self._update_tracker()
        if predictions is None:
            predictions = np.ones((self._prediction_config.max_peds, 
                                   self._prediction_config.horizon, 2)) * np.inf

        return SocialNavObs(predictions=predictions, goal=goal), SocialNavOutcome.STEP            

    def _create_sim(self) -> tuple[Simulation, Renderer | None]:
        # Get currrent social case - ped and robot poses and goals
        social_case = self._cases[self._case_idx][1]
        num_peds = social_case.pedestrians_poses.shape[0]

        # Create robot model
        if self._kinematics_config.kinematics == "unicycle":
            robot_model = UnicycleRobotModel(
                initial_pose=social_case.robot_pose,
                robot_radius=self._kinematics_config.robot_radius
            )
        elif self._kinematics_config.kinematics == "bicycle":
            robot_model = BicycleRobotModel(
                wheel_base=self._kinematics_config.spec_args["l"],
                initial_center_pose=social_case.robot_pose,
                robot_radius=self._kinematics_config.robot_radius
            )
        else:
            raise ValueError(f"Invalid kinematics: {self._kinematics_config.kinematics}")

        # Create sensor - pedestrian detector
        sensor = PedestrianDetector(
            config=PedestrianDetectorConfig(
                fov=self._exterior_config.detector_fov,
                max_dist=self._exterior_config.detector_range,
                return_type=PedestrianDetectorConfig.RETURN_ABSOLUTE
            )
        )

        # World size and max speeds for pedestrians
        max_speeds = np.random.uniform(
            self._exterior_config.max_speed_limits[0], 
            self._exterior_config.max_speed_limits[1],
            (num_peds,))

        # Create pedestrians model
        if num_peds > 0:
            ped_poses = social_case.pedestrians_poses
            ped_goals = social_case.pedestrians_goals
            waypoint_tracker = FixedWaypointTracker(initial_positions=ped_poses[:, :2],
                                                    waypoints=ped_goals[:, np.newaxis, :],
                                                    loop=True)
            if self._exterior_config.ped_model == "hsfm":
                pedestrians = HeadedSocialForceModelPolicy(waypoint_tracker=waypoint_tracker,
                                                           n_pedestrians=num_peds,
                                                           initial_poses=ped_poses,
                                                           pedestrian_linear_velocity_magnitude=max_speeds,
                                                           robot_visible=False)
            elif self._exterior_config.ped_model == "orca":
                pedestrians = ORCAPedestriansModel(dt=self._time_config.sim_dt,
                                                waypoint_tracker=waypoint_tracker,
                                                n_pedestrians=num_peds,
                                                params=ORCAParams(),
                                                initial_poses=ped_poses,
                                                max_speeds=max_speeds)
            else:
                raise ValueError(f"Unknown pedestrian model: {self._exterior_config.ped_model}")
        else:
            pedestrians = None

        # Get goal for the robot
        self._current_goal = social_case.robot_goal

        # Create simulation and renderer
        sim = Simulation(
            sim_dt=self._time_config.sim_dt,
            world_map=EmptyWorld(),
            robot_model=robot_model,
            pedestrians_model=pedestrians,
            sensors=[sensor],
            rt_factor=1. if self._visualize else None)
        if self._visualize:
            renderer = Renderer(simulation=sim,
                                resolution=60.0,
                                screen_size=(800, 800),
                                camera="robot")
        else:
            renderer = None

        if renderer is not None:
            renderer.draw("goal", CircleDrawing(self._current_goal, 0.1, (255, 0, 0), 0))

        return sim, renderer

    def _update_tracker(self) -> np.ndarray | None:
        sim_state = self._sim.current_state
        detections = {k: np.array(v) for k, v in
                                   sim_state.sensors["pedestrian_detector"].reading.pedestrians.items()}
        self._tracker.update(detections)
        predictions = self._tracker.get_predictions(current_poses=False)

        if len(predictions) == 0:
            return None

        predictions = np.stack([v[0] for v in predictions.values()])
        # predictions = predictions[:, 1:, :]

        renderer = self._renderer
        if renderer is not None:
            renderer.draw("predictions", CircleDrawing(predictions.reshape((-1, 2)), 0.05, (255, 0, 0), 0))

        robot_pose = sim_state.world.robot.pose
        predictions = np.array([to_relative_frame(e, robot_pose) for e in predictions])

        return predictions

    def _get_goal(self) -> np.ndarray:
        robot_pose = self._sim.current_state.world.robot.pose
        return to_relative_frame(self._current_goal, robot_pose)

    def _load_benchmarks(self,
                         benchmarks_dir: Path,
                         scenario: str | None = None) -> list[SocialCase]:
        benchmarks_dir = benchmarks_dir / "data"
        pkl_files = list(benchmarks_dir.glob("*.pkl"))
        all_cases = []
        for pkl_file in pkl_files:
            if scenario is not None:
                if scenario not in pkl_file.stem:
                    continue
            with open(pkl_file, "rb") as f:
                cases = pickle.load(f)
            parsed_scenario = pkl_file.stem.split("_")[0]
            # cases = [(parsed_scenario, i, SocialCase.load(e)) for i, e in enumerate(cases)]
            cases = [(parsed_scenario, SocialCase.load(e)) for e in cases]
            all_cases.extend(cases)
        return all_cases
