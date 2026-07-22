from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, List, Dict, Any


def wrap_angle(angle: np.ndarray | float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def to_relative_frame(poses: np.ndarray, reference_pose: np.ndarray | None = None) -> np.ndarray:
    assert (len(poses.shape) == 2 and (poses.shape[1] == 3 or poses.shape[1] == 2)) \
        or (poses.shape == (3,) or poses.shape == (2,)), \
        f"poses must have shape (N, 3) or (N, 2) or (3,) or (2,), got {poses.shape}"
    if len(poses.shape) == 1:
        poses = poses[np.newaxis, :]
        single_pose = True
    else:
        single_pose = False
    if poses.shape[1] == 2:
        poses = np.concatenate([poses, np.zeros((poses.shape[0], 1))], axis=1)
        virtual_orientation = True
    else:
        virtual_orientation = False

    if reference_pose is not None:
        assert reference_pose.shape == (3,), f"Reference pose must have shape (3,), got {reference_pose.shape}"
    else:
        reference_pose = poses[0]

    new_poses = np.zeros_like(poses)
    reference_yaw = wrap_angle(reference_pose[2])
    new_poses[:, 2] = wrap_angle(poses[:, 2] - reference_yaw)
    new_poses[:, :2] = poses[:, :2] - reference_pose[:2]
    rotation_matrix = np.array([[np.cos(-reference_yaw), -np.sin(-reference_yaw)],
                                [np.sin(-reference_yaw), np.cos(-reference_yaw)]])
    # for i in range(new_poses.shape[0]):
    #     new_poses[i, :2] = rotation_matrix @ new_poses[i, :2]
    new_poses[:, :2] = np.einsum("ji,ni ->nj", rotation_matrix, new_poses[:, :2])

    if virtual_orientation:
        new_poses = new_poses[:, :2]

    if single_pose:
        new_poses = new_poses[0]

    return new_poses


def to_global_frame(poses: np.ndarray, reference_pose: np.ndarray | None = None) -> np.ndarray:
    """
    Inverse of to_relative_frame: transforms a pose or set of poses from the reference-relative frame to the global frame.
    `poses` (N,3), (N,2), (3,), (2,). `reference_pose` shape (3,).
    """
    assert (len(poses.shape) == 2 and (poses.shape[1] == 3 or poses.shape[1] == 2)) \
        or (poses.shape == (3,) or poses.shape == (2,)), \
        f"poses must have shape (N, 3) or (N, 2) or (3,) or (2,), got {poses.shape}"
    if len(poses.shape) == 1:
        poses = poses[np.newaxis, :]
        single_pose = True
    else:
        single_pose = False
    if poses.shape[1] == 2:
        poses = np.concatenate([poses, np.zeros((poses.shape[0], 1))], axis=1)
        virtual_orientation = True
    else:
        virtual_orientation = False

    if reference_pose is not None:
        assert reference_pose.shape == (3,), f"Reference pose must have shape (3,), got {reference_pose.shape}"
    else:
        reference_pose = np.zeros(3)

    reference_yaw = wrap_angle(reference_pose[2])
    # Rotate back to the global frame (apply rotation with positive reference_yaw)
    rotation_matrix = np.array([
        [np.cos(reference_yaw), -np.sin(reference_yaw)],
        [np.sin(reference_yaw),  np.cos(reference_yaw)]
    ])
    global_poses = np.zeros_like(poses)
    # Position: rotate then add reference position
    global_poses[:, :2] = np.einsum("ji,ni->nj", rotation_matrix, poses[:, :2]) + reference_pose[:2]
    # Angle: add reference yaw then wrap
    global_poses[:, 2] = wrap_angle(poses[:, 2] + reference_yaw)

    if virtual_orientation:
        global_poses = global_poses[:, :2]

    if single_pose:
        global_poses = global_poses[0]

    return global_poses


class Trajectory:
    
    def __init__(self, 
                 dt: float,
                 poses: Optional[Union[np.ndarray, List[Tuple[float, float, float]]]] = None):
        self._dt = dt
        if poses is None:
            self._poses = []
        else:
            if isinstance(poses, np.ndarray):
                self._poses = [(float(e[0]), float(e[1]), float(e[2])) for e in poses]
            else:
                self._poses = poses.copy()
    
    @staticmethod
    def load(data: Dict[str, Any]) -> Trajectory:
        return Trajectory(data["dt"], data["poses"])

    def __len__(self) -> int:
        return len(self._poses)
    
    def __getitem__(self, idx) -> Trajectory:
        if isinstance(idx, slice):
            return Trajectory(self._dt, self._poses[idx])
        else:
            return Trajectory(self._dt, [self._poses[idx]])
    
    @property
    def dt(self) -> float:
        return self._dt
    
    @property
    def poses(self) -> np.ndarray:
        return np.array(self._poses)
    
    @property
    def has_missing_poses(self) -> bool:
        return np.any(np.isnan(self.poses))
    
    @property
    def only_missing_poses(self) -> bool:
        return np.all(np.isnan(self.poses)) and len(self._poses) > 0
    
    def pose_at(self, idx: int) -> np.ndarray:
        return np.array(self._poses[idx])
    
    def add_pose(self, pose: Optional[Union[np.ndarray, Tuple[float, float, float]]]):
        if pose is not None:
            if isinstance(pose, np.ndarray):
                self._poses.append((float(pose[0]), float(pose[1]), float(pose[2])))
            else:
                self._poses.append(pose)
        else:
            self._poses.append((np.nan, np.nan, np.nan))

    def to_relative_frame(self, reference_pose: Optional[np.ndarray] = None) -> Trajectory:
        new_poses = to_relative_frame(self.poses, reference_pose)
        return Trajectory(self._dt, new_poses)

    def dump(self) -> Dict[str, Any]:
        return {
            "dt": self._dt,
            "poses": self._poses.copy()
        }


MeanAndCovType = Tuple[np.ndarray, Optional[np.ndarray]]


class AbstractPedestriansPredictor(ABC):

    def __init__(self,
                 dt: float,
                 horizon_length: int,
                 history_length: int):
        self._dt = dt
        self._horizon_length = horizon_length
        self._history_length = history_length

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def horizon_length(self) -> int:
        return self._horizon_length

    @property
    def history_length(self) -> int:
        return self._history_length

    @abstractmethod
    def predict(self, history: np.ndarray) -> MeanAndCovType:
        raise NotImplementedError()


class AbstractPedestriansTracker(ABC):

    def __init__(self, dt: float):
        self._dt = dt

    @property
    def dt(self) -> float:
        return self._dt

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def update(self, observation: Dict[int, np.ndarray]):
        raise NotImplementedError()

    @abstractmethod
    def get_predictions(self, current_poses: bool) -> Dict[int, MeanAndCovType]:
        raise NotImplementedError()

    @abstractmethod
    def get_current_poses(self) -> Dict[int, np.ndarray]:
        raise NotImplementedError()


class AbstractPedestrianTrackerFactory(ABC):

    @abstractmethod
    def __call__(self, scene, *args, **kwargs) -> AbstractPedestriansTracker:
        raise NotImplementedError()


class LinearHeuristicPredictor(AbstractPedestriansPredictor):

    def __init__(self,
                 horizon: int,
                 history: int,
                 dt: float,
                 initial_cov: Optional[np.ndarray] = None):
        super().__init__(dt=dt,
                         horizon_length=horizon,
                         history_length=history)
        assert isinstance(horizon, int) and horizon >= 1, f"Horizon must be positive int >= 1, got {horizon}"
        assert isinstance(history, int) and history >= 2, f"History must be positive int >= 2, got {history}"
        assert dt > 0., f"dt must be > 0, got {dt}"
        if initial_cov is not None:
            assert len(initial_cov.shape) == 2 and initial_cov[0, 1] == initial_cov[1, 0], \
                "Initial covariance must be proper single covariance"
        self._dt = dt
        self._initial_cov = initial_cov if initial_cov is not None else np.diag([0.1, 0.1])

    def predict(self, history: np.ndarray) -> MeanAndCovType:
        """Predicts positions.

        :param history: History of positions (from old at 0 to current at -1), shape (n_pedestrians, history_length, 2)
        :return: Prediction means and covariances
        """
        assert (len(history.shape) == 3
                and history.shape[1] == self.history_length
                and history.shape[2] == 2), \
            f"History must have shape (history_length, n_pedestrians, 2 (x,y state dim)), got {history.shape}"
        velocities = np.mean(np.diff(history, axis=1) / self._dt, axis=1)
        initial_positions = history[:, -1, :]
        time_deltas = np.arange(1, self.horizon_length + 1) * self._dt
        predicted_poses = np.stack([initial_positions[i] + velocities[i] * time_deltas[:, np.newaxis]
                                    for i in range(velocities.shape[0])], axis=0)

        sigma_x = np.sqrt(self._initial_cov[0, 0])
        sigma_y = np.sqrt(self._initial_cov[1, 1])
        rho = self._initial_cov[0, 1] / (sigma_x * sigma_y)
        scale = 1.1
        covs = [np.array([[((scale ** i) * sigma_x) ** 2, rho * ((scale ** i) * sigma_x) * ((scale ** i) * sigma_y)],
                          [rho * ((scale ** i) * sigma_x) * ((scale ** i) * sigma_y), ((scale ** i) * sigma_y) ** 2]])
                for i in range(1, self.horizon_length + 1)]
        covs = np.array(covs)
        covs = np.stack([covs for _ in range(history.shape[0])])

        return predicted_poses, covs


class _ShiftBuffer:

    def __init__(self, size: int, dims: Tuple[int, ...]):
        self._size = size
        self._dims = dims
        self._buffer = np.zeros((size,) + dims)

    def put(self, value: np.ndarray):
        self._buffer[:-1, :] = self._buffer[1:, :]
        self._buffer[-1, :] = value

    def fill(self, value: np.ndarray):
        assert value.shape == self._dims
        self._buffer = np.tile(value, (self._size, 1))

    @property
    def all(self) -> np.ndarray:
        return self._buffer

    @all.setter
    def all(self, value: np.ndarray):
        assert value.shape == (self._size,) + self._dims
        self._buffer = value

    @property
    def first(self) -> np.ndarray:
        return self._buffer[0]

    @property
    def last(self) -> np.ndarray:
        return self._buffer[-1]


class _PedestrianTrack:

    _POSE_DIM = 2  # x, y

    def __init__(self, history_length: int, prediction_length: int,
                 initial_state: np.ndarray):
        self._states = _ShiftBuffer(history_length, (_PedestrianTrack._POSE_DIM,))
        self._predicted_poses = np.zeros((prediction_length, _PedestrianTrack._POSE_DIM))
        self._predicted_covs = np.zeros((prediction_length, _PedestrianTrack._POSE_DIM, _PedestrianTrack._POSE_DIM))

        if len(initial_state.shape) == 1:
            self._states.fill(initial_state)
        else:
            self._states.all = initial_state

    def update_states(self, state: np.ndarray):
        self._states.put(state)

    def update_predictions(self, pred_poses: np.ndarray, pred_covs: np.ndarray):
        self._predicted_poses = pred_poses
        self._predicted_covs = pred_covs

    @property
    def history(self) -> np.ndarray:
        return self._states.all

    @property
    def current_pose(self) -> np.ndarray:
        return self._states.last

    @property
    def predicted_poses(self) -> np.ndarray:
        return self._predicted_poses

    @property
    def predicted_covs(self) -> np.ndarray:
        return self._predicted_covs


class _GhostTrack:

    def __init__(self,
                 states: np.ndarray,
                 predicted_poses: np.ndarray,
                 predicted_covs: np.ndarray):
        self._states = _ShiftBuffer(states.shape[0], states.shape[1:])
        self._states.all = states
        self._predicted_poses = _ShiftBuffer(predicted_poses.shape[0], predicted_poses.shape[1:])
        self._predicted_poses.all = predicted_poses
        self._predicted_covs = _ShiftBuffer(predicted_covs.shape[0], predicted_covs.shape[1:])
        self._predicted_covs.all = predicted_covs
        self._tracking_time = 0

    def update(self):
        self._states.put(self._predicted_poses.first)
        self._predicted_poses.put(self._predicted_poses.last)
        self._predicted_covs.put(self._predicted_covs.last)
        self._tracking_time += 1

    @property
    def tracking_time(self) -> int:
        return self._tracking_time

    @property
    def history(self) -> np.ndarray:
        return self._states.all

    @property
    def current_pose(self) -> np.ndarray:
        return self._states.last

    @property
    def predicted_poses(self) -> np.ndarray:
        return self._predicted_poses.all

    @property
    def predicted_covs(self) -> np.ndarray:
        return self._predicted_covs.all


class GTPedestrianTracker(AbstractPedestriansTracker):

    def __init__(self,
                 predictor: AbstractPedestriansPredictor,
                 max_ghost_tracking_time: int):
        super().__init__(dt=predictor.dt)
        self._predictor = predictor
        self._track_history_length = predictor.history_length
        self._max_ghost_tracking_time = max_ghost_tracking_time

        self._tracks: Dict[int, _PedestrianTrack] = {}
        self._ghosts: Dict[int, _GhostTrack] = {}

    def reset(self):
        self._tracks = {}
        self._ghosts = {}

    def update(self, observation: Dict[int, np.ndarray]):
        tracked_peds = set(self._tracks.keys())
        observed_peds = set(observation.keys())

        # Pedestrians that were tracked and present observation - just continue tracking them
        updated_peds = tracked_peds.intersection(observed_peds)
        for k in updated_peds:
            self._tracks[k].update_states(observation[k])

        # Pedestrians that are observed, but not present in current tracks
        # They are "returned ghosts" or new pedestrians
        new_peds = observed_peds.difference(tracked_peds)
        for k in new_peds:
            if k in self._ghosts:
                # Pedestrian is a ghost that was "pseudo-tracked"
                pedestrian = _PedestrianTrack(self._track_history_length,
                                              self._predictor.horizon_length,
                                              self._ghosts[k].history)
                pedestrian.update_states(observation[k])
                self._tracks[k] = pedestrian
                self._ghosts.pop(k)
            else:
                # Pedestrian was simply not observed
                self._tracks[k] = _PedestrianTrack(self._track_history_length,
                                                   self._predictor.horizon_length,
                                                   observation[k])

        # Pedestrians that were tracked but not observed now - they become "ghosts"
        lost_peds = tracked_peds.difference(observed_peds)
        if self._max_ghost_tracking_time != 0:
            for k in lost_peds:
                pedestrian = self._tracks[k]
                ghost = _GhostTrack(pedestrian.history,
                                    pedestrian.predicted_poses,
                                    pedestrian.predicted_covs)
                self._ghosts[k] = ghost
                self._tracks.pop(k)

        # Finally, do the predictions
        self._do_predictions_for_tracks()
        self._update_ghosts()

    def get_predictions(self, current_poses: bool = True) -> Dict[int, MeanAndCovType]:
        # TODO: Current poses
        preds = {}
        preds.update({k: (v.predicted_poses.copy(), v.predicted_covs.copy()) for k, v in self._tracks.items()})
        preds.update({k: (v.predicted_poses.copy(), v.predicted_covs.copy()) for k, v in self._ghosts.items()})
        return preds

    def get_current_poses(self, return_velocities: bool = False) -> Dict[int, np.ndarray]:
        limit = 2 if not return_velocities else 4
        result = {}
        result.update({k: v.current_pose[:limit] for k, v in self._tracks.items()})
        result.update({k: v.current_pose[:limit] for k, v in self._ghosts.items()})
        return result

    @property
    def joint_track(self) -> Optional[np.ndarray]:
        # Return: (history_length, n_pedestrians, state_dim)
        if len(self._tracks) != 0:
            return np.concatenate([v.history[:, np.newaxis, :] for v in self._tracks.values()], axis=1)
        return None

    @property
    def track_dict(self) -> Dict[int, np.ndarray]:
        return {k: v.history.copy() for k, v in self._tracks.items()}

    def _do_predictions_for_tracks(self):
        if len(self._tracks) == 0:
            return {}

        joint_history = np.concatenate([v.history[np.newaxis, :, :] for v in self._tracks.values()], axis=0)
        pred_poses, pred_covs = self._predictor.predict(joint_history)
        ped_ids = list(self._tracks.keys())
        for i in range(len(self._tracks)):
            ped_id = ped_ids[i]
            self._tracks[ped_id].update_predictions(pred_poses[i, :, :], pred_covs[i, :, :])

    def _update_ghosts(self):
        for k in set(self._ghosts.keys()):
            ghost = self._ghosts[k]
            if ghost.tracking_time > self._max_ghost_tracking_time:
                self._ghosts.pop(k)
            else:
                ghost.update()


class LinearGTTrackerFactory(AbstractPedestrianTrackerFactory):

    def __init__(self,
                 horizon: int,
                 history: int,
                 dt: float,
                 max_ghost_tracking_time: int,
                 initial_cov: Optional[np.ndarray] = None):
        self._horizon = horizon
        self._history = history
        self._dt = dt
        self._max_ghost_tracking_time = max_ghost_tracking_time
        self._initial_cov = initial_cov

    def __call__(self, scene, *args, **kwargs) -> AbstractPedestriansTracker:
        predictor = LinearHeuristicPredictor(horizon=self._horizon,
                                             history=self._history,
                                             dt=self._dt,
                                             initial_cov=self._initial_cov)
        tracker = GTPedestrianTracker(predictor=predictor,
                                      max_ghost_tracking_time=self._max_ghost_tracking_time)
        return tracker
