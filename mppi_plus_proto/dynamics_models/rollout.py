import torch
from mppi_plus_proto.dynamics_models.models import AbstractDynamicsModel


class RolloutCollector:

    def __init__(self, 
                 dynamics: AbstractDynamicsModel,
                 horizon: int,
                 action_lb: tuple[float, ...],
                 action_ub: tuple[float, ...],
                 action_mid: tuple[float, ...] | None = None,
                 state_dim_cut: int | None = None) -> None:
        assert dynamics.backend == "torch"
        self._dynamcis = dynamics
        self._horizon = horizon
        self._action_lb = torch.tensor(action_lb, 
                                       device=dynamics.device, 
                                       dtype=torch.float32)
        self._action_ub = torch.tensor(action_ub, 
                                       device=dynamics.device, 
                                       dtype=torch.float32)
        if action_mid is not None:
            self._action_mid = torch.tensor(action_mid,
                                            device=dynamics.device,
                                            dtype=torch.float32)
        else:
            self._action_mid = None
        self._unnormalize_scale = (self._action_ub - self._action_lb) / 2.
        self._unnormalize_shift = (self._action_ub + self._action_lb) / 2.
        if state_dim_cut is not None:
            self._state_dim = state_dim_cut
        else:
            self._state_dim = dynamics.state_dim

        self._device = dynamics.device

    @property
    def device(self) -> str:
        return self._device
    
    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._dynamcis.action_dim

    @property
    def action_lb(self) -> torch.Tensor:
        return self._action_lb

    @property
    def action_ub(self) -> torch.Tensor:
        return self._action_ub

    @property
    def action_mid(self) -> torch.Tensor:
        return self._action_mid

    def unnormalize(self, u_seq: torch.Tensor) -> torch.Tensor:
        u_seq = self._unnormalize_scale * u_seq + self._unnormalize_shift
        return u_seq

    def rollout(self, 
                u_seq: torch.Tensor,
                x_init: torch.Tensor | None = None,
                clip: bool = False) -> torch.Tensor:
        if clip:
            u_seq = torch.clamp(u_seq, 
                                self.action_lb.clone().to(u_seq.device), 
                                self.action_ub.clone().to(u_seq.device))
        x_seq = self._dynamcis.rollout(u_seq, x_init)
        x_seq = x_seq[:, :, :self._state_dim]
        return x_seq
