import numpy as np

from typing import List, Tuple, Union, Dict, Optional, Any
from oo_ctrl.np.core import (AbstractNumPyMPC,
                             AbstractNumPyModel,
                             AbstractNumPyCost,
                             AbstractStateTransform,
                             AbstractPresampler)
from oo_ctrl.np.cost_monitor import CostMonitor


class SamplingController(AbstractNumPyMPC):
    
    _U_INIT_ZERO = "zeros"
    _U_INIT_UNIFORM = "uniform"

    def __init__(self,
                 model: AbstractNumPyModel,
                 cost: Union[List[Union[Tuple[float, AbstractNumPyCost], AbstractNumPyCost]], 
                             Union[Tuple[float, AbstractNumPyCost], AbstractNumPyCost]],
                 presampler: Optional[AbstractPresampler],
                 state_transform: Optional[AbstractStateTransform] = None,
                 cost_monitor: bool = False,
                 return_state_seq: bool = False,
                 return_pre_samples: bool = False
                 ):
        composite_cost = []
        if isinstance(cost, AbstractNumPyCost):
            composite_cost.append((1., cost))
        else:
            for cost_component in cost:
                if isinstance(cost_component, AbstractNumPyCost):
                    composite_cost.append((1., cost_component))
                else:
                    assert (len(cost_component) == 2) \
                        and(isinstance(cost_component[0], int) \
                        or isinstance(cost_component[0], float)) \
                        and cost_component[0] > 0 \
                        and isinstance(cost_component, AbstractNumPyCost), \
                            f"If tuple, cost component must have format (weight, cost) and weight > 0"
                    composite_cost.append(cost_component)
        cost = composite_cost
            
        super(SamplingController, self).__init__()
        self._model = model
        self._cost = cost
        self._state_transform = state_transform
        self._presampler = presampler
        
        self._u_prev = None
        
        self._cost_monitor = CostMonitor() if cost_monitor else None
        self._return_state_seq = return_state_seq
        self._return_pre_samples = return_pre_samples
        
    @property
    def cost_monitor(self) -> Optional[CostMonitor]:
        return self._cost_monitor
        
    def step(self,
             current_state: np.ndarray,
             observation: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        info = {}
        # Transform state to the dynamics model space if needed
        if self._state_transform is not None:
            current_state = self._state_transform.inverse(current_state)

        # Inint nominal trajectory
        u_nominal, x_seq_pre_samples, x_seq_pre_samples_min = self._init_nominal(current_state, observation)
        if self._return_pre_samples and x_seq_pre_samples is not None:
            info["x_seq_pre_samples"] = x_seq_pre_samples
            info["x_seq_pre_samples_min"] = x_seq_pre_samples_min
        
        if self._u_prev is None:
            self._u_prev = np.zeros_like(u_nominal)
        self._u_prev[:-1] = u_nominal[1:, :].copy()
        self._u_prev[-1] = u_nominal[-1].copy()
    
        info["u_seq"] = u_nominal.copy()
        info["x_seq"] = x_seq_pre_samples_min.copy()
        if self._cost_monitor:
            self._log_result_cost(x_seq_pre_samples_min, u_nominal.copy(), observation)     
        
        return u_nominal[0], info
        
    def reset(self):
        self._u_prev = None
        self._cost_monitor = CostMonitor() if self._cost_monitor is not None else None

    def _calculate_costs(self,
                         x: np.ndarray,
                         u: np.ndarray,
                         observation: Optional[Dict[str, Any]]) -> Tuple[np.ndarray,
                                                                         Dict[str, np.ndarray]]:
        result = 0.
        values_horizon = {}
        for w, cost in self._cost:
            cost_values_horizon = cost(x, u, observation)
            values_horizon[cost.name] = cost_values_horizon
            cost_sum = np.sum(cost_values_horizon, axis=1)
            result = result + w * cost_sum
        return result, values_horizon

    def _log_result_cost(self,
                         x_seq: np.ndarray,
                         u_seq: np.ndarray,
                         observation: Optional[Dict[str, Any]]):
        x_seq = x_seq[np.newaxis, ...]
        u_seq = u_seq[np.newaxis, ...]
        _, values_horizon = self._calculate_costs(x_seq, 
                                                  u_seq,
                                                  observation)
        for k, v in values_horizon.items():
            self._cost_monitor.log_cost(k, v[0, :])

    def _init_nominal(self,
                      current_state: np.ndarray,
                      observation: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, 
                                                                             np.ndarray,
                                                                             np.ndarray]:
        # Sample candidates
        u_seq = self._presampler.sample(state=current_state, 
                                        observation=observation) # (n_samples, horizon, dim)
        # Add previous solution to the candidates
        if self._u_prev is not None:
            u_seq = np.concat((u_seq, self._u_prev[np.newaxis, :, :]), axis=0)
        u_seq = self._model.clip(u_seq)

        # Do rollout
        x_prev = np.tile(current_state, (u_seq.shape[0], 1))
        x_seq = []
        for i in range(u_seq.shape[1]):
            x_prev = self._model(x_prev, u_seq[:, i, :])
            x_seq.append(x_prev)
        x_seq = np.stack(x_seq, axis=1) # (n_samples, horizon, dim)
        if self._state_transform is not None:
            x_seq = self._state_transform.forward(x_seq)

        s, _ = self._calculate_costs(x_seq, u_seq, observation) # (n_samples,)
        min_idx = np.argmin(s)
        u_nominal = u_seq[min_idx]

        return u_nominal, x_seq, x_seq[min_idx]
