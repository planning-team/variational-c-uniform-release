import math

import torch
from tqdm import trange

from mppi_plus_proto.trainers.mutual_inf import RolloutCollector


class GDSampler:

    def __init__(self,
                 rollout_collector: RolloutCollector,
                 num_steps: int,
                 lr: float,
                 bandwidth: float | None = None,
                 silent: bool = False):
        self._rollout_collector = rollout_collector
        self._num_steps = num_steps
        self._lr = lr
        self._bandwidth = bandwidth
        self._device = self._rollout_collector.device
        self._silent = silent

    @property
    def device(self) -> str:
        return self._device

    def sample(self, n_samples: int, x0: torch.Tensor | None = None, samples: torch.Tensor | None = None) -> torch.Tensor:
        u_lb = self._rollout_collector.action_lb.to(self._device)
        u_ub = self._rollout_collector.action_ub.to(self._device)

        if samples is not None:
            u_samples = samples.clone().to(self._device)
        else:
            u_samples = u_lb + (u_ub - u_lb) * torch.rand(n_samples, self._rollout_collector.horizon, self._rollout_collector.action_dim, device=self._device)
        u_samples.requires_grad_(True)
        
        # We use a standard optimizer now instead of Langevin noise, 
        # because the repulsion naturally handles the "exploration".
        optimizer = torch.optim.Adam([u_samples], lr=self._lr)
        
        for k in range(self._num_steps):
            optimizer.zero_grad()
            
            # Calculate Repulsion Energy
            loss = self._compute_repulsion_energy(u_samples, x0)
            loss.backward()
            optimizer.step()
            
            # Enforce kinematic boundaries
            with torch.no_grad():
                u_samples.clamp_(min=u_lb, max=u_ub)
                
            self._print(f"Step {k:3d} | Repulsion Loss: {loss.item():.2f}")

        return u_samples.clone().detach()
    
    def _compute_repulsion_energy(self, u_samples: torch.Tensor, x0: torch.Tensor):
        """
        Forces trajectories to repel each other in the (x, y) workspace.
        Minimizing this energy creates a uniform spatial distribution.
        """
        x_seqs = self._rollout_collector.rollout(u_samples, x0) # Shape: [Batch, H, d]

        total_repulsion = 0.0
        
        n = x_seqs.shape[0]

        for t in range(1, self._rollout_collector.horizon):
            xy_t = x_seqs[:, t, :]  # [Batch, d]

            diffs = xy_t.unsqueeze(1) - xy_t.unsqueeze(0)
            dists_sq = torch.sum(diffs**2, dim=-1)

            if self._bandwidth is None:
                # SVGD median heuristic: h = med^2 / log(n)
                triu_idx = torch.triu_indices(n, n, offset=1)
                pairwise_dists_sq = dists_sq[triu_idx[0], triu_idx[1]]
                med_sq = torch.median(pairwise_dists_sq)
                h = (med_sq / math.log(n)).clamp(min=1e-8)
            else:
                h = 2 * self._bandwidth ** 2

            kernel = torch.exp(-dists_sq / h)

            mask = torch.eye(n, device=self._device)
            kernel = kernel * (1 - mask)

            total_repulsion += torch.sum(kernel)
            
        return total_repulsion

    def _print(self, message: str) -> None:
        if not self._silent:
            print(message)


class GaussianPerturbedSampler:

    def __init__(self,
                 base_sampler: GDSampler,
                 selection_pct: float,
                 num_perturbations: int,
                 perturbation_std: tuple[float, ...],
                 action_lb: tuple[float, ...],
                 action_ub: tuple[float, ...]):
        self._base_sampler = base_sampler
        self._selection_pct = selection_pct
        self._num_perturbations = num_perturbations
        self._perturbation_std = perturbation_std
        self._action_lb = action_lb
        self._action_ub = action_ub

    def sample(self, n_samples: int, x0: torch.Tensor | None = None) -> torch.Tensor:
        base_samples = self._base_sampler.sample(n_samples, x0)

        n_selected = max(1, int(self._selection_pct * n_samples))
        indices = torch.randperm(n_samples)[:n_selected]
        selected = base_samples[indices]  # [n_selected, H, action_dim]

        device = base_samples.device
        std = torch.tensor(self._perturbation_std, device=device)  # [action_dim]
        noise = torch.randn(
            n_selected, self._num_perturbations, *selected.shape[1:],
            device=device,
        ) * std
        perturbed = selected.unsqueeze(1) + noise  # [n_selected, m, H, action_dim]
        perturbed = perturbed.reshape(-1, *selected.shape[1:])

        lb = torch.tensor(self._action_lb, device=device)
        ub = torch.tensor(self._action_ub, device=device)
        perturbed = perturbed.clamp(min=lb, max=ub)

        return torch.cat([base_samples, perturbed], dim=0)

    @property
    def device(self) -> str:
        return self._base_sampler.device


class MultiGenSampler:

    def __init__(self, base_sampler, silent: bool = False):
        self._base_sampler = base_sampler
        self._silent = silent

    def sample(self, n_gens: int, gen_size: int, x0: torch.Tensor | None = None) -> torch.Tensor:
        itr = trange(n_gens, desc="MultiGenSampler", disable=self._silent)
        batches = [self._base_sampler.sample(gen_size, x0) for _ in itr]
        return torch.cat(batches, dim=0)

    @property
    def device(self) -> str:
        return self._base_sampler.device


class MultiSelectGDSampler:

    def __init__(self,  
                 base_sampler: GDSampler,
                 n_base_samples: int,
                 n_children: int,
                 n_child_samples: int,
                 action_std: tuple[float, ...]):
        
        self._base_sampler = base_sampler
        self._n_base_samples = n_base_samples
        self._n_children = n_children
        self._n_child_samples = n_child_samples
        self._action_std = torch.tensor(action_std, device=base_sampler.device)

    @property
    def device(self) -> str:
        return self._base_sampler.device

    def sample(self, n_base_samples: int | None = None, x0: torch.Tensor | None = None) -> torch.Tensor:
        n_base_samples = n_base_samples if n_base_samples is not None else self._n_base_samples
        base_samples = self._base_sampler.sample(n_base_samples, x0)
        result = [base_samples]

        indices = torch.randperm(base_samples.shape[0])[:self._n_children]
        selected = base_samples[indices]  # [n_children, H, action_dim]

        u_lb = self._base_sampler._rollout_collector.action_lb.to(base_samples.device)
        u_ub = self._base_sampler._rollout_collector.action_ub.to(base_samples.device)

        for i in range(self._n_children):
            noise = torch.randn(
                self._n_child_samples, *selected.shape[1:],
                device=base_samples.device,
            ) * self._action_std
            perturbed = selected[i].unsqueeze(0) + noise  # [n_child_samples, H, action_dim]
            perturbed = perturbed.clamp(min=u_lb, max=u_ub)
            child_samples = self._base_sampler.sample(perturbed.shape[0], x0, samples=perturbed)
            result.append(child_samples)

        return torch.cat(result, dim=0)
