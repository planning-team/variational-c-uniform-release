import torch

from abc import ABC, abstractmethod
from pathlib import Path
from torch.utils.data import Dataset


class AbstractNormalizer(ABC):

    @abstractmethod
    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def unnormalize(self, data: torch.Tensor) -> torch.Tensor:
        pass


class DummyNormalizer(AbstractNormalizer):

    def __init__(self):
        pass

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        return data

    def unnormalize(self, data: torch.Tensor) -> torch.Tensor:
        return data


class MeanStdNormalizer(AbstractNormalizer):

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self._mean = mean
        self._std = std

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        mean = self._mean.clone().to(data.device)
        std = self._std.clone().to(data.device)
        if len(data.shape) == 3:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return (data - mean) / std

    def unnormalize(self, data: torch.Tensor) -> torch.Tensor:
        mean = self._mean.clone().to(data.device)
        std = self._std.clone().to(data.device)
        if len(data.shape) == 3:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return data * std + mean


class StdNormalizer(AbstractNormalizer):
    def __init__(self, std: torch.Tensor):
        self._std = std

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        std = self._std.clone().to(data.device)
        if len(data.shape) == 3:
            std = std.unsqueeze(0)
        return data / std

    def unnormalize(self, data: torch.Tensor) -> torch.Tensor:
        std = self._std.clone().to(data.device)
        if len(data.shape) == 3:
            std = std.unsqueeze(0)
        return data * std


class ZeroToOneNormalizer(AbstractNormalizer):
    def __init__(self, min: torch.Tensor, max: torch.Tensor):
        self._min = min
        self._max = max

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        min = self._min.clone().to(data.device)
        max = self._max.clone().to(data.device)
        if len(data.shape) == 3:
            min = min.unsqueeze(0)
            max = max.unsqueeze(0)
        return (data - min) / (max - min)

    def unnormalize(self, data: torch.Tensor) -> torch.Tensor:
        min = self._min.clone().to(data.device)
        max = self._max.clone().to(data.device)
        if len(data.shape) == 3:
            min = min.unsqueeze(0)
            max = max.unsqueeze(0)
        return data * (max - min) + min


class MinusOneToOneNormalizer(AbstractNormalizer):
    def __init__(self, min: torch.Tensor, max: torch.Tensor):
        self._min = min
        self._max = max
        self._one = torch.tensor(1., requires_grad=False)
        self._two = torch.tensor(2., requires_grad=False)

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        min = self._min.clone().to(data.device)
        max = self._max.clone().to(data.device)
        one = self._one.clone().to(data.device)
        two = self._two.clone().to(data.device)
        if len(data.shape) == 3:
            min = min.unsqueeze(0)
            max = max.unsqueeze(0)
        return two * (data - min) / (max - min) - one

    def unnormalize(self, data: torch.Tensor) -> torch.Tensor:
        min = self._min.clone().to(data.device)
        max = self._max.clone().to(data.device)
        one = self._one.clone().to(data.device)
        two = self._two.clone().to(data.device)
        if len(data.shape) == 3:
            min = min.unsqueeze(0)
            max = max.unsqueeze(0)
        return (data + one) * (max - min) / two + min


class ActionsDataset(Dataset):

    def __init__(self,
                 data_file: str | Path,
                 normalization: str | None = None):
        super(ActionsDataset, self).__init__()
        self._data = torch.load(data_file)

        self._action_dim = self._data.shape[2]
        self._horizon = self._data.shape[1]

        if normalization is not None:
            if normalization == "mean_std":
                mean = self._data.mean(dim=0)
                std = self._data.std(dim=0).clamp(min=1e-6)
                self._normalizer = MeanStdNormalizer(mean, std)
            elif normalization == "std":
                std = self._data.std(dim=0).clamp(min=1e-6)
                self._normalizer = StdNormalizer(std)
            elif normalization == "zero_to_one":
                min = self._data.min(dim=0)
                max = self._data.max(dim=0)
                self._normalizer = ZeroToOneNormalizer(min, max)
            elif normalization == "minus_one_to_one":
                min = self._data.min(dim=0)
                max = self._data.max(dim=0)
                self._normalizer = MinusOneToOneNormalizer(min, max)
            else:
                raise ValueError(f"Invalid normalization: {normalization}")
        else:
            self._normalizer = DummyNormalizer()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._normalizer.normalize(self._data[idx])

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def normalizer(self) -> AbstractNormalizer:
        return self._normalizer

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        return self._normalizer.normalize(data)

    def unnormalize(self, data: torch.Tensor) -> torch.Tensor:
        return self._normalizer.unnormalize(data)
