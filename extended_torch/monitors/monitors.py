from abc import ABC, abstractmethod
from typing import Callable, Literal

from extended_torch.losses import Loss
from extended_torch.metrics import Metric


Criterion = Loss | Metric
Phase = Literal["train", "valid"]
Comparator = Callable[[float, float], bool]


class Monitor(ABC):

    @abstractmethod
    def update(self, phase: Phase, model) -> None:
        pass
