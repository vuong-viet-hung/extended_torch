import operator
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Literal

import torch

from extended_torch.losses import Loss
from extended_torch.metrics import Metric


Criterion = Loss | Metric
Phase = Literal["train", "valid"]
Comparator = Callable[[float, float], bool]


class Monitor(ABC):
    @abstractmethod
    def update(self, phase: Phase, model) -> None:
        pass


class EarlyStopping(Monitor):

    def __init__(
            self, criterion: Criterion, patience: int, phase: Phase = "valid"
    ) -> None:
        self._criterion = criterion
        self._patience = patience
        self._phase = phase
        self._comparator = (
            operator.gt if isinstance(criterion, Loss) else operator.lt
        )
        self._best_result = None
        self._epoch_without_improve = 0

    def update(self, phase: Phase, model) -> None:
        if phase != self._phase:
            return
        current_result = self._criterion.result()
        if (
                self._best_result is None
                or self._comparator(self._best_result, current_result)
        ):
            self._best_result = current_result
            self._epoch_without_improve = 0
            return
        self._epoch_without_improve += 1
        if self._epoch_without_improve > self._patience:
            model.running = False


class ModelCheckpoint(Monitor):

    def __init__(
        self,
        criterion: Criterion,
        model_path: str | Path,
        optimizer_path: str | Path,
        phase: Phase = "valid",
    ) -> None:
        self._criterion = criterion
        self._phase = phase
        self._comparator = (
            operator.gt if isinstance(criterion, Loss) else operator.lt
        )
        self._best_result = None
        self._model_path = Path(model_path)
        self._optimizer_path = Path(optimizer_path)

    def update(self, phase: Phase, model) -> None:
        if phase != self._phase:
            return
        current_result = self._criterion.result()
        if (
                self._best_result is None
                or self._comparator(self._best_result, current_result)
        ):
            self._best_result = current_result
            self.save_model(model)

    def save_model(self, model):

        if not self._model_path.parent.exists():
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._optimizer_path.parent.exists():
            self._optimizer_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(model.net.state_dict(), self._model_path)
        torch.save(model.optimizer.state_dict(), self._optimizer_path)
