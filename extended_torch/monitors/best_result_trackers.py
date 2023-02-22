import operator
from abc import ABCMeta, abstractmethod
from pathlib import Path

import torch

from .monitors import Monitor, Criterion, Phase, Comparator
from extended_torch.losses import Loss


class BestResultTracker(Monitor, metaclass=ABCMeta):

    def __init__(
        self,
        criterion: Criterion,
        phase: Phase = "valid",
        comparator: Comparator | None = None,
    ) -> None:
        self.criterion = criterion
        self.phase = phase
        self.comparator = (
            operator.gt if isinstance(criterion, Loss) else operator.lt
        ) if comparator is None else comparator
        self.best_result = None

    @abstractmethod
    def update(self, phase: Phase, model) -> None:
        pass


class EarlyStopping(BestResultTracker):

    def __init__(
        self, criterion: Criterion, patience: int, phase: Phase = "valid"
    ) -> None:
        super().__init__(criterion, phase)
        self._patience = patience
        self._epoch_without_improve = 0

    def update(self, phase: Phase, model) -> None:
        if phase != self.phase:
            return
        current_result = self.criterion.result()
        if (
            self.best_result is None
            or self.comparator(self.best_result, current_result)
        ):
            self.best_result = current_result
            self._epoch_without_improve = 0
            return
        self._epoch_without_improve += 1
        if self._epoch_without_improve > self._patience:
            model.running = False


class ModelCheckpoint(BestResultTracker):

    def __init__(
        self,
        criterion: Criterion,
        model_path: str | Path,
        optimizer_path: str | Path
    ) -> None:
        super().__init__(criterion)
        self._model_path = Path(model_path)
        self._optimizer_path = Path(optimizer_path)

    def update(self, phase: Phase, model) -> None:

        if phase != self.phase:
            return
        current_result = self.criterion.result()
        if (
            self.best_result is None
            or self.comparator(self.best_result, current_result)
        ):
            self.best_result = current_result
            self.save_model(model)

    def save_model(self, model):
        
        if not self._model_path.parent.exists():
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._optimizer_path.parent.exists():
            self._optimizer_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(model.net.state_dict(), self._model_path)
        torch.save(model.optimizer.state_dict(), self._optimizer_path)
