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
        patience: int = 1,
        comparator: Comparator | None = None,
    ) -> None:
        self._criterion = criterion
        self._phase = phase
        self._patience = patience
        self._comparator = (
            operator.gt if isinstance(criterion, Loss) else operator.lt
        ) if comparator is None else comparator
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
            self.action(model)

    @abstractmethod
    def action(self, model) -> None:
        pass


class EarlyStopping(BestResultTracker):

    def __init__(
        self, criterion: Criterion, patience: int, phase: Phase = "valid"
    ) -> None:
        super().__init__(criterion, phase, patience)

    def action(self, model) -> None:
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

    def action(self, model) -> None:

        if not self._model_path.parent.exists():
            self._model_path.mkdir(parents=True, exist_ok=True)
        if not self._optimizer_path.parent.exists():
            self._optimizer_path.mkdir(parents=True, exist_ok=True)

        torch.save(model.net.state_dict(), self._model_path)
        torch.save(model.optimizer.state_dict(), self._optimizer_path)
