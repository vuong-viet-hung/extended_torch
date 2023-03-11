import torch
import torch.utils.data

from pathlib import Path
from typing import Literal

from extended_torch.losses import Loss
from extended_torch.metrics import Metric
from extended_torch.monitors import Monitor


Phase = Literal["train", "valid"]


class Model:

    def __init__(
        self,
        net: torch.nn.Module,
        loss: Loss,
        optimizer: torch.optim.Optimizer,
        metrics: list[Metric] | None = None,
        monitors: list[Monitor] | None = None,
        device: torch.device | str = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        net.to(device)
        self.net = net
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = [] if metrics is None else metrics
        self.monitors = [] if monitors is None else monitors
        self.device = device
        self.running = True
        self.tqdm, self.trange = _import_tqdm()

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        n_epochs: int,
    ):
        progress_bar = self.trange(n_epochs)

        for epoch in progress_bar:

            progress_bar.set_description(
                f"Epoch {epoch + 1}/{len(progress_bar)}: "
                f"lr = {self.optimizer.param_groups[0]['lr']:.2e}"
            )
            self.train_one_epoch(train_loader)
            self.valid_one_epoch(valid_loader)

            if not self.running:
                break

    def train_one_epoch(
        self, train_loader: torch.utils.data.DataLoader
    ) -> None:
        progress_bar = self.tqdm(train_loader)
        self.net.train()

        for input_batch, target_batch in progress_bar:
            self.train_one_step(input_batch, target_batch)
            progress_bar.set_description(f"Train: {self.format_results()}")

        self.notify_monitors("train")
        self.reset()

    def train_one_step(
        self, input_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> None:
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)

        output_batch = self.net(input_batch)
        computed_loss = self.loss(output_batch, target_batch)

        computed_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.update(output_batch, target_batch)

    def valid_one_epoch(
        self, valid_loader: torch.utils.data.DataLoader
    ) -> None:
        progress_bar = self.tqdm(valid_loader)
        self.net.eval()

        with torch.no_grad():
            for input_batch, target_batch in progress_bar:
                self.valid_one_step(input_batch, target_batch)
                progress_bar.set_description(f"Valid: {self.format_results()}")

        self.notify_monitors("valid")
        self.reset()

    def valid_one_step(
        self, input_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> None:
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        output_batch = self.net(input_batch)
        self.update(output_batch, target_batch)

    def test(self, test_loader: torch.utils.data.DataLoader) -> None:
        progress_bar = self.tqdm(test_loader)
        self.net.eval()

        with torch.no_grad():
            for input_batch, target_batch in progress_bar:
                self.valid_one_step(input_batch, target_batch)
                progress_bar.set_description(f"Test: {self.format_results()}")

        self.reset()

    def predict(
            self, input_loader: torch.utils.data.DataLoader
    ) -> torch.Tensor:
        self.net.eval()
        with torch.no_grad():
            return torch.cat(
                [
                    self.net(input_batch.to(self.device))
                    for input_batch in input_loader
                ]
            )

    def save(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.net.state_dict(), save_dir / "net.pth")
        torch.save(self.optimizer.state_dict(), save_dir / "optimizer.pth")

    def update(
        self, output_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> None:
        self.loss.update(output_batch, target_batch)

        for metric in self.metrics:
            metric.update(output_batch, target_batch)

    def reset(self) -> None:
        self.loss.reset()
        for metric in self.metrics:
            metric.reset()

    def format_results(self) -> str:
        loss_result_format = f"loss = {self.loss.result():.4f}"
        metric_results_format = "".join(
            f", {metric.name} = {metric.result():.4f}" for metric in self.metrics
        )
        results_format = loss_result_format + metric_results_format
        return results_format

    def notify_monitors(self, phase: Phase):
        for monitor in self.monitors:
            monitor.update(phase, self)


def _import_tqdm() -> tuple:
    from tqdm.auto import tqdm, trange
    return tqdm, trange
