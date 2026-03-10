from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable

from spatial_lineage.stable.contracts.batch import Batch
from spatial_lineage.stable.contracts.loss import BaseLoss
from spatial_lineage.stable.contracts.model import BaseModel
from spatial_lineage.stable.core.config import ConfigNode
from spatial_lineage.stable.core.logging import RunLogger
from spatial_lineage.stable.core.paths import ExperimentPaths
from spatial_lineage.stable.evaluation.metrics.classification import accuracy_score
from spatial_lineage.stable.training.checkpoint import save_checkpoint


@dataclass(frozen=True)
class TrainingResult:
    best_metric: float
    checkpoint_path: str


class TrainingEngine:
    def __init__(
        self,
        config: ConfigNode,
        model: BaseModel,
        loss_fn: BaseLoss,
        train_batches: Iterable[Batch],
        valid_batches: Iterable[Batch],
        logger: RunLogger,
        paths: ExperimentPaths,
    ) -> None:
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        self.train_batches = list(train_batches)
        self.valid_batches = list(valid_batches)
        self.logger = logger
        self.paths = paths

    def run(self) -> TrainingResult:
        self.paths.ensure()
        frozen_config_path = self.config.save(self.paths.root / "frozen_config.yaml")
        self.logger.info(f"Saved frozen config to {frozen_config_path}")

        best_accuracy = -1.0
        best_checkpoint = self.paths.checkpoints / "best.json"
        for epoch in range(1, self.config.training.epochs + 1):
            train_metrics = self._run_epoch(self.train_batches)
            self.logger.log_metrics("train", epoch, train_metrics)

            if epoch % self.config.training.eval_interval == 0 and self.valid_batches:
                valid_metrics = self._run_epoch(self.valid_batches)
                self.logger.log_metrics("val", epoch, valid_metrics)
            else:
                valid_metrics = {}

            payload = {
                "epoch": epoch,
                "experiment": self.config.experiment.name,
                "model_name": self.config.model.name,
                "model_state": self.model.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": valid_metrics,
            }
            latest_path = save_checkpoint(self.paths.checkpoints / "latest.json", payload)
            self.logger.info(f"Saved checkpoint to {latest_path}")

            epoch_accuracy = valid_metrics.get("accuracy", train_metrics["accuracy"])
            if epoch_accuracy >= best_accuracy:
                best_accuracy = epoch_accuracy
                save_checkpoint(best_checkpoint, payload)

        return TrainingResult(best_metric=best_accuracy, checkpoint_path=str(best_checkpoint))

    def _run_epoch(self, batches: list[Batch]) -> dict[str, float]:
        losses: list[float] = []
        accuracies: list[float] = []
        for batch in batches:
            output = self.model.forward(batch)
            loss_output = self.loss_fn.compute(batch, output)
            losses.append(loss_output.loss)
            if batch.targets is not None:
                predictions = [row.index(max(row)) for row in output.logits]
                accuracies.append(accuracy_score(batch.targets, predictions))

        return {
            "loss": mean(losses) if losses else 0.0,
            "accuracy": mean(accuracies) if accuracies else 0.0,
        }
