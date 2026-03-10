from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from spatial_lineage.stable.contracts.batch import Batch
from spatial_lineage.stable.contracts.model import BaseModel
from spatial_lineage.stable.contracts.outputs import EvalReport
from spatial_lineage.stable.contracts.predictor import BasePredictor
from spatial_lineage.stable.core.config import ConfigNode
from spatial_lineage.stable.core.logging import RunLogger
from spatial_lineage.stable.core.paths import ExperimentPaths
from spatial_lineage.stable.core.registry import EVALUATORS
from spatial_lineage.stable.evaluation.metrics.classification import accuracy_score, macro_f1_score
from spatial_lineage.stable.evaluation.metrics.ranking import macro_ovr_auroc
from spatial_lineage.stable.inference.writers import write_records_jsonl
from spatial_lineage.stable.training.checkpoint import load_checkpoint


class EvaluationEngine:
    def __init__(
        self,
        config: ConfigNode,
        model: BaseModel,
        predictor: BasePredictor,
        batches: Iterable[Batch],
        logger: RunLogger,
        paths: ExperimentPaths,
    ) -> None:
        self.config = config
        self.model = model
        self.predictor = predictor
        self.batches = list(batches)
        self.logger = logger
        self.paths = paths

    def run(self) -> EvalReport:
        self.paths.ensure()
        checkpoint = load_checkpoint(self.config.evaluation.checkpoint_path)
        self.model.load_state_dict(checkpoint.get("model_state", {}))

        bundle = self.predictor.predict(self.model, self.batches)
        targets = _collect_targets(self.batches)
        metrics = self._compute_metrics(targets, bundle)

        metrics_path = self.paths.eval_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)

        predictions_path = write_records_jsonl(self.paths.eval_dir / "predictions.jsonl", bundle.to_records())
        confusion_path = self._write_confusion_matrix(targets, bundle.predicted_labels)
        self.logger.info(f"Saved evaluation artifacts to {self.paths.eval_dir}")

        return EvalReport(
            metrics=metrics,
            output_paths={
                "metrics": str(metrics_path),
                "predictions": str(predictions_path),
                "confusion_matrix": str(confusion_path),
            },
        )

    def _compute_metrics(self, targets: list[int], bundle) -> dict[str, float]:
        metrics: dict[str, float] = {}
        requested_metrics = list(self.config.evaluation.metrics)
        for metric_name in requested_metrics:
            if metric_name == "accuracy":
                metrics[metric_name] = accuracy_score(targets, bundle.predicted_labels)
            elif metric_name == "macro_f1":
                metrics[metric_name] = macro_f1_score(targets, bundle.predicted_labels)
            elif metric_name == "auroc":
                metrics[metric_name] = macro_ovr_auroc(targets, bundle.lineage_scores)
            else:
                evaluator = EVALUATORS.build(metric_name, config=self.config)
                metrics.update(evaluator.evaluate(_flatten_batches(self.batches), bundle))
        return metrics

    def _write_confusion_matrix(self, targets: list[int], predictions: list[int]) -> Path:
        labels = sorted(set(targets + predictions))
        matrix = {label: {other: 0 for other in labels} for label in labels}
        for target, prediction in zip(targets, predictions):
            matrix[target][prediction] += 1

        output_path = self.paths.eval_dir / "confusion_matrix.csv"
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["target\\prediction", *labels])
            for label in labels:
                writer.writerow([label, *[matrix[label][other] for other in labels]])
        return output_path


def _collect_targets(batches: list[Batch]) -> list[int]:
    targets: list[int] = []
    for batch in batches:
        targets.extend(batch.targets or [])
    return targets


def _flatten_batches(batches: list[Batch]) -> Batch:
    cell_ids: list[str] = []
    expression: list[list[float]] = []
    spatial_coords: list[tuple[float, float]] = []
    clone_ids: list[str] = []
    targets: list[int] = []
    metadata = []
    for batch in batches:
        cell_ids.extend(batch.cell_ids)
        expression.extend(batch.expression)
        spatial_coords.extend(batch.spatial_coords)
        clone_ids.extend(batch.clone_ids)
        targets.extend(batch.targets or [])
        metadata.extend(batch.metadata)
    return Batch(
        cell_ids=cell_ids,
        expression=expression,
        spatial_coords=spatial_coords,
        clone_ids=clone_ids,
        targets=targets,
        metadata=metadata,
    )
