from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelOutput:
    logits: list[list[float]]
    embeddings: list[list[float]]
    aux_losses: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LossOutput:
    loss: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class PredictionBundle:
    cell_ids: list[str]
    predicted_labels: list[int]
    lineage_scores: list[list[float]]
    trajectory_latent: list[list[float]]
    metadata: list[dict[str, Any]]

    def to_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for index, cell_id in enumerate(self.cell_ids):
            record = dict(self.metadata[index])
            record.update(
                {
                    "cell_id": cell_id,
                    "predicted_label": self.predicted_labels[index],
                    "lineage_scores": self.lineage_scores[index],
                    "trajectory_latent": self.trajectory_latent[index],
                }
            )
            records.append(record)
        return records


@dataclass(frozen=True)
class EvalReport:
    metrics: dict[str, float]
    output_paths: dict[str, str]
