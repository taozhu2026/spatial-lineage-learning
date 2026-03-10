from __future__ import annotations

from typing import Any, Iterable

from spatial_lineage.stable.contracts.batch import Batch
from spatial_lineage.stable.contracts.model import BaseModel
from spatial_lineage.stable.contracts.outputs import PredictionBundle
from spatial_lineage.stable.contracts.predictor import BasePredictor
from spatial_lineage.stable.core.registry import PREDICTORS
from spatial_lineage.stable.utils.tensor import argmax, softmax


@PREDICTORS.register("lineage_score_predictor")
class LineageScorePredictor(BasePredictor):
    def __init__(self, config: Any) -> None:
        super().__init__(config)

    def predict(self, model: BaseModel, batches: Iterable[Batch]) -> PredictionBundle:
        cell_ids: list[str] = []
        predicted_labels: list[int] = []
        lineage_scores: list[list[float]] = []
        trajectory_latent: list[list[float]] = []
        metadata = []

        for batch in batches:
            output = model.forward(batch)
            for index, logits in enumerate(output.logits):
                scores = softmax(logits)
                lineage_scores.append(scores)
                predicted_labels.append(argmax(scores))
                trajectory_latent.append(output.embeddings[index])
                cell_ids.append(batch.cell_ids[index])
                metadata.append(dict(batch.metadata[index]))

        return PredictionBundle(
            cell_ids=cell_ids,
            predicted_labels=predicted_labels,
            lineage_scores=lineage_scores,
            trajectory_latent=trajectory_latent,
            metadata=metadata,
        )
