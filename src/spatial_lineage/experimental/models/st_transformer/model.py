from __future__ import annotations

from typing import Any

from spatial_lineage.experimental.feature_builders.lineage_graph import clone_bias
from spatial_lineage.experimental.feature_builders.neighborhood_pooling import pooled_expression
from spatial_lineage.experimental.feature_builders.spatial_graph import spatial_signal
from spatial_lineage.stable.contracts.batch import Batch
from spatial_lineage.stable.contracts.model import BaseModel
from spatial_lineage.stable.contracts.outputs import ModelOutput
from spatial_lineage.stable.core.registry import MODELS


@MODELS.register("st_transformer_classifier")
class STTransformerClassifier(BaseModel):
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.num_classes = int(config.task.num_classes)
        self.hidden_dim = int(config.model.hidden_dim)
        self._state: dict[str, Any] = {"hidden_dim": self.hidden_dim}

    def forward(self, batch: Batch) -> ModelOutput:
        logits: list[list[float]] = []
        embeddings: list[list[float]] = []
        for expression, coords, clone_id in zip(batch.expression, batch.spatial_coords, batch.clone_ids):
            pooled = pooled_expression(expression)
            spatial = spatial_signal(coords)
            lineage = clone_bias(clone_id)
            embedding = [pooled, spatial[0], spatial[1], lineage]
            embeddings.append(embedding)
            logits.append(
                [
                    pooled * (index + 1) + spatial[0] * 0.3 + spatial[1] * 0.2 + lineage * (index + 1)
                    for index in range(self.num_classes)
                ]
            )
        return ModelOutput(logits=logits, embeddings=embeddings, aux_losses={"embedding_dim": float(len(embeddings[0])) if embeddings else 0.0})

    def state_dict(self) -> dict[str, Any]:
        return dict(self._state)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._state.update(state)
