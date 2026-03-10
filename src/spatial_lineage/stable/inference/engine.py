from __future__ import annotations

import json
from typing import Iterable

from spatial_lineage.stable.contracts.batch import Batch
from spatial_lineage.stable.contracts.model import BaseModel
from spatial_lineage.stable.contracts.outputs import PredictionBundle
from spatial_lineage.stable.contracts.predictor import BasePredictor
from spatial_lineage.stable.core.config import ConfigNode
from spatial_lineage.stable.core.logging import RunLogger
from spatial_lineage.stable.core.paths import ExperimentPaths
from spatial_lineage.stable.inference.postprocess import summarize_bundle
from spatial_lineage.stable.inference.writers import write_prediction_bundle
from spatial_lineage.stable.training.checkpoint import load_checkpoint


class InferenceEngine:
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

    def run(self) -> dict[str, str]:
        self.paths.ensure()
        checkpoint = load_checkpoint(self.config.inference.checkpoint_path)
        self.model.load_state_dict(checkpoint.get("model_state", {}))
        bundle = self.predictor.predict(self.model, self.batches)
        output_paths = write_prediction_bundle(
            output_dir=self.paths.inference_dir,
            bundle=bundle,
            output_format=self.config.inference.output_format,
        )

        summary_path = self.paths.inference_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summarize_bundle(bundle), handle, indent=2, sort_keys=True)
        output_paths["summary"] = str(summary_path)
        self.logger.info(f"Saved inference artifacts to {self.paths.inference_dir}")
        return output_paths
