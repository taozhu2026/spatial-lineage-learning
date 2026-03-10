from __future__ import annotations

from spatial_lineage import experimental  # noqa: F401
from spatial_lineage.stable import data as stable_data  # noqa: F401
from spatial_lineage.stable.core.config import ConfigNode
from spatial_lineage.stable.core.logging import RunLogger
from spatial_lineage.stable.core.paths import ExperimentPaths
from spatial_lineage.stable.core.registry import DATASETS, MODELS, PREDICTORS
from spatial_lineage.stable.core.seed import seed_everything
from spatial_lineage.stable.inference.engine import InferenceEngine


def build_inference_job(config: ConfigNode) -> InferenceEngine:
    _require_sections(config, ["runtime", "experiment", "model", "data", "inference"])
    seed_everything(int(config.runtime.seed))
    paths = ExperimentPaths.from_config(config)
    logger = RunLogger(paths.logs)

    dataset_cls = DATASETS.get(config.dataset.name if config.get("dataset") else "spatial_lineage_dataset")
    dataset = dataset_cls.from_config(config, split="inference")
    model = MODELS.build(config.model.name, config=config)
    predictor_name = config.predictor.name if config.get("predictor") else "lineage_score_predictor"
    predictor = PREDICTORS.build(predictor_name, config=config)
    return InferenceEngine(
        config=config,
        model=model,
        predictor=predictor,
        batches=dataset.batches(config.inference.batch_size),
        logger=logger,
        paths=paths,
    )


def _require_sections(config: ConfigNode, sections: list[str]) -> None:
    missing = [section for section in sections if config.get(section) is None]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")
