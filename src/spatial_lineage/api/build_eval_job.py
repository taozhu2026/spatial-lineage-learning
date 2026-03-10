from __future__ import annotations

from spatial_lineage import experimental  # noqa: F401
from spatial_lineage.stable import data as stable_data  # noqa: F401
from spatial_lineage.stable.core.config import ConfigNode
from spatial_lineage.stable.core.logging import RunLogger
from spatial_lineage.stable.core.paths import ExperimentPaths
from spatial_lineage.stable.core.registry import DATASETS, MODELS, PREDICTORS
from spatial_lineage.stable.core.seed import seed_everything
from spatial_lineage.stable.evaluation.engine import EvaluationEngine


def build_eval_job(config: ConfigNode) -> EvaluationEngine:
    _require_sections(config, ["runtime", "experiment", "model", "data", "dataset", "evaluation"])
    seed_everything(int(config.runtime.seed))
    paths = ExperimentPaths.from_config(config)
    logger = RunLogger(paths.logs)

    dataset_cls = DATASETS.get(config.dataset.name)
    dataset = dataset_cls.from_config(config, split=config.evaluation.split)
    model = MODELS.build(config.model.name, config=config)
    predictor_name = config.predictor.name if config.get("predictor") else "lineage_score_predictor"
    predictor = PREDICTORS.build(predictor_name, config=config)
    return EvaluationEngine(
        config=config,
        model=model,
        predictor=predictor,
        batches=dataset.batches(config.evaluation.batch_size),
        logger=logger,
        paths=paths,
    )


def _require_sections(config: ConfigNode, sections: list[str]) -> None:
    missing = [section for section in sections if config.get(section) is None]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")
