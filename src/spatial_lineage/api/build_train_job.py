from __future__ import annotations

from spatial_lineage import experimental  # noqa: F401
from spatial_lineage.stable import data as stable_data  # noqa: F401
from spatial_lineage.stable.core.config import ConfigNode
from spatial_lineage.stable.core.logging import RunLogger
from spatial_lineage.stable.core.paths import ExperimentPaths
from spatial_lineage.stable.core.registry import DATASETS, LOSSES, MODELS
from spatial_lineage.stable.core.seed import seed_everything
from spatial_lineage.stable.training.engine import TrainingEngine


def build_train_job(config: ConfigNode) -> TrainingEngine:
    _require_sections(config, ["runtime", "experiment", "training", "model", "data", "dataset", "task", "loss"])
    seed_everything(int(config.runtime.seed))
    paths = ExperimentPaths.from_config(config)
    logger = RunLogger(paths.logs)

    dataset_cls = DATASETS.get(config.dataset.name)
    train_dataset = dataset_cls.from_config(config, split="train")
    valid_dataset = dataset_cls.from_config(config, split="val")
    model = MODELS.build(config.model.name, config=config)
    loss_fn = LOSSES.build(config.loss.name, config=config)
    return TrainingEngine(
        config=config,
        model=model,
        loss_fn=loss_fn,
        train_batches=train_dataset.batches(config.training.batch_size),
        valid_batches=valid_dataset.batches(config.evaluation.batch_size if config.get("evaluation") else config.training.batch_size),
        logger=logger,
        paths=paths,
    )


def _require_sections(config: ConfigNode, sections: list[str]) -> None:
    missing = [section for section in sections if config.get(section) is None]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")
