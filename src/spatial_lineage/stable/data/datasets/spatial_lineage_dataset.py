from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from spatial_lineage.stable.core.config import ConfigNode
from spatial_lineage.stable.core.registry import DATASETS
from spatial_lineage.stable.data.collators import collate_records
from spatial_lineage.stable.data.io import read_anndata_records, read_parquet_records
from spatial_lineage.stable.data.samplers import batch_records
from spatial_lineage.stable.data.transforms import apply_transforms


def _as_plain_dict(config: ConfigNode | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(config, ConfigNode):
        return config.to_dict()
    return config or {}


@DATASETS.register("spatial_lineage_dataset")
@dataclass
class SpatialLineageDataset:
    split: str
    records: list[dict[str, Any]]

    @classmethod
    def from_config(cls, config: ConfigNode, split: str) -> "SpatialLineageDataset":
        data_config = _as_plain_dict(config.data)
        dataset_config = _as_plain_dict(config.get("dataset"))

        inline_records = dataset_config.get("inline_records", {})
        if split in inline_records:
            records = list(inline_records[split])
        elif split == "inference" and "inline_records" in _as_plain_dict(config.get("inference")):
            records = list(_as_plain_dict(config.inference)["inline_records"])
        else:
            path = _resolve_split_path(config, split)
            records = _load_records(path=path, data_config=data_config)

        return cls(split=split, records=apply_transforms(records, data_config))

    def batches(self, batch_size: int) -> Iterable:
        for record_batch in batch_records(self.records, batch_size=batch_size):
            yield collate_records(record_batch)


def _resolve_split_path(config: ConfigNode, split: str) -> Path:
    if split == "inference":
        input_path = config.inference.input_path
        if not input_path:
            raise ValueError("inference.input_path is required when inline_records are absent")
        return Path(input_path)

    split_key = f"{split}_path"
    path = config.dataset.get(split_key)
    if not path:
        raise ValueError(f"dataset.{split_key} is required when inline_records are absent")
    return Path(path)


def _load_records(path: Path, data_config: dict[str, Any]) -> list[dict[str, Any]]:
    reader_name = data_config.get("reader", "anndata")
    if reader_name == "anndata":
        return read_anndata_records(path, data_config)
    if reader_name == "parquet":
        return read_parquet_records(path, data_config)
    raise ValueError(f"Unsupported data.reader: {reader_name}")
