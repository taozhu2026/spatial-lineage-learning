from __future__ import annotations

from pathlib import Path
from typing import Any


def read_parquet_records(path: str | Path, data_config: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except ImportError as error:
        raise RuntimeError("Install `pandas` and `pyarrow` to read parquet datasets.") from error

    dataset_path = Path(path)
    frame = pd.read_parquet(dataset_path)
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        records.append(
            {
                "cell_id": str(row["cell_id"]),
                "expression": list(row["expression"]),
                "spatial_coords": list(row["spatial_coords"]),
                "clone_id": str(row["clone_id"]),
                "label": int(row.get("label", 0)),
            }
        )
    return records
