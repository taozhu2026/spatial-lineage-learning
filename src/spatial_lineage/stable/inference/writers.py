from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spatial_lineage.stable.contracts.outputs import PredictionBundle


def write_prediction_bundle(
    output_dir: str | Path,
    bundle: PredictionBundle,
    output_format: str,
) -> dict[str, str]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    records = bundle.to_records()

    if output_format == "jsonl":
        predictions_path = write_records_jsonl(target_dir / "predictions.jsonl", records)
    elif output_format == "csv":
        predictions_path = write_records_csv(target_dir / "predictions.csv", records)
    elif output_format == "parquet":
        predictions_path = _write_records_parquet(target_dir / "predictions.parquet", records)
    else:
        raise ValueError(f"Unsupported inference.output_format: {output_format}")

    embeddings_path = target_dir / "embeddings.json"
    with embeddings_path.open("w", encoding="utf-8") as handle:
        json.dump(bundle.trajectory_latent, handle, indent=2)

    return {
        "predictions": str(predictions_path),
        "embeddings": str(embeddings_path),
    }


def write_records_jsonl(path: str | Path, records: list[dict[str, Any]]) -> Path:
    target = Path(path)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return target


def write_records_csv(path: str | Path, records: list[dict[str, Any]]) -> Path:
    import csv

    target = Path(path)
    fieldnames = sorted(records[0]) if records else []
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return target


def _write_records_parquet(path: str | Path, records: list[dict[str, Any]]) -> Path:
    try:
        import pandas as pd  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "Parquet output requires `pandas` and `pyarrow`; use jsonl/csv for lightweight runs."
        ) from error

    target = Path(path)
    pd.DataFrame(records).to_parquet(target, index=False)
    return target
