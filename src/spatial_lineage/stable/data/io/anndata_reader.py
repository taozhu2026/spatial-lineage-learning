from __future__ import annotations

from pathlib import Path
from typing import Any


def read_anndata_records(path: str | Path, data_config: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        import anndata  # type: ignore
    except ImportError as error:
        raise RuntimeError("Install `anndata` to read .h5ad datasets.") from error

    dataset_path = Path(path)
    adata = anndata.read_h5ad(dataset_path)
    expression = adata.X.toarray().tolist() if hasattr(adata.X, "toarray") else adata.X.tolist()
    spatial_key = "spatial"
    clone_key = "clone_id"
    label_key = "lineage_label"
    if isinstance(data_config, dict):
        spatial_key = data_config.get("spatial_key", spatial_key)
        clone_key = data_config.get("clone_key", clone_key)
        label_key = data_config.get("label_key", label_key)

    records: list[dict[str, Any]] = []
    spatial_coords = adata.obsm[spatial_key].tolist()
    clone_ids = adata.obs[clone_key].tolist()
    labels = adata.obs[label_key].tolist() if label_key in adata.obs else [0] * len(clone_ids)
    for index, vector in enumerate(expression):
        records.append(
            {
                "cell_id": str(adata.obs_names[index]),
                "expression": vector,
                "spatial_coords": spatial_coords[index],
                "clone_id": str(clone_ids[index]),
                "label": int(labels[index]),
            }
        )
    return records
