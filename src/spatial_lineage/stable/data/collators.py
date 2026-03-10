from __future__ import annotations

from typing import Any

from spatial_lineage.stable.contracts.batch import Batch


def collate_records(records: list[dict[str, Any]]) -> Batch:
    return Batch(
        cell_ids=[str(record["cell_id"]) for record in records],
        expression=[list(record["expression"]) for record in records],
        spatial_coords=[tuple(record["spatial_coords"]) for record in records],
        clone_ids=[str(record["clone_id"]) for record in records],
        targets=[int(record["label"]) for record in records] if "label" in records[0] else None,
        metadata=[
            {
                "clone_id": str(record["clone_id"]),
                "spatial_coords": list(record["spatial_coords"]),
                "target": int(record.get("label", -1)),
            }
            for record in records
        ],
    )
