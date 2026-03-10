from __future__ import annotations

from collections import Counter

from spatial_lineage.stable.contracts.outputs import PredictionBundle
from spatial_lineage.stable.evaluation.metrics.calibration import mean_confidence


def summarize_bundle(bundle: PredictionBundle) -> dict[str, object]:
    label_counts = Counter(bundle.predicted_labels)
    return {
        "num_predictions": len(bundle.predicted_labels),
        "label_distribution": dict(label_counts),
        "mean_confidence": mean_confidence(bundle.lineage_scores),
    }
