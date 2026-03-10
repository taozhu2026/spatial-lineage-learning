from __future__ import annotations

import unittest

from spatial_lineage import experimental  # noqa: F401
from spatial_lineage.stable.contracts.batch import Batch
from spatial_lineage.stable.core.config import load_config
from spatial_lineage.stable.core.registry import MODELS
from tests.helpers import config_path


class STTransformerModelTestCase(unittest.TestCase):
    def test_forward_returns_logits_and_embeddings(self) -> None:
        config = load_config(config_path("experiments", "train", "mock_demo.yaml"))
        model = MODELS.build("st_transformer_classifier", config=config)
        batch = Batch(
            cell_ids=["c1", "c2"],
            expression=[[0.9, 0.1, 0.2], [0.1, 0.8, 0.7]],
            spatial_coords=[(0.0, 0.0), (3.0, 3.0)],
            clone_ids=["clone_a", "clone_b"],
            targets=[0, 1],
            metadata=[{"clone_id": "clone_a"}, {"clone_id": "clone_b"}],
        )

        output = model.forward(batch)

        self.assertEqual(len(output.logits), 2)
        self.assertEqual(len(output.embeddings[0]), 4)
