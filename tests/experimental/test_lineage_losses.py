from __future__ import annotations

import unittest

from spatial_lineage import experimental  # noqa: F401
from spatial_lineage.experimental.losses.clone_aware_ce import CloneAwareCrossEntropy
from spatial_lineage.stable.contracts.batch import Batch
from spatial_lineage.stable.contracts.outputs import ModelOutput
from spatial_lineage.stable.core.config import load_config
from tests.helpers import config_path


class CloneAwareLossTestCase(unittest.TestCase):
    def test_clone_aware_cross_entropy_returns_positive_loss(self) -> None:
        config = load_config(config_path("experiments", "train", "mock_demo.yaml"))
        loss_fn = CloneAwareCrossEntropy(config=config)
        batch = Batch(
            cell_ids=["c1", "c2"],
            expression=[[0.9, 0.1, 0.2], [0.1, 0.8, 0.7]],
            spatial_coords=[(0.0, 0.0), (3.0, 3.0)],
            clone_ids=["clone_a", "clone_b"],
            targets=[0, 1],
            metadata=[{}, {}],
        )
        output = ModelOutput(
            logits=[[2.0, 0.5], [0.4, 2.3]],
            embeddings=[[0.1, 0.0], [0.2, 0.5]],
        )

        result = loss_fn.compute(batch, output)

        self.assertGreater(result.loss, 0.0)
