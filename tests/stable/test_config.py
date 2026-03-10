from __future__ import annotations

import tempfile
import unittest

from spatial_lineage.stable.core.config import load_config
from tests.helpers import config_path


class ConfigTestCase(unittest.TestCase):
    def test_load_config_resolves_inheritance(self) -> None:
        config = load_config(config_path("experiments", "train", "mock_demo.yaml"))

        self.assertEqual(config.experiment.name, "mock_demo_train")
        self.assertEqual(config.model.name, "st_transformer_classifier")
        self.assertEqual(config.training.batch_size, 2)
        self.assertEqual(config.dataset.name, "spatial_lineage_dataset")

    def test_load_config_applies_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_config(
                config_path("experiments", "train", "mock_demo.yaml"),
                overrides=[
                    f"runtime.output_root={temp_dir}",
                    "training.batch_size=4",
                ],
            )

            self.assertEqual(config.runtime.output_root, temp_dir)
            self.assertEqual(config.training.batch_size, 4)
