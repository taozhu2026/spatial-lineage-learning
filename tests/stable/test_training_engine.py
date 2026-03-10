from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from spatial_lineage.api import build_train_job
from spatial_lineage.stable.core.config import load_config
from tests.helpers import config_path


class TrainingEngineTestCase(unittest.TestCase):
    def test_training_job_writes_checkpoint_and_frozen_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "experiments"
            config = load_config(
                config_path("experiments", "train", "mock_demo.yaml"),
                overrides=[
                    f"runtime.output_root={output_root}",
                    "experiment.name=test_training_job",
                ],
            )

            result = build_train_job(config).run()

            self.assertTrue(Path(result.checkpoint_path).exists())
            self.assertTrue((output_root / "test_training_job" / "frozen_config.yaml").exists())
