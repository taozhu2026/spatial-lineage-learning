from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from spatial_lineage.api import build_inference_job, build_train_job
from spatial_lineage.stable.core.config import load_config
from tests.helpers import config_path


class InferenceEngineTestCase(unittest.TestCase):
    def test_inference_job_writes_summary_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "experiments"
            train_config = load_config(
                config_path("experiments", "train", "mock_demo.yaml"),
                overrides=[
                    f"runtime.output_root={output_root}",
                    "experiment.name=train_for_infer",
                ],
            )
            train_result = build_train_job(train_config).run()

            infer_config = load_config(
                config_path("experiments", "inference", "mock_demo_infer.yaml"),
                overrides=[
                    f"runtime.output_root={output_root}",
                    "experiment.name=test_infer_job",
                    f"inference.checkpoint_path={train_result.checkpoint_path}",
                ],
            )
            output_paths = build_inference_job(infer_config).run()

            self.assertTrue(Path(output_paths["predictions"]).exists())
            self.assertTrue(Path(output_paths["summary"]).exists())
