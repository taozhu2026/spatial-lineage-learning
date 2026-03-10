from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from spatial_lineage.api import build_eval_job, build_train_job
from spatial_lineage.stable.core.config import load_config
from tests.helpers import config_path


class EvaluationEngineTestCase(unittest.TestCase):
    def test_eval_job_writes_metrics_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "experiments"
            train_config = load_config(
                config_path("experiments", "train", "mock_demo.yaml"),
                overrides=[
                    f"runtime.output_root={output_root}",
                    "experiment.name=train_for_eval",
                ],
            )
            train_result = build_train_job(train_config).run()

            eval_config = load_config(
                config_path("experiments", "eval", "mock_demo_eval.yaml"),
                overrides=[
                    f"runtime.output_root={output_root}",
                    "experiment.name=test_eval_job",
                    f"evaluation.checkpoint_path={train_result.checkpoint_path}",
                ],
            )
            report = build_eval_job(eval_config).run()

            self.assertIn("accuracy", report.metrics)
            self.assertTrue(Path(report.output_paths["metrics"]).exists())
            self.assertTrue(Path(report.output_paths["predictions"]).exists())
