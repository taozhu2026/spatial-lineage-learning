# Contributing

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Install `.[data]` as well if you need AnnData or parquet support.

## Workflow

1. Keep infrastructure changes inside `src/spatial_lineage/stable/`.
2. Put new research code inside `src/spatial_lineage/experimental/`.
3. Add or update configs under `configs/` instead of hard-coding experiment choices.
4. Add tests for any new registry component or engine behavior.

## Validation

Run before opening a pull request:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python scripts/train.py configs/experiments/train/mock_demo.yaml
python scripts/eval.py configs/experiments/eval/mock_demo_eval.yaml
python scripts/infer.py configs/experiments/inference/mock_demo_infer.yaml
```
