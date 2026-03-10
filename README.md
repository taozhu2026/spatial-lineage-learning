# spatial-lineage-learning

`spatial-lineage-learning` is a config-driven Python training scaffold for spatial transcriptomics and lineage tracing research.

It is designed for teams that need:

- stable infrastructure for long-lived training pipelines
- fast iteration on models, losses, feature builders, and evaluators
- one codebase for `train`, `eval`, and `inference`
- experiment switching through YAML config composition instead of code edits

## Highlights

- `stable/` and `experimental/` are explicitly separated
- YAML `_base_` inheritance supports reusable config composition
- train/eval/infer share the same registry and runtime conventions
- includes runnable mock configs for local validation
- includes real-project templates for Visium/Xenium-style datasets

## Repository Layout

```text
configs/
├── base/                  # reusable runtime, model, data, train, eval, inference defaults
├── datasets/              # dataset-specific configs
├── tasks/                 # task and loss configs
├── experiments/           # train/eval/inference experiment entries
└── schemas/               # required-section contracts for each command

src/spatial_lineage/
├── stable/
│   ├── core/              # config, registry, logging, experiment paths, seeding
│   ├── contracts/         # batch/model/loss/trainer/evaluator/predictor interfaces
│   ├── data/              # readers, dataset wrapper, collators, transforms, samplers
│   ├── training/          # training engine, checkpointing, optimizer, scheduler
│   ├── evaluation/        # evaluation engine and stable metrics
│   └── inference/         # inference engine, postprocess, writers
├── experimental/
│   ├── models/            # research models
│   ├── losses/            # research losses
│   ├── feature_builders/  # spatial and lineage feature construction
│   ├── evaluators/        # research metrics/evaluators
│   └── predictors/        # research-time prediction logic
├── api/                   # job builders
└── commands/              # CLI entrypoints

scripts/                   # thin executable wrappers
tests/                     # stable and experimental unit tests
tools/                     # helper scripts for config/checkpoint inspection
```

## Stable vs Experimental

`stable/` contains infrastructure that should rarely change:

- config parsing and overrides
- registry wiring
- data loading conventions
- engine orchestration
- checkpoint and artifact writing

`experimental/` contains code that will change frequently:

- new model backbones
- new losses
- new graph or neighborhood feature builders
- research evaluators and predictors

Rule: `experimental/*` may depend on `stable/*`, but `stable/*` should not import concrete research implementations directly.

## Quick Start

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[dev,data]"
```

Use the `data` extra if you want `.h5ad` or parquet IO support.

### 2. Run the local mock pipeline

```bash
python scripts/train.py configs/experiments/train/mock_demo.yaml
python scripts/eval.py configs/experiments/eval/mock_demo_eval.yaml
python scripts/infer.py configs/experiments/inference/mock_demo_infer.yaml
```

### 3. Run tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Config System

Configs are composed with `_base_` inheritance.

Example:

```yaml
_base_:
  - ../../base/runtime.yaml
  - ../../base/train.yaml
  - ../../base/eval.yaml
  - ../../base/model.yaml
  - ../../datasets/mock_small.yaml
  - ../../tasks/lineage_classification.yaml

experiment:
  name: mock_demo_train

training:
  epochs: 2
  batch_size: 2
```

CLI overrides are supported:

```bash
python scripts/train.py configs/experiments/train/mock_demo.yaml \
  -o runtime.output_root=/tmp/spatial_lineage_runs \
  -o training.batch_size=8
```

## Commands

Training:

```bash
python scripts/train.py configs/experiments/train/exp001_st_ln_transformer.yaml
```

Evaluation:

```bash
python scripts/eval.py configs/experiments/eval/exp001_eval_holdout.yaml
```

Inference:

```bash
python scripts/infer.py configs/experiments/inference/exp001_predict_new_slide.yaml
```

Installed console scripts:

- `spatial-lineage-train`
- `spatial-lineage-eval`
- `spatial-lineage-infer`

## What Is Included Today

- config loader with recursive inheritance and CLI overrides
- component registries for models, losses, datasets, evaluators, and predictors
- stable training, evaluation, and inference engines
- a minimal `st_transformer_classifier` example model
- mock dataset configs that run without external files
- real dataset config templates for future AnnData-based projects
- unit tests for config loading, registry behavior, and end-to-end command flows

## Development Notes

- default lightweight runs use inline mock records and pure Python logic
- `.h5ad` reading requires `anndata`
- parquet reading/writing requires `pandas` and `pyarrow`
- experiment outputs are written under `experiments/` and are gitignored

## Roadmap

- replace the mock model path with a real PyTorch training stack
- add typed schema validation for each config family
- add distributed training adapters
- add richer evaluation for spatial coherence and trajectory alignment
- add dataset converters for Visium, Xenium, and custom lineage assays

## License

MIT. See [LICENSE](LICENSE).
