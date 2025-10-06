# Repository Guidelines

## Project Structure & Module Organization
`MaxText/` contains the core JAX implementation, configs, and reusable kernels; keep new model logic or utilities in the matching subpackage (e.g., `input_pipeline/`, `layers/`, `optimizers/`). `tests/` hosts regression and conversion workflows, while Python unit tests live in `MaxText/tests`. Use `end_to_end/`, `train/`, and `tuning/` for runnable job recipes; shared scripts belong in `scripts/`. The `lm-evaluation-harness/` submodule is required for Orbax adapters—run `git submodule update --init --recursive` after cloning.

## Build, Test, and Development Commands
Install dependencies with `python3 -m pip install -r requirements.txt` (or `pip install -e .` for editable dev). Use `bash setup.sh MODE=stable DEVICE=tpu` to prime TPU/GPU hosts. Run style and unit checks via `bash unit_test_and_lint.sh`, which executes `pylint` on tracked Python files and `pytest --pyargs MaxText.tests`. For targeted runs, invoke `python3 -m pytest tests -m "not tpu_only"` and add markers when hardware-specific.

## Coding Style & Naming Conventions
Python code follows Google style enforced by PyInk and Pylint: two-space indentation, max line length 125, and module/package names in lower_snake_case. Prefer explicit function names (`run_prefill_benchmark`) and docstrings describing TPU/GPU nuances. Before committing, call `bash code_style.sh --check` to keep formatting synchronized.

## Testing Guidelines
Write fast assertions under `MaxText/tests` using the `test_*.py` pattern; integration and conversion checks belong in `tests/` with `.sh` harnesses when TPU orchestration is required. Respect markers from `pytest.ini` (`tpu_only`, `gpu_only`, `integration_test`) so CI can shard appropriately. Add regression coverage for new configs or kernels alongside realistic fixtures, and document required hardware in the docstring.

## Knowledge Distillation & Loss Calculation
`MaxText/train.py` wires `use_kd=true` configs through `_split_kd_state`, keeping the teacher weights frozen while the student state updates. `kd_loss_fn` computes token-masked CE plus a temperature-scaled KL against teacher logits, then blends them via `kd_alpha`, reusing MTP and MoE regularizers from the standard path. Under gradient accumulation, the loop in `train_step` tracks token-weighted CE/KD components (`kd_weighted_sum`) so the final loss remains `(1-α)*CE + α*KD + aux_losses`. Log outputs expose `learning/ce_loss`, `learning/kd_loss`, and MoE metrics, making it easy to confirm the KD blend during experiments.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects (`sync run files`, `fix save convert path`). Mirror that style, keep scope focused, and bundle setup changes separately from training logic. Pull requests should describe motivation, list affected scripts/configs, and note any required assets or datasets. Include the commands you ran (`unit_test_and_lint.sh`, targeted `pytest`, custom benchmark scripts) so reviewers can reproduce results quickly.
