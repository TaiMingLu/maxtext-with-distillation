
This is a fork of maxtext for training with distillation. For the native README of MaxText see [README_ORIGINAL.md](README_ORIGINAL.md).


# Distillation Extension for MaxText

This repository layers a configurable knowledge distillation (KD) workflow on top of MaxText. It focuses on student-teacher pretraining where the teacher supplies softened token distributions that guide the student while preserving the standard next-token objective.

## Overview
- Supports temperature-scaled KD blended with cross-entropy via `kd_alpha`.
- Adds optional top-k and top-p truncation of teacher distributions, with a choice between renormalizing probability mass or tracking dropped mass as an `OTHER` bucket.
- Keeps teacher parameters frozen while student weights update, with compatibility for multi-token prediction (MTP), mixture-of-experts (MoE), and gradient accumulation.
- Exposes KD metrics (`learning/kd_loss`, `evaluation/kd_loss`) alongside existing training logs.

## Distillation Options
| Setting | Description |
| --- | --- |
| `kd_alpha` | Weight on the KD loss vs. hard-label cross-entropy (`0.0` = pure CE, `1.0` = pure KD). |
| `kd_temperature` | Temperature applied to both teacher and student logits before the KL calculation. |
| `kd_top_k` | Retain only the highest-probability `k` teacher tokens (set `0` to disable). |
| `kd_top_p` | Retain the smallest prefix of teacher tokens whose cumulative probability is at least `p` (set `0.0` to disable). |
| `kd_truncation_strategy` | `renormalize` redistributes probability across the kept tokens; `other_bucket` adds a single aggregate category for the discarded mass. |
| `kd_teacher_parameters_path` | Path to the frozen teacher checkpoint (required when `use_kd=true`). |
| `kd_teacher_model_name` | Optional preset to load a different teacher architecture than the student. |

`kd_top_k` and `kd_top_p` are mutually exclusive. Leave both disabled to match full-vocabulary KD.

## Quick Start
1. Prepare a teacher checkpoint compatible with MaxText parameter loading.
2. Enable KD in your config or CLI overrides:
   ```bash
   python MaxText/train.py <config>.yml \
     use_kd=true \
     kd_teacher_parameters_path=gs://path/to/teacher \
     kd_alpha=0.5 \
     kd_temperature=2.0 \
     kd_top_k=20 \
     kd_truncation_strategy=renormalize
   ```
3. Launch training as usual; the training loop automatically performs both student and teacher forward passes and blends the losses.

## Workflow Details
- **Teacher Inference:** The teacher is evaluated without dropout, and its logits are stop-gradient to avoid updates.
- **Loss Composition:** The final loss is `(1 - kd_alpha) * CE + kd_alpha * T^2 * KL + aux_losses`. KD components are masked by `targets_segmentation` to ignore padding.
- **Gradient Accumulation:** Weighted sums of CE and KD are tracked per microbatch to maintain consistent averaging when `gradient_accumulation_steps > 1`.
- **MoE / MTP:** Existing auxiliary losses remain active; KD augments, rather than replaces, the original training path.

## Monitoring & Evaluation
- Watch `learning/kd_loss`, `learning/ce_loss`, and any MoE/MTP metrics during training.
- During evaluation, `evaluation/kd_loss` confirms the student still matches the teacher distribution under validation data.
- For truncation experiments, compare runs with different `kd_top_k`, `kd_top_p`, or `kd_truncation_strategy` to gauge their effect on convergence and compute cost.

## Tips for Effective KD
- Start with moderate temperature values (`1.5`–`2.0`) and adjust `kd_alpha` to balance imitation vs. label fitting.
- Use top-k or top-p truncation when teacher vocab sweeps are expensive or when focusing gradients on a small candidate set is desirable.
- Prefer `other_bucket` when you still want to penalize student probability outside the retained tokens; choose `renormalize` to ignore gradients on truncated classes entirely.
- Validate that teacher and student share tokenization and vocabulary definitions to avoid alignment issues.

With these controls, you can tailor distillation to match compute budgets, highlight high-probability teacher guidance, and mix KD with other MaxText training features.
