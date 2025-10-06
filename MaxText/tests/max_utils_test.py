"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

""" Tests for the common Max Utils """
import unittest

import jax
from jax import numpy as jnp
from jax import random

from flax import linen as nn

import optax

from MaxText import max_utils


class MaxUtilsSummaryStats(unittest.TestCase):
  """Tests for the summary stats functions in max_utils.py"""

  def test_l2norm_pytree(self):
    x = {"a": jax.numpy.array([0, 2, 0]), "b": jax.numpy.array([0, 3, 6])}
    pytree_l2_norm = max_utils.l2norm_pytree(x)
    self.assertTrue(jax.numpy.allclose(pytree_l2_norm, 7, rtol=1e-05, atol=1e-08, equal_nan=False))


class MaxUtilsPytree(unittest.TestCase):
  """Tests initialization of training and decode states in max_utils.py"""

  def setUp(self):
    self.model = nn.Dense(features=5)
    self.key1, self.key2 = random.split(random.key(0))
    self.input = random.normal(self.key1, (10,))  # Dummy input data
    self.params = self.model.init(self.key2, self.input)

  def test_calculate_num_params_from_pytree(self):
    example_tree = [
        [1, "a", object()],
        (1, (2, 3), ()),
        [1, {"k1": 2, "k2": (3, 4)}, 5],
        {"a": 2, "b": (2, 3)},
        jnp.array([1, 2, 3]),
    ]
    self.assertEqual(max_utils.calculate_num_params_from_pytree(example_tree), 17)
    # Model params
    self.assertEqual(max_utils.calculate_num_params_from_pytree(self.params), 55)


class MaxUtilsT5XCrossEntropy(unittest.TestCase):
  """Tests for the cross entropy functions in max_utils.py"""

  def test_t5x_cross_entropy(self):
    # Generate random targets and logits
    key = jax.random.PRNGKey(0)
    targets = jax.random.randint(key, shape=(48, 2048), dtype=jax.numpy.int32, minval=1, maxval=10)
    logits = jax.random.uniform(key, shape=(48, 2048, 4096), dtype=jax.numpy.float32)

    # Calculate xent from optax implementation
    optax_xent = optax.softmax_cross_entropy_with_integer_labels(logits, targets)

    # Calculate xent from custom T5X implementation
    one_hot_targets = jax.nn.one_hot(targets, 4096)
    t5x_xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets, 0.0)
    t5x_xent = nn.with_logical_constraint(t5x_xent, ("activation_batch", "activation_length"))

    # Compare results
    self.assertTrue(jax.numpy.allclose(optax_xent, t5x_xent, rtol=1e-05, atol=1e-08, equal_nan=False))


class MaxUtilsCustomMesh(unittest.TestCase):
  """Tests for the is_valid_custom_mesh function in max_utils.py"""

  def test_empty_value(self):
    self.assertFalse(max_utils.is_valid_custom_mesh([1, 1, 1, 1, 1, 64, 4, 1], ""))

  def test_valid_64x4(self):
    self.assertTrue(max_utils.is_valid_custom_mesh([1, 1, 1, 1, 1, 64, 4, 1], "hybrid_ring_64x4"))

  def test_valid_32x8(self):
    self.assertTrue(max_utils.is_valid_custom_mesh([1, 1, 32, 1, 1, 8, 1, 1], "hybrid_ring_32x8"))

  def test_invalid_64x4(self):
    with self.assertRaises(ValueError):
      max_utils.is_valid_custom_mesh([1, 1, 1, 1, 1, 16, 16, 1], "hybrid_ring_64x4")

  def test_invalid_strategy(self):
    with self.assertRaises(ValueError):
      max_utils.is_valid_custom_mesh([1, 1, 1, 1, 1, 16, 16, 1], "invalid_strategy")


class MaxUtilsKDTruncation(unittest.TestCase):
  """Tests for KD top-k / top-p truncation helpers."""

  def setUp(self):
    self.student_logits = jnp.array([[0.0, 0.0, 0.0, 0.0]])
    self.teacher_logits = jnp.array([[2.0, 1.0, 0.0, -1.0]])

  def test_top_k_renormalize_matches_manual(self):
    actual = max_utils.kl_divergence_between_logits(
        self.student_logits, self.teacher_logits, 1.0, top_k=2, truncation_strategy="renormalize"
    )
    teacher_probs = jax.nn.softmax(self.teacher_logits, axis=-1)
    teacher_kept = teacher_probs[..., :2]
    teacher_kept /= jnp.sum(teacher_kept, axis=-1, keepdims=True)
    student_log_probs = jax.nn.log_softmax(self.student_logits, axis=-1)
    expected = jnp.sum(
        teacher_kept * (jnp.log(teacher_kept) - student_log_probs[..., :2]), axis=-1
    )
    self.assertTrue(jnp.allclose(actual, expected, rtol=1e-6, atol=1e-6))

  def test_top_p_other_bucket_penalizes_outside(self):
    actual = max_utils.kl_divergence_between_logits(
        self.student_logits, self.teacher_logits, 1.0, top_p=0.8, truncation_strategy="other_bucket"
    )
    teacher_probs = jax.nn.softmax(self.teacher_logits, axis=-1)
    student_log_probs = jax.nn.log_softmax(self.student_logits, axis=-1)
    student_probs = jnp.exp(student_log_probs)
    teacher_kept = teacher_probs[..., :2]
    student_kept_probs = student_probs[..., :2]
    expected = jnp.sum(
        teacher_kept * (jnp.log(teacher_kept) - student_log_probs[..., :2]), axis=-1
    )
    teacher_other = 1.0 - jnp.sum(teacher_kept, axis=-1)
    student_other = 1.0 - jnp.sum(student_kept_probs, axis=-1)
    expected += teacher_other * (jnp.log(teacher_other) - jnp.log(student_other))
    self.assertTrue(jnp.allclose(actual, expected, rtol=1e-6, atol=1e-6))

  def test_invalid_top_k_top_p_combo_raises(self):
    with self.assertRaisesRegex(ValueError, "Only one of top_k or top_p"):
      max_utils.kl_divergence_between_logits(
          self.student_logits,
          self.teacher_logits,
          1.0,
          top_k=2,
          top_p=0.5,
      )


if __name__ == "__main__":
  unittest.main()
