import random
import unittest

import torch

from gr00t.experiment.opqd import OPQDSelectionConfig, select_phase_balanced_states


class OPQDSelectionTest(unittest.TestCase):
    def test_balanced_selection_has_expected_budget_and_gap(self) -> None:
        q_values = torch.linspace(0.0, 1.0, 160)
        r_values = torch.linspace(1.0, 0.0, 160)
        result = select_phase_balanced_states(
            q_values,
            r_values,
            OPQDSelectionConfig(),
            rng=random.Random(7),
        )
        self.assertEqual(len(result.indices), 32)
        self.assertEqual([result.phases.count(phase) for phase in range(4)], [8, 8, 8, 8])
        self.assertFalse(result.gap_relaxed)
        self.assertTrue(
            all(right - left >= 4 for left, right in zip(result.indices, result.indices[1:]))
        )
        self.assertEqual(sum(reason.startswith("priority") for reason in result.reasons), 16)
        self.assertEqual(sum(reason.startswith("random") for reason in result.reasons), 16)

    def test_seed_reproduces_random_half(self) -> None:
        q_values = torch.rand(120, generator=torch.Generator().manual_seed(3))
        r_values = torch.rand(120, generator=torch.Generator().manual_seed(4))
        first = select_phase_balanced_states(
            q_values, r_values, OPQDSelectionConfig(), rng=random.Random(11)
        )
        second = select_phase_balanced_states(
            q_values, r_values, OPQDSelectionConfig(), rng=random.Random(11)
        )
        self.assertEqual(first.indices, second.indices)
        self.assertEqual(first.reasons, second.reasons)
        torch.testing.assert_close(first.weights, second.weights)

    def test_short_trajectory_relaxes_gap_without_duplicates(self) -> None:
        q_values = torch.arange(20, dtype=torch.float32)
        result = select_phase_balanced_states(
            q_values,
            q_values.flip(0),
            OPQDSelectionConfig(min_temporal_gap=4),
            rng=random.Random(0),
        )
        self.assertEqual(len(result.indices), 20)
        self.assertEqual(len(set(result.indices)), 20)
        self.assertTrue(result.gap_relaxed)

    def test_empty_trajectory(self) -> None:
        result = select_phase_balanced_states(
            torch.empty(0),
            torch.empty(0),
            OPQDSelectionConfig(),
            rng=random.Random(0),
        )
        self.assertEqual(result.indices, ())


if __name__ == "__main__":
    unittest.main()
