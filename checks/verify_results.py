"""Check solver identities and independently recompute central archived results."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from coupled_gradient_particles import advance_pair
from gradient_particles import advance_particles, reconstruct_cumulative_field

POLICIES = ('RAW', 'FULL', 'WITHIN', 'FINAL')


class ResultChecks(unittest.TestCase):
    def setUp(self):
        x = np.linspace(-0.5, 0.5, 24)
        self.initial = (x, np.r_[np.full(12, 0.04), np.full(12, -0.04)], 0.0)
        self.grid = np.linspace(-2, 2, 101)
        self.numbers = json.loads((ROOT / 'reproduction/analysis/numbers.json').read_text())

    def test_one_sign_paths_coincide(self):
        initial = (self.initial[0], np.full(24, -1 / 24), 1.0)
        for flux in (lambda u: np.zeros_like(u), lambda u: u, lambda u: u**2):
            expected = advance_pair(initial, 0.1, 0.005, 40,
                                    np.random.default_rng(73), 'RAW', flux)
            for policy in POLICIES:
                with self.subTest(policy=policy):
                    actual = advance_pair(initial, 0.1, 0.005, 40,
                                          np.random.default_rng(73), policy, flux)
                    for pair in (0, 1):
                        for component in (0, 1):
                            np.testing.assert_array_equal(actual[pair][component],
                                                          expected[pair][component])

    def test_first_replica_matches_single_update(self):
        noise = np.random.default_rng(52).standard_normal((30, 24))

        class Supply:
            def __init__(self):
                self.index = 0

            def standard_normal(self, size):
                result = noise[self.index].copy()
                self.index += 1
                return result

            def normal(self, loc, scale, size):
                return loc + scale * self.standard_normal(size)

        x, masses, left = self.initial
        single = advance_particles(x, masses, left, 0.1, 2.0, 0.005, 30,
                                   np.random.default_rng(0), rng_brownian=Supply(),
                                   conditional_mean_transport=True)
        for policy in POLICIES:
            with self.subTest(policy=policy):
                first, _, _ = advance_pair(self.initial, 0.1, 0.005, 30,
                                           Supply(), policy, lambda u: u)
                np.testing.assert_array_equal(first[0], single['x'])
                np.testing.assert_array_equal(first[1], single['m'])
                np.testing.assert_array_equal(
                    reconstruct_cumulative_field(*first, left, self.grid),
                    reconstruct_cumulative_field(single['x'], single['m'], left, self.grid))

    def test_masses_and_input_are_preserved(self):
        original = [np.copy(a) for a in self.initial[:2]]
        for policy in POLICIES:
            result = advance_pair(self.initial, 0.1, 0.005, 40,
                                  np.random.default_rng(12), policy, lambda u: u**2)
            for positions, masses in result[:2]:
                self.assertTrue(np.isfinite(positions).all())
                np.testing.assert_array_equal(np.sort(masses), np.sort(original[1]))
        for actual, expected in zip(self.initial[:2], original):
            np.testing.assert_array_equal(actual, expected)

    def test_reconstruction_at_jumps(self):
        field = reconstruct_cumulative_field([1, 0, 0], [-0.25, 0.5, 0.25], 2,
                                             [-1, 0, 0.5, 1, 2])
        np.testing.assert_array_equal(field, [2, 2.75, 2.75, 2.5, 2.5])

    def test_principal_variance_and_mse(self):
        with np.load(ROOT / 'reproduction/evidence/principal_fields.npz') as z:
            dx = z['x'][1] - z['x'][0]
            variances = {}
            for policy in ('RAW', 'FULL', 'WITHIN'):
                field = (z[policy + '_A'] + z[policy + '_B']) / 2
                variance = dx * field.var(axis=0, ddof=1).sum()
                mse = np.mean(dx * ((field - z['reference'])**2).sum(axis=1))
                variances[policy] = variance
                np.testing.assert_allclose(variance, self.numbers['field']['integrated_variance'][policy])
                np.testing.assert_allclose(mse, self.numbers['field']['integrated_mse'][policy])
                bias = dx * ((field.mean(axis=0) - z['reference'])**2).sum()
                np.testing.assert_allclose(mse, bias + (len(field) - 1) / len(field) * variance)
            for name, row in self.numbers['field']['ratios'].items():
                numerator, denominator = name.split('/')
                np.testing.assert_allclose(row['variance'], variances[numerator] / variances[denominator])

    def test_heldout_target_attainment(self):
        for key, filename in (('heldout_A', 'heldout_work_experiment_a/compare.json'),
                              ('heldout_B', 'heldout_work_experiment_b/worktarget.json')):
            raw = json.loads((ROOT / 'output' / filename).read_text())
            summary = self.numbers[key]
            count = len(summary['arms'])
            for policy, reported in summary['arms'].items():
                with self.subTest(experiment=key, policy=policy):
                    errors = np.asarray(raw['realised'][policy]['per_block'])
                    se = errors.std(ddof=1) / np.sqrt(len(errors))
                    upper = errors.mean() + t.ppf(1 - 0.05 / (2 * count), len(errors) - 1) * se
                    np.testing.assert_allclose(reported['attained'], errors.mean())
                    np.testing.assert_allclose(reported['se'], se)
                    np.testing.assert_allclose(reported['upper_95_bonferroni'], upper)
                    self.assertLess(upper, summary['target'])
                    multiplier = 1 if policy in ('SINGLE', 'RQMC') else 2
                    self.assertEqual(reported['trajectories'], multiplier * raw['plans'][policy]['B'])


if __name__ == '__main__':
    from verify_coupling import CouplingTests
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(CouplingTests),
        unittest.defaultTestLoader.loadTestsFromTestCase(ResultChecks),
    ])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
