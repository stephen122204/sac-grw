"""Check solver identities and independently recompute central archived results."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ResultChecks(unittest.TestCase):
    def setUp(self):
        self.numbers = json.loads((ROOT / 'reproduction/analysis/numbers.json').read_text())


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
