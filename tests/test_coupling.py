"""Independent checks of the published coupling and its discrete baseline."""

import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from coupled_gradient_particles import advance_pair
from relaxation_gbmc import advance_rbgbmc_particles, reconstruct_cumulative_field


class RecordedRNG:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.draws = []

    def standard_normal(self, n):
        z = self.rng.standard_normal(n)
        self.draws.append(z.copy())
        return z


class PrescribedNoise:
    def __init__(self, displacements):
        self.values = iter(displacements)

    def normal(self, loc, scale, size):
        value = next(self.values)
        assert value.shape == (size,)
        return value.copy()


class CouplingTests(unittest.TestCase):
    def test_both_replicas_against_production_complete_state(self):
        x = np.linspace(-0.1, 0.1, 12)
        x[4] = x[3]  # ties and transport crossings
        m = np.tile([0.05, -0.05], 6)
        for policy in ("RAW", "FULL", "WITHIN", "FINAL"):
            pre = []
            rng = RecordedRNG(318)
            A, B, ul = advance_pair(
                (x, m, 0.0),
                0.1,
                0.005,
                20,
                rng,
                policy,
                lambda u: u,
                pre_diffusion=lambda *s: pre.append(s),
            )
            sd = np.sqrt(2 * 0.1 * 0.005)
            noiseA = []
            noiseB = []
            for k, z in enumerate(rng.draws):
                ma, mb = pre[k][2], pre[k][5]
                za = sd * z
                if policy == "WITHIN":
                    zb = np.empty_like(z)
                    for sign in (-1, 1):
                        ia = np.flatnonzero(np.sign(ma) == sign)
                        ib = np.flatnonzero(np.sign(mb) == sign)
                        zb[ib] = -z[ia]
                    zb = sd * zb
                elif policy == "FULL" or (policy == "FINAL" and k == 19):
                    zb = -sd * (np.sign(ma) * np.sign(mb)) * z
                else:
                    zb = -sd * z
                noiseA.append(za)
                noiseB.append(zb)
            for target, noise, v in ((A, noiseA, pre[-1][3]), (B, noiseB, pre[-1][6])):
                out = advance_rbgbmc_particles(
                    x,
                    m,
                    0.0,
                    0.1,
                    2.0,
                    0.005,
                    20,
                    np.random.default_rng(9),
                    rng_brownian=PrescribedNoise(noise),
                    conditional_mean_transport=True,
                )
                np.testing.assert_array_equal(out["x"], target[0])
                np.testing.assert_array_equal(out["m"], target[1])
                np.testing.assert_array_equal(out["v"], v)
                grid = np.linspace(-1, 1, 101)
                np.testing.assert_array_equal(
                    reconstruct_cumulative_field(*target, ul, grid),
                    reconstruct_cumulative_field(out["x"], out["m"], 0.0, grid),
                )
        self.assertTrue(any(np.any(pre[k][2] != pre[k][5]) for k in range(20)))

    def test_one_sign_identity_and_instrumentation_is_read_only(self):
        initial = (np.linspace(-1, 1, 20), np.full(20, -0.1), 1.0)

        def mutate(*s):
            for a in s[1:]:
                a[:] = 999

        for fp in (lambda u: u, lambda u: np.zeros_like(u), lambda u: u * u):
            a = advance_pair(
                initial, 0.1, 0.005, 20, np.random.default_rng(2), "RAW", fp
            )
            b = advance_pair(
                initial,
                0.1,
                0.005,
                20,
                np.random.default_rng(2),
                "FULL",
                fp,
                pre_diffusion=mutate,
            )
            for i in (0, 1):
                for j in (0, 1):
                    np.testing.assert_array_equal(a[i][j], b[i][j])

    def test_a_replica_is_policy_independent(self):
        initial = (np.linspace(-0.1, 0.1, 12), np.tile([0.05, -0.05], 6), 0.0)
        baseline = advance_pair(
            initial, 0.1, 0.005, 20, np.random.default_rng(81), "RAW", lambda u: u
        )
        for p in ("FULL", "WITHIN", "FINAL"):
            other = advance_pair(
                initial, 0.1, 0.005, 20, np.random.default_rng(81), p, lambda u: u
            )
            np.testing.assert_array_equal(other[0][0], baseline[0][0])
            np.testing.assert_array_equal(other[0][1], baseline[0][1])

    def test_finalstage_identity_by_independent_indicator_quadrature(self):
        # Unequal magnitudes, unlike pairs, and separated particles.
        a = np.array([-0.6, -0.15, 0.05, 0.7])
        b = np.array([-0.5, -0.2, 0.25, 0.9])
        ma = np.array([0.3, -0.1, -0.4, 0.2])
        mb = np.array([-0.2, 0.4, 0.1, -0.3])
        sd = 0.18

        def delta(x):
            pa = ndtr((x - a) / sd)
            pb = ndtr((x - b) / sd)
            # Conditional pair variance differs only in matched cross-covariances.
            jr = np.maximum(pa + pb - 1, 0)
            js = np.where(ma * mb < 0, np.minimum(pa, pb), jr)
            return 0.5 * np.sum(ma * mb * (jr - js))

        numeric = quad(delta, -4, 4, epsabs=1e-12, points=list(a) + list(b), limit=200)[
            0
        ]
        d = a - b
        s = 2 * sd
        g = s * np.sqrt(2 / np.pi) * np.exp(-d * d / (2 * s * s)) + d * (
            2 * ndtr(d / s) - 1
        )
        exact = 0.25 * np.sum(np.abs(ma * mb) * (g - np.abs(d)) * (ma * mb < 0))
        self.assertAlmostEqual(exact, numeric, places=10)
        self.assertGreater(exact, 0)

    def test_incompatible_compensation_options_are_rejected(self):
        for option in ("conditional_mean_transport", "redraw_after_diffusion"):
            with self.assertRaises(ValueError):
                advance_rbgbmc_particles(
                    np.array([-0.1, 0.1]),
                    np.array([0.5, -0.5]),
                    0.0,
                    0.1,
                    2.0,
                    0.005,
                    2,
                    np.random.default_rng(1),
                    compensate_transport_variance=True,
                    **{option: True}
                )

    def test_invalid_inputs_are_rejected(self):
        init = (np.arange(4.0), np.array([1.0, -1.0, 1.0, -1.0]), 0.0)
        for p in ("typo", ""):
            with self.assertRaises(ValueError):
                advance_pair(
                    init, 0.1, 0.01, 3, np.random.default_rng(0), p, lambda u: u
                )
        with self.assertRaises(ValueError):
            advance_pair(
                (np.arange(4.0), np.zeros(4), 0.0),
                0.1,
                0.01,
                3,
                np.random.default_rng(0),
                "FULL",
                lambda u: u,
            )


if __name__ == "__main__":
    unittest.main()
