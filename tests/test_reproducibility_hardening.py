"""Reproducibility-hardening tests added after the pre-submission audit.

Covers the invariants the audit found untested: exact signed-mass conservation
on stepper output, strict (non-falling-back) manuscript fits, value-keyed
multi-viscosity seed identity, resume fingerprint refusal, subset-summary
completeness, per-cell row validation, and completeness of the archived
production realization profiles (including exact replay of the published
spread-slope bootstrap interval).
"""
import contextlib
import csv
import io
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from relaxation_gbmc import advance_rbgbmc_particles, initialize_tanh_shock_particles
from studies import study_multiviscosity_sweep as mv
from studies.study_gbmc_production_n_refinement import _fit_tanh

T6_DIR = os.path.join(ROOT, 'output', 'final_prepublication_tests',
                      'gbmc_production_n_refinement')
MV_DIR = os.path.join(ROOT, 'output', 'final_prepublication_tests',
                      'gbmc_multiviscosity_sweep')


def _quiet_run_study(**kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return mv.run_study(**kwargs)


def _fits(out_dir):
    with open(os.path.join(out_dir, 'per_run.csv')) as handle:
        return {(float(r['nu']), r['arm'], int(r['seed_idx'])): float(r['nu_fit'])
                for r in csv.DictReader(handle)}


# --------------------------------------------------------------------------- #
# Exact stepper invariants
# --------------------------------------------------------------------------- #

def test_stepper_output_conserves_signed_mass_exactly():
    """sum(m) and the sorted mass multiset are unchanged by a full run."""
    x0, m0, u_left = initialize_tanh_shock_particles(500, nu=0.5, amplitude=1.0,
                                                     center=2.0)
    run = advance_rbgbmc_particles(x0.copy(), m0.copy(), u_left, 0.5, 2.0,
                                   0.005, 50, np.random.default_rng(7))
    assert abs(float(run['m'].sum()) - (-2.0)) < 1e-12
    np.testing.assert_array_equal(np.sort(run['m']), np.sort(m0))


# --------------------------------------------------------------------------- #
# Strict fitting: manuscript values can never come from the silent fallback
# --------------------------------------------------------------------------- #

def _fit_inputs():
    x = np.linspace(0.0, 4.0, 400)
    u = -np.tanh((x - 2.0) / (2.0 * 0.5))
    return x, u


def test_fit_records_curve_fit_method():
    x, u = _fit_inputs()
    diag = {}
    xc, nu, A = _fit_tanh(x, u, 1.0, 2.0, 0.5, method_out=diag)
    assert diag['method'] == 'curve_fit'
    assert abs(nu - 0.5) < 1e-6 and abs(xc - 2.0) < 1e-6


def test_fit_strict_raises_instead_of_falling_back(monkeypatch):
    import scipy.optimize

    def broken(*args, **kwargs):
        raise RuntimeError('synthetic curve_fit failure')

    monkeypatch.setattr(scipy.optimize, 'curve_fit', broken)
    x, u = _fit_inputs()
    with pytest.raises(RuntimeError, match='strict fitting is required'):
        _fit_tanh(x, u, 1.0, 2.0, 0.5, strict=True)
    # Non-strict path still falls back, and says so.
    diag = {}
    _fit_tanh(x, u, 1.0, 2.0, 0.5, strict=False, method_out=diag)
    assert diag['method'] == 'fallback'


# --------------------------------------------------------------------------- #
# Value-keyed multi-viscosity seed identity
# --------------------------------------------------------------------------- #

def test_multinu_streams_are_value_keyed_and_distinct():
    a1, b1 = mv._rngs(0.1, 3)
    a2, b2 = mv._rngs(0.1, 3)
    np.testing.assert_array_equal(a1.random(5), a2.random(5))
    np.testing.assert_array_equal(b1.normal(0, 1, 5), b2.normal(0, 1, 5))
    assert not np.array_equal(mv._rngs(0.1, 3)[0].random(3),
                              mv._rngs(0.05, 3)[0].random(3))
    assert not np.array_equal(mv._rngs(0.1, 3)[0].random(3),
                              mv._rngs(0.1, 4)[0].random(3))


def test_multinu_subset_and_reorder_reproduce_identical_cells(tmp_path):
    outA, outB, outC = (str(tmp_path / d) for d in ('A', 'B', 'C'))
    _quiet_run_study(S_override=1, out_base=outA, nu_seq=[0.1, 0.05])
    _quiet_run_study(S_override=1, out_base=outB, nu_seq=[0.05])
    _quiet_run_study(S_override=1, out_base=outC, nu_seq=[0.05, 0.1])
    fA, fB, fC = _fits(outA), _fits(outB), _fits(outC)
    assert all(fA[k] == fB[k] for k in fB)
    assert fA == fC


def test_multinu_incompatible_resume_is_refused(tmp_path):
    out = str(tmp_path / 'sweep')
    os.makedirs(out)
    wrong = mv._fingerprint(50)
    wrong['S'] = 49
    json.dump({'config': wrong, 'done': []},
              open(os.path.join(out, 'manifest.json'), 'w'))
    with pytest.raises(RuntimeError, match='Refusing to resume'):
        _quiet_run_study(S_override=50, out_base=out, nu_seq=[0.5])


def test_multinu_rejects_viscosity_outside_canonical_list(tmp_path):
    with pytest.raises(ValueError, match='not in the canonical study list'):
        _quiet_run_study(S_override=1, out_base=str(tmp_path / 'x'),
                         nu_seq=[0.3])


def test_multinu_subset_invocation_keeps_complete_summary(tmp_path):
    out = str(tmp_path / 'sweep')
    _quiet_run_study(S_override=1, out_base=out, nu_seq=[0.1, 0.05])
    before = json.load(open(os.path.join(out, 'summary.json')))
    _quiet_run_study(S_override=1, out_base=out, nu_seq=[0.05])
    after = json.load(open(os.path.join(out, 'summary.json')))
    assert after['metadata']['nu_seq'] == [0.1, 0.05]
    assert after['metadata']['requested_nu_seq'] == [0.05]
    assert len(after['summary']) == len(before['summary']) == 6
    assert after['summary'] == before['summary']
    assert after['paired_nu_contrasts'] == before['paired_nu_contrasts']


def test_multinu_resume_regenerates_cell_with_missing_rows(tmp_path):
    out = str(tmp_path / 'sweep')
    _quiet_run_study(S_override=1, out_base=out, nu_seq=[0.5])
    reference = _fits(out)
    per_run = os.path.join(out, 'per_run.csv')
    rows = list(csv.DictReader(open(per_run)))
    kept = [r for r in rows if r['arm'] != 'two_speed_a4']
    with open(per_run, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(kept)
    _quiet_run_study(S_override=1, out_base=out, nu_seq=[0.5])
    assert _fits(out) == reference


# --------------------------------------------------------------------------- #
# Archive completeness: production realization profiles
# --------------------------------------------------------------------------- #

def test_production_full_profile_archive_reproduces_table_and_interval():
    """Table 1 and the published spread-slope bootstrap interval must be
    recomputable from the archived realization-level profiles alone."""
    z = np.load(os.path.join(T6_DIR, 'production_profiles_full.npz'))
    rates = json.load(open(os.path.join(T6_DIR, 'rates.json')))
    per_N = {int(float(r['N'])): r
             for r in csv.DictReader(open(os.path.join(T6_DIR,
                                                       'per_N_summary.csv')))}
    assert list(z['seeds']) == list(range(42, 92))
    x, u_ref = z['x'], z['u_exact']
    dx = float(x[1] - x[0])
    N_seq = sorted(per_N)
    spreads = []
    for N in N_seq:
        u = z[f'N{N}']
        assert u.shape == (50, 400)
        u_mean = u.mean(axis=0)
        E_bias = float(np.sqrt(np.sum((u_mean - u_ref) ** 2 * dx)))
        E_spread = float(np.sqrt(np.mean(
            np.sum((u - u_mean[None, :]) ** 2 * dx, axis=1))))
        E_total = float(np.sqrt(np.mean(
            np.sum((u - u_ref[None, :]) ** 2 * dx, axis=1))))
        assert abs(E_bias - float(per_N[N]['E_bias'])) < 1e-12
        assert abs(E_spread - float(per_N[N]['E_spread'])) < 1e-12
        assert abs(E_total - float(per_N[N]['E_total'])) < 1e-12
        spreads.append(E_spread)
    # least-squares slope
    slope = float(np.polyfit(np.log10(np.array(N_seq, dtype=float)),
                             np.log10(np.array(spreads)), 1)[0])
    assert abs(slope - rates['spread_slope']) < 1e-12
    # exact replay of the realization-level bootstrap interval (rng 123,
    # 4000 replicates, resampling seeds within each N; L/N_OUT weighting as
    # in the study's _bootstrap_slope_ci)
    rng = np.random.default_rng(123)
    lx = np.log10(np.array(N_seq, dtype=float))
    boots = []
    for _ in range(4000):
        ly = []
        for N in N_seq:
            idx = rng.integers(0, 50, size=50)
            samp = z[f'N{N}'][idx]
            samp_mean = samp.mean(axis=0)
            spr = float(np.sqrt(np.mean(
                np.sum((samp - samp_mean[None, :]) ** 2 * (4.0 / 400), axis=1))))
            ly.append(np.log10(max(spr, 1e-15)))
        boots.append(float(np.polyfit(lx, np.array(ly), 1)[0]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    assert abs(lo - rates['spread_ci_lo']) < 1e-12
    assert abs(hi - rates['spread_ci_hi']) < 1e-12


def test_production_archive_has_no_fallback_fits():
    rows = list(csv.DictReader(open(os.path.join(T6_DIR, 'per_run.csv'))))
    assert len(rows) == 350
    assert all(r['fit_method'] == 'curve_fit' for r in rows)
    assert all(r['failed'].strip().lower() not in ('true', '1') for r in rows)


# --------------------------------------------------------------------------- #
# Multi-viscosity archive health
# --------------------------------------------------------------------------- #

def test_transient_archive_reference_and_health():
    """The transient archive must carry its reference, the documented
    tolerance, complete cells, exact zero total signed mass, and profiles that
    reproduce the summary decomposition."""
    base = os.path.join(ROOT, 'output', 'final_prepublication_tests',
                        'gbmc_smooth_transient')
    ref = np.load(os.path.join(base, 'reference.npz'))
    meta = json.loads(str(ref['meta']))
    assert meta['reference_tolerance'] < 1e-4
    assert meta['self_consistency_max_abs'] < 1e-10
    rows = list(csv.DictReader(open(os.path.join(base, 'per_run.csv'))))
    assert len(rows) == 450
    assert all(abs(float(r['mass_final'])) <= 1e-12 for r in rows)
    sj = json.load(open(os.path.join(base, 'summary.json')))
    x, u_ref = ref['x'], ref['u_ref']
    dx = float(x[1] - x[0])
    for s in sj['summary']:
        cell = os.path.join(base, f"cell_{s['arm']}_dt{s['dt']:g}"
                            .replace('.', 'p') + '.npz')
        prof = np.load(cell)['profiles']
        assert prof.shape == (50, 400)
        um = prof.mean(axis=0)
        E_bias = float(np.sqrt(np.sum((um - u_ref) ** 2 * dx)))
        assert abs(E_bias - s['E_bias']) < 1e-12
    assert all(p['resolves_sign'] for p in sj['paired_l2_contrasts'])


def test_multinu_archive_is_complete_and_healthy():
    rows = list(csv.DictReader(open(os.path.join(MV_DIR, 'per_run.csv'))))
    assert len(rows) == 750
    cells = {}
    for r in rows:
        cells.setdefault((float(r['nu']), r['arm']), []).append(int(r['seed_idx']))
    assert len(cells) == 15
    assert {nu for nu, _ in cells} == {0.5, 0.25, 0.1, 0.05, 0.025}
    for key, seeds in cells.items():
        assert sorted(seeds) == list(range(50)), key
    assert all(str(r['l2']).strip() not in ('', 'nan') for r in rows)
    assert sum(int(r['at_bound']) for r in rows) == 0
    for (nu, arm) in cells:
        path = os.path.join(MV_DIR, f'cell_{arm}_nu{nu:g}'.replace('.', 'p')
                            + '.npz')
        assert np.load(path)['profiles'].shape == (50, 400)
    manifest = json.load(open(os.path.join(MV_DIR, 'manifest.json')))
    assert manifest['config'] == mv._fingerprint(50)
    assert len(manifest['done']) == 15
