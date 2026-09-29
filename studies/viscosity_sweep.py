"""Viscosity sweep of the three pairings on the two-pulse Burgers problem."""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gradient_particles import reconstruct_cumulative_field
from studies.cubic_flux_transfer import advance_pair_flux, XG
from studies.spectral_reference import solve, evaluate
from studies.pilot_prediction_transfer import init_from
from studies.profile_time_flux_sweep import ARM, u0_two

OUT = ROOT / 'output/viscosity_sweep'
T, H, N, REPS = 1.0, 0.005, 2048, 48
NUS = (0.02, 0.05, 0.1, 0.2, 0.5)
KEY, BOOT_SEED, BOOT = 2700, 2701, 3000


def batch(arm, init, nu, key):
    K = round(T / H); x0, m0, ul = init; P = []
    for r in range(REPS):
        s = np.random.default_rng(np.random.SeedSequence([key, ARM[arm], r]))
        A, B, u = advance_pair_flux((x0, m0, ul), nu, H, K, s, arm, 'burgers')
        P.append(0.5 * (reconstruct_cumulative_field(A[0], A[1], u, XG)
                        + reconstruct_cumulative_field(B[0], B[1], u, XG)))
    return np.array(P)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dx = float(XG[1] - XG[0]); rng = np.random.default_rng(BOOT_SEED)
    t0 = time.perf_counter(); init = init_from(u0_two, N)
    print("PREDECLARED PREDICTION: FULL/WITHIN < 1, interval below 1, at every nu.\n")
    print(f"   {'nu':>5} {'V_RAW':>10} {'V_WITHIN':>10} {'V_FULL':>10} "
          f"{'FULL/WITHIN':>22} {'FULL/RAW':>22} {'WITHIN/RAW':>22} "
          f"{'bias L2':>9} {'floor':>8} {'s':>5}")
    rows = []
    for nu in NUS:
        tc = time.perf_counter()
        _, _, uh, kk = solve('burgers', T, nu, M=2048, dt=1e-4, u0=u0_two)
        ref = evaluate(uh, kk, XG, 2048)
        V, secs = {}, {}
        for a in ARM:
            ta = time.perf_counter(); V[a] = batch(a, init, nu, KEY)
            secs[a] = time.perf_counter() - ta
        iv = {a: float(dx * np.sum(V[a].var(0, ddof=1))) for a in V}
        bv = {a: np.array([dx * np.sum(V[a][i].var(0, ddof=1))
                           for i in rng.integers(0, REPS, (BOOT, REPS))])
              for a in ('FULL', 'WITHIN', 'RAW')}
        ci = lambda num, den: [float(q) for q in
                               np.percentile(bv[num] / bv[den], [2.5, 97.5])]
        cFW, cFR, cWR = ci('FULL', 'WITHIN'), ci('FULL', 'RAW'), ci('WITHIN', 'RAW')
        # Each replica keeps the single-simulation law, so the three arms share
        # one mean; the pooled mean measures the time-step bias at this nu.
        ubar = np.mean([V[a].mean(0) for a in ARM], axis=0)
        floor = float(np.sqrt(dx * np.sum(sum(V[a].var(0, ddof=1) for a in ARM)
                                          / (9 * REPS))))
        bias = float(np.sqrt(dx * np.sum((ubar - ref) ** 2)))
        sc = time.perf_counter() - tc
        print(f"   {nu:>5.2f} {iv['RAW']:10.3e} {iv['WITHIN']:10.3e} {iv['FULL']:10.3e} "
              f"{iv['FULL']/iv['WITHIN']:7.4f} [{cFW[0]:.3f},{cFW[1]:.3f}] "
              f"{iv['FULL']/iv['RAW']:7.4f} [{cFR[0]:.3f},{cFR[1]:.3f}] "
              f"{iv['WITHIN']/iv['RAW']:7.4f} [{cWR[0]:.3f},{cWR[1]:.3f}] "
              f"{bias:9.2e} {floor:8.2e} {sc:5.0f}")
        rows.append(dict(nu=nu, V_raw=iv['RAW'], V_within=iv['WITHIN'], V_full=iv['FULL'],
                         within_over_raw=iv['WITHIN'] / iv['RAW'],
                         full_over_raw=iv['FULL'] / iv['RAW'],
                         full_over_within=iv['FULL'] / iv['WITHIN'],
                         ci=cFW, ci_full_over_raw=cFR, ci_within_over_raw=cWR,
                         diffusion_step_sd=float(np.sqrt(2 * nu * H)),
                         ref_max_slope=float(np.max(np.abs(np.gradient(ref, XG)))),
                         bias_L2=bias, bias_noise_floor_L2=floor,
                         ref_L2=float(np.sqrt(dx * np.sum(ref ** 2))),
                         seconds=sc, seconds_per_arm=secs))
    total = time.perf_counter() - t0
    json.dump(dict(purpose='dependence of the variance reduction on the viscosity nu '
                           'for the two-pulse Burgers problem at T=1',
                   profile='two-pulse, u0 = 0.8 exp(-(x+0.7)^2/(2*0.5^2)) '
                           '- 0.5 exp(-(x-0.4)^2/(2*0.3^2))',
                   flux='burgers', init='init_from (midpoint signed quantiles, N/2 per sign)',
                   N=N, h=H, T=T, nus=list(NUS), reps=REPS, arms=ARM,
                   grid=dict(lo=float(XG[0]), hi=float(XG[-1]), n=len(XG)),
                   seed_scheme=f'np.random.SeedSequence([{KEY}, ARM[arm], rep])',
                   bootstrap=dict(resamples=BOOT, seed=BOOT_SEED, method='percentile 95%, '
                                  'reps resampled independently per arm'),
                   reference='Fourier pseudospectral, M=2048, dt=1e-4, [-16,16] periodic',
                   prediction='FULL/WITHIN < 1 with interval below 1 at every nu',
                   total_seconds=total, rows=rows),
              open(OUT / 'viscosity.json', 'w'), indent=1)
    print(f"\n   {total:.0f}s   saved -> {OUT}")


if __name__ == '__main__':
    main()
