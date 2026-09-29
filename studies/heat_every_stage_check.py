"""Every-stage variance decomposition for the heat equation, flux f = 0 (Section 3.4, Proposition 3.10, Appendix A.6).

For an admissible coupling, the conditional mean of each terminal field given the
first k+1 draws is u_- + sum_i m_i Phi((x - X_i(t_{k+1}))/s_k), s_k = sigma sqrt(K-k-1),
so Var Ubar(x) = sum_k E V_k(x) with V_k the conditional variance of the stage-k
increment of the martingale E[Ubar(x) | F_k]. Integrated over the line,

  int V_k dx = (1/4) [ (sum mA^2 + sum mB^2) c_k
                       + sum_i mA_i mB_i ( g_{rho_k}(d_i) - g_{tau_i}(d_i) ) ],

  c_k = (sqrt(sigma^2 + s_k^2) - s_k)/sqrt(pi),  rho_k = sqrt(2 sigma^2 + 2 s_k^2),
  tau_i = sqrt(4 sigma^2 + 2 s_k^2) for a reflected pair, sqrt(2) s_k for a synchronized pair,

and the stage-k gain of the sign rule over reflection from a common state is
Gamma_k = (1/4) sum_{unlike} |mA mB| [g_{sqrt(4 sigma^2+2 s_k^2)}(d) - g_{sqrt(2) s_k}(d)].

(a)  checks the conditional-mean formula directly by continuing one paired state many
     times, and checks Var Ubar = sum_k E V_k by comparing the empirical whole-line
     variance of the terminal pair mean with the closed-form stage terms summed along the
     same trajectories, for sign-switched and for reflected pairing;
(b)  checks Gamma_k at k = K/2 against a direct paired Monte Carlo over the stage-k
     increments, with the conditional-mean field evaluated on a grid, and against
     Gauss-Hermite quadrature of V_k;
(c)  reports the per-stage gains Gamma_k along sign-switched trajectories and the share of
     the last stage in their sum.
Alternating signs on [-0.25, 0.25] as in Test 6, nu = 0.1, T = 1, h = 0.005.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from coupled_gradient_particles import advance_pair
from gradient_particles import reconstruct_cumulative_field
from studies.final_stage_identity_checks import g as g_reference
from studies.sign_structure_map import config

OUT = ROOT / 'output/heat_every_stage'
NU, T, H = 0.1, 1.0, 0.005
K = round(T / H)
SD = np.sqrt(2.0 * NU * H)
N = 128
PAIRS = 8000
XG = np.linspace(-4.0, 4.0, 4001)          # whole-line window for the terminal fields
CONT = 4000                                # continuations for the conditional-mean check
MC_BATCHES, MC_PER_BATCH, MC_GRID = 40, 1000, 241
GAIN_STATES = 6
CAPTURE = (K // 4, K // 2 - 10, K // 2, 3 * K // 4, K - 2)
HEAT = lambda u: np.zeros_like(u)          # f' = 0: transport does nothing, sorting remains


def trap_weights(x):
    w = np.full(len(x), float(x[1] - x[0]))
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def g_tau(d, tau):
    """E|d + tau Z| (eq:folded), vectorized in d and tau, with g_0(d) = |d|."""
    d = np.asarray(d, float)
    tau = np.broadcast_to(np.asarray(tau, float), d.shape)
    safe = np.where(tau > 0, tau, 1.0)
    r = d / safe
    val = 2.0 * safe * np.exp(-0.5 * r * r) / np.sqrt(2.0 * np.pi) + d * (2.0 * ndtr(r) - 1.0)
    return np.where(tau > 0, val, np.abs(d))


def scale_after(k):
    """Diffusion scale of the stages after stage k."""
    return SD * np.sqrt(K - np.asarray(k, float) - 1.0)


def stage_terms(k, a, mA, b, mB):
    """Closed-form whole-line stage terms at the states entering stages k.
    Arrays have shape (..., N) and k broadcasts against the leading axes.
    Returns int V_k under reflection, under the sign rule, and Gamma_k."""
    s = scale_after(k)[..., None]
    d = a - b
    prod = mA * mB
    rho = np.sqrt(2 * SD**2 + 2 * s**2)
    t_ref = np.sqrt(4 * SD**2 + 2 * s**2)
    t_syn = np.sqrt(2.0) * s
    c = (np.sqrt(SD**2 + s**2) - s) / np.sqrt(np.pi)
    base = 0.25 * np.sum((mA**2 + mB**2) * c, axis=-1)
    g_rho, g_ref, g_syn = g_tau(d, rho), g_tau(d, t_ref), g_tau(d, t_syn)
    v_ref = base + 0.25 * np.sum(prod * (g_rho - g_ref), axis=-1)
    g_rule = np.where(prod < 0, g_syn, g_ref)
    v_sw = base + 0.25 * np.sum(prod * (g_rho - g_rule), axis=-1)
    gamma = 0.25 * np.sum(np.where(prod < 0, np.abs(prod) * (g_ref - g_syn), 0.0), axis=-1)
    return v_ref, v_sw, gamma


def run_pairs(policy, key, n_pairs, capture_paths=0):
    """Paired heat runs with the paper's coupled solver. For each trajectory, record the
    terminal pair mean on XG and the closed-form stage terms at every stage."""
    x0, m0, ul = config('clustered', N)
    w = trap_weights(XG)
    ubar = np.empty((n_pairs, len(XG)), dtype=np.float32)   # exact: dyadic field values
    v_own = np.empty((n_pairs, K))
    gam = np.empty((n_pairs, K))
    captured = {}
    for r in range(n_pairs):
        states = np.empty((K, 4, N))

        def hook(k, xA, mA, vA, xB, mB, vB):
            states[k] = (xA, mA, xB, mB)
            if r < capture_paths and k in CAPTURE:
                captured[(r, k)] = (xA, mA, xB, mB)

        rng = np.random.default_rng(np.random.SeedSequence([key, r]))
        A, B, u = advance_pair((x0, m0, ul), NU, H, K, rng, policy, HEAT, pre_diffusion=hook)
        ubar[r] = 0.5 * (reconstruct_cumulative_field(A[0], A[1], u, XG)
                         + reconstruct_cumulative_field(B[0], B[1], u, XG))
        v_ref, v_sw, g_k = stage_terms(np.arange(K), *states.transpose(1, 0, 2))
        v_own[r] = v_sw if policy == 'FULL' else v_ref
        gam[r] = g_k
    return ubar, v_own, gam, captured, w


def decomposition_check(ubar, v_own, w):
    """Empirical whole-line variance of Ubar versus sum_k E[int V_k] on the same paths,
    with influence-function (delta-method) standard errors."""
    R = len(ubar)
    mu = ubar.mean(0, dtype=np.float64)
    y = np.empty(R)
    for c0 in range(0, R, 500):
        dev = ubar[c0:c0 + 500].astype(np.float64) - mu
        y[c0:c0 + 500] = (dev**2) @ w * R / (R - 1)   # mean of y: integrated sample variance
    q = v_own.sum(1)
    z = stats.norm.ppf(0.975)
    diff = y - q

    def summ(v):
        m, se = float(v.mean()), float(v.std(ddof=1) / np.sqrt(R))
        return dict(estimate=m, se=se, ci95=[m - z * se, m + z * se])

    return dict(empirical_integrated_variance=summ(y), sum_of_stage_terms=summ(q),
                empirical_minus_stage_sum=summ(diff),
                relative_difference=float(diff.mean() / q.mean()))


def continue_pairs(xA, mA, xB, mB, first_stage, rng, policy):
    """Vectorized continuation of paired heat states (rows) from the state entering
    first_stage, with the stage rule of Algorithm 2 (sort, choose multipliers, draw)."""
    for _ in range(first_stage, K):
        o = np.argsort(xA, axis=1, kind='stable')
        xA, mA = np.take_along_axis(xA, o, 1), np.take_along_axis(mA, o, 1)
        o = np.argsort(xB, axis=1, kind='stable')
        xB, mB = np.take_along_axis(xB, o, 1), np.take_along_axis(mB, o, 1)
        eps = -np.sign(mA) * np.sign(mB) if policy == 'FULL' else -np.ones_like(mA)
        z = rng.standard_normal(xA.shape)
        xA = xA + SD * z
        xB = xB + SD * eps * z
    return xA, mA, xB, mB


def stepper_agreement():
    """The vectorized continuation reproduces advance_pair bit for bit from the start."""
    x0, m0, ul = config('clustered', N)
    worst = 0.0
    for policy in ('FULL', 'RAW'):
        A, B, _ = advance_pair((x0, m0, ul), NU, H, K, np.random.default_rng(77), policy, HEAT)
        xA, mA, xB, mB = continue_pairs(x0[None], m0[None], x0[None].copy(), m0[None].copy(),
                                        0, np.random.default_rng(77), policy)
        for u, v in ((A[0], xA[0]), (A[1], mA[0]), (B[0], xB[0]), (B[1], mB[0])):
            worst = max(worst, float(np.max(np.abs(u - v))))
    assert worst == 0.0, worst
    return worst


def conditional_mean_check(state, j, policy, key, earlier=None, j_earlier=None):
    """Direct check of E[U(x) | F_j] = u_- + sum_i m_i Phi((x - X_i)/s) for each
    simulation, from the paired state entering stage j (positions after stage j-1).
    Continuations are compared with the formula through projections on smooth bumps
    (Hotelling test) and pointwise. The same test applied to E[U(x) | F_j'] for an
    earlier stage j' shows that it separates conditioning on F_j from F_j'."""
    xA, mA, xB, mB = (np.tile(v, (CONT, 1)) for v in state)
    rng = np.random.default_rng(np.random.SeedSequence([key, j]))
    fA, gA, fB, gB = continue_pairs(xA, mA, xB, mB, j, rng, policy)
    w = trap_weights(XG)
    bumps = np.exp(-0.5 * ((XG[None, :] - np.linspace(-1.5, 1.5, 7)[:, None]) / 0.3)**2) * w
    J = len(bumps)

    def phi_sum(x_now, m_now, jj):
        return np.sum(m_now[:, None] * ndtr((XG[None, :] - x_now[:, None])
                                            / (SD * np.sqrt(K - jj))), axis=0)

    def hotelling(T, target):
        diff = T.mean(0) - target
        t2 = CONT * diff @ np.linalg.solve(np.cov(T, rowvar=False), diff)
        return float(stats.f.sf((CONT - J) / (J * (CONT - 1)) * t2, J, CONT - J))

    out = {}
    for name, x_end, m_end, col in (('A', fA, gA, 0), ('B', fB, gB, 2)):
        U = np.array([reconstruct_cumulative_field(x_end[r], m_end[r], 0.0, XG)
                      for r in range(CONT)])
        pred = phi_sum(state[col], state[col + 1], j)
        T = U @ bumps.T
        mean, se = U.mean(0), U.std(0, ddof=1) / np.sqrt(CONT)
        live = se > 0
        rec = dict(hotelling_p=hotelling(T, bumps @ pred),
                   max_abs_error=float(np.max(np.abs(mean - pred))),
                   max_abs_z=float(np.max(np.abs(mean - pred)[live] / se[live])))
        if earlier is not None:
            rec['hotelling_p_for_earlier_conditioning'] = hotelling(
                T, bumps @ phi_sum(earlier[col], earlier[col + 1], j_earlier))
        out[name] = rec
    return dict(policy=policy, stage_entered=j, remaining_stages=K - j,
                scale=float(SD * np.sqrt(K - j)), continuations=CONT, projections=J,
                earlier_stage=j_earlier, **out)


def gh_integrated_V(a, mA, b, mB, eps, s, xg):
    """Whole-line V_k by Gauss-Hermite in each pair's increment, pair decomposition (ii)."""
    z, wz = np.polynomial.hermite_e.hermegauss(120)
    wz = wz / wz.sum()
    tot = np.zeros(len(xg))
    for i in range(len(a)):
        f = 0.5 * (mA[i] * ndtr((xg[None, :] - a[i] - SD * z[:, None]) / s)
                   + mB[i] * ndtr((xg[None, :] - b[i] - SD * eps[i] * z[:, None]) / s))
        mu = wz @ f
        tot += wz @ (f * f) - mu * mu
    return float(trap_weights(xg) @ tot)


def window(a, b, s, pts):
    sc = np.sqrt(SD**2 + s**2)
    return np.linspace(min(a.min(), b.min()) - 8 * sc, max(a.max(), b.max()) + 8 * sc, pts)


def stage_gain_mc(state, k, key):
    """Paired Monte Carlo of int [V_k^reflect - V_k^switch] dx from one state, using the
    same stage-k draws for both rules and the conditional-mean field M_{k+1} on a grid."""
    a, mA, b, mB = state
    s = float(scale_after(k))
    xg = window(a, b, s, MC_GRID)
    w = trap_weights(xg)
    eps = -np.sign(mA) * np.sign(mB)
    rng = np.random.default_rng(np.random.SeedSequence([key, k]))

    def field(pos, m):
        return 0.5 * np.einsum('rng,n->rg', ndtr((xg[None, None, :] - pos[:, :, None]) / s), m)

    per_batch, chunk = [], 100
    for _ in range(MC_BATCHES):
        MR = np.empty((MC_PER_BATCH, len(xg)))
        MS = np.empty_like(MR)
        for c0 in range(0, MC_PER_BATCH, chunk):
            z = rng.standard_normal((chunk, len(a)))
            fa = field(a + SD * z, mA)
            MR[c0:c0 + chunk] = fa + field(b - SD * z, mB)
            MS[c0:c0 + chunk] = fa + field(b + SD * eps * z, mB)
        per_batch.append(w @ MR.var(0, ddof=1) - w @ MS.var(0, ddof=1))
    v = np.array(per_batch)
    m, se = float(v.mean()), float(v.std(ddof=1) / np.sqrt(MC_BATCHES))
    t = stats.t.ppf(0.975, MC_BATCHES - 1)
    return m, se, [m - t * se, m + t * se]


def pairwise_extremality(state, k, n_pairs=8):
    """(iii) at an intermediate stage: over Gaussian couplings with correlation rho in
    [-1, 1], the pair contribution to V_k(x) is smallest at rho = eps (sign rule) at
    every x. Returns the largest excess of the sign rule over the minimum over rho."""
    a, mA, b, mB = state
    s = float(scale_after(k))
    z, wz = np.polynomial.hermite_e.hermegauss(60)
    wz = wz / wz.sum()
    rhos = np.linspace(-1.0, 1.0, 41)
    worst, checked = 0.0, 0
    sep = np.abs(a - b)
    like = np.flatnonzero(mA * mB > 0)
    unlike = np.flatnonzero(mA * mB < 0)
    chosen = np.concatenate([like[np.argsort(sep[like])][:n_pairs // 2],
                             unlike[np.argsort(sep[unlike])][:n_pairs // 2]])
    for i in chosen:
        xg = np.linspace(min(a[i], b[i]) - 6 * s, max(a[i], b[i]) + 6 * s, 301)
        pa = ndtr((xg[None, :] - a[i] - SD * z[:, None]) / s)      # (node, x)
        va = wz @ pa**2 - (wz @ pa)**2
        contrib = []
        for rho in rhos:
            zb = rho * z[:, None] + np.sqrt(max(1 - rho * rho, 0.0)) * z[None, :]
            pb = ndtr((xg[None, None, :] - b[i] - SD * zb[:, :, None]) / s)
            W = wz[:, None] * wz[None, :]
            eb = np.einsum('ij,ijx->x', W, pb)
            vb = np.einsum('ij,ijx->x', W, pb**2) - eb**2
            cov = np.einsum('ij,ix,ijx->x', W, pa, pb) - (wz @ pa) * eb
            contrib.append(0.25 * (mA[i]**2 * va + mB[i]**2 * vb + 2 * mA[i] * mB[i] * cov))
        contrib = np.array(contrib)
        rule = contrib[0] if mA[i] * mB[i] > 0 else contrib[-1]
        worst = max(worst, float(np.max(rule - contrib.min(0))))
        checked += 1
    return dict(stage=k, pairs_checked=checked, like_pairs=int(min(len(like), n_pairs // 2)),
                unlike_pairs=int(min(len(unlike), n_pairs // 2)), rho_grid=len(rhos),
                max_excess_of_sign_rule_over_min=worst,
                scale_of_contribution=float(0.25 * 2 * np.max(np.abs(mA * mB))))


def ratio_ci(num, den):
    """Ratio of means with a delta-method interval, per-path numerator and denominator."""
    R = len(num)
    rat = num.mean() / den.mean()
    infl = (num - rat * den) / den.mean()
    se = infl.std(ddof=1) / np.sqrt(R)
    z = stats.norm.ppf(0.975)
    return dict(estimate=float(rat), se=float(se), ci95=[float(rat - z * se), float(rat + z * se)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    d = np.linspace(-0.5, 0.5, 101)
    g_err = max(float(np.max(np.abs(g_tau(d, tau) - g_reference(d, tau))))
                for tau in (0.0, SD, 2 * SD, 0.3))
    assert g_err < 1e-12, g_err
    agree = stepper_agreement()

    print(f"Heat equation, alternating signs, N={N}, nu={NU}, T={T}, h={H}, K={K}, "
          f"{PAIRS} pairs per coupling\n")
    runs = {}
    for policy, key in (('FULL', 7101), ('RAW', 7102)):
        runs[policy] = run_pairs(policy, key, PAIRS, capture_paths=GAIN_STATES)
        print(f"   {policy}: {PAIRS} pairs done  ({time.perf_counter() - t0:.0f}s)")

    # (a) conditional mean and decomposition
    capS = runs['FULL'][3]
    j = K // 2
    je = j - 10
    cond = [conditional_mean_check(capS[(0, j)], j, 'FULL', 7201, capS[(0, je)], je),
            conditional_mean_check(capS[(1, j)], j, 'RAW', 7202, capS[(1, je)], je)]
    decomp = {}
    for policy, name in (('FULL', 'switch'), ('RAW', 'reflect')):
        ubar, v_own, _, _, w = runs[policy]
        decomp[name] = decomposition_check(ubar, v_own, w)
    print(f"\n(a) conditional mean from a state entering stage {j} ({CONT} continuations, "
          f"Hotelling p on 7 projections; p for E[U | F_{je}] in parentheses):")
    for c in cond:
        print(f"   continued by {c['policy']:>4}: A p={c['A']['hotelling_p']:.3f} "
              f"max|z|={c['A']['max_abs_z']:.2f} "
              f"({c['A']['hotelling_p_for_earlier_conditioning']:.1e})   "
              f"B p={c['B']['hotelling_p']:.3f} max|z|={c['B']['max_abs_z']:.2f} "
              f"({c['B']['hotelling_p_for_earlier_conditioning']:.1e})")
    print("    decomposition Var Ubar = sum_k E V_k (whole line):")
    for name, r in decomp.items():
        e, q, df = (r['empirical_integrated_variance'], r['sum_of_stage_terms'],
                    r['empirical_minus_stage_sum'])
        print(f"   {name:>8}: empirical {e['estimate']:.5e} [{e['ci95'][0]:.5e}, {e['ci95'][1]:.5e}]"
              f"   stage sum {q['estimate']:.5e} [{q['ci95'][0]:.5e}, {q['ci95'][1]:.5e}]"
              f"   difference CI [{df['ci95'][0]:.3e}, {df['ci95'][1]:.3e}]")

    # (b) the stage gain at k = K/2
    kb = K // 2
    gains = []
    print(f"\n(b) Gamma_k at k={kb} from {GAIN_STATES} switched states "
          f"({MC_BATCHES}x{MC_PER_BATCH} paired draws):")
    for r in range(GAIN_STATES):
        a, mA, b, mB = capS[(r, kb)]
        _, _, cf = stage_terms(np.array(kb), a, mA, b, mB)
        cf = float(cf)
        s = float(scale_after(kb))
        eps = -np.sign(mA) * np.sign(mB)
        quad = {}
        for pts in (MC_GRID, 2001):
            xg = window(a, b, s, pts)
            quad[pts] = (gh_integrated_V(a, mA, b, mB, -np.ones(N), s, xg)
                         - gh_integrated_V(a, mA, b, mB, eps, s, xg))
        m, se, ci = stage_gain_mc(capS[(r, kb)], kb, 7300 + r)
        ok = ci[0] <= cf <= ci[1]
        unlike = int(np.sum(mA * mB < 0))
        gains.append(dict(path=r, stage=kb, unlike_pairs=unlike, closed_form=cf,
                          quadrature_mc_grid=quad[MC_GRID], quadrature_fine_grid=quad[2001],
                          mc=m, mc_se=se, mc_ci95=ci, z=(m - cf) / se, agrees=bool(ok)))
        print(f"   path {r}: unlike {unlike:3d}/{N}  closed form {cf:.6e}  quadrature "
              f"{quad[2001]:.6e}  MC {m:.6e} [{ci[0]:.6e}, {ci[1]:.6e}]  z={(m - cf) / se:+.2f}")
    other_stages = []
    for r in range(GAIN_STATES):
        for kk in CAPTURE:
            if kk == kb:
                continue
            a, mA, b, mB = capS[(r, kk)]
            s = float(scale_after(kk))
            eps = -np.sign(mA) * np.sign(mB)
            xg = window(a, b, s, 2001)
            vq_ref = gh_integrated_V(a, mA, b, mB, -np.ones(N), s, xg)
            vq_sw = gh_integrated_V(a, mA, b, mB, eps, s, xg)
            v_ref, v_sw, cf = (float(v) for v in stage_terms(np.array(kk), a, mA, b, mB))
            other_stages.append(dict(path=r, stage=kk, closed_form_gamma=cf,
                                     quadrature_gamma=vq_ref - vq_sw,
                                     rel_err_gamma=abs(vq_ref - vq_sw - cf) / cf,
                                     rel_err_intV_reflect=abs(vq_ref - v_ref) / v_ref,
                                     rel_err_intV_switch=abs(vq_sw - v_sw) / v_sw))
    worst_rel = max(max(o['rel_err_gamma'], o['rel_err_intV_reflect'], o['rel_err_intV_switch'])
                    for o in other_stages)
    extremal = pairwise_extremality(capS[(0, kb)], kb)
    print(f"   quadrature vs closed form at k in {CAPTURE}: max relative error {worst_rel:.1e}")
    print(f"   (iii) sign rule minus min over Gaussian correlations: "
          f"{extremal['max_excess_of_sign_rule_over_min']:.1e}")

    # (c) per-stage gains along switched trajectories
    gS = runs['FULL'][2]
    gR = runs['RAW'][2]
    mean_k = gS.mean(0)
    se_k = gS.std(0, ddof=1) / np.sqrt(PAIRS)
    total = gS.sum(1)
    last_share = ratio_ci(gS[:, -1], total)
    last10_share = ratio_ci(gS[:, -10:].sum(1), total)
    first_half_share = ratio_ci(gS[:, :K // 2].sum(1), total)
    qS, qR = runs['FULL'][1].sum(1), runs['RAW'][1].sum(1)
    red = float(qR.mean() - qS.mean())
    red_se = float(np.sqrt(qR.var(ddof=1) / PAIRS + qS.var(ddof=1) / PAIRS))
    zc = stats.norm.ppf(0.975)
    tot_m, tot_se = float(total.mean()), float(total.std(ddof=1) / np.sqrt(PAIRS))
    print(f"\n(c) per-stage gains Gamma_k along switched trajectories:")
    print(f"   sum_k E Gamma_k = {tot_m:.5e} [{tot_m - zc * tot_se:.5e}, {tot_m + zc * tot_se:.5e}]")
    print(f"   last-stage share {last_share['estimate']:.4f} {np.round(last_share['ci95'], 4).tolist()}, "
          f"last 10 stages {last10_share['estimate']:.4f}, first half {first_half_share['estimate']:.4f}")
    print(f"   peak mean gain at stage {int(np.argmax(mean_k))}")
    print(f"   V_reflect - V_switch (stage sums, independent runs) = {red:.5e} "
          f"[{red - zc * red_se:.5e}, {red + zc * red_se:.5e}]")
    print(f"   V_switch / V_reflect = {qS.mean() / qR.mean():.4f}")
    print(f"   switching only at the last stage after reflection gains E_reflect Gamma_(K-1) = "
          f"{gR[:, -1].mean():.4e}, share {gR[:, -1].mean() / red:.4f} of V_reflect - V_switch")

    res = dict(
        setup=dict(configuration='alternating signs on [-0.25, 0.25], |m| = 1/N (Test 6)',
                   flux='zero (heat equation)', N=N, nu=NU, T=T, h=H, K=K, sigma=float(SD),
                   pairs_per_coupling=PAIRS, window=[float(XG[0]), float(XG[-1])],
                   window_points=len(XG), g_tau_max_deviation_from_paper_g=g_err,
                   vectorized_continuation_matches_advance_pair=agree == 0.0),
        a_conditional_mean=cond,
        a_decomposition=decomp,
        b_stage_gain=dict(stage=kb, scale=float(scale_after(kb)), batches=MC_BATCHES,
                          draws_per_batch=MC_PER_BATCH, grid_points=MC_GRID, states=gains,
                          agree=f"{sum(gn['agrees'] for gn in gains)}/{len(gains)}",
                          quadrature_other_stages=other_stages,
                          quadrature_max_relative_error=worst_rel,
                          pairwise_extremality=extremal),
        c_stage_gains=dict(
            mean_gamma_by_stage=mean_k.tolist(), se_gamma_by_stage=se_k.tolist(),
            sum_mean_gamma=dict(estimate=tot_m, se=tot_se,
                                ci95=[tot_m - zc * tot_se, tot_m + zc * tot_se]),
            last_stage_share=last_share, last_10_stages_share=last10_share,
            first_half_share=first_half_share, peak_stage=int(np.argmax(mean_k)),
            mean_gamma_at=dict((str(k), float(mean_k[k])) for k in (0, 1, 10, 50, 100, 150,
                                                                     190, 198, 199)),
            V_reflect_stage_sum=float(qR.mean()), V_switch_stage_sum=float(qS.mean()),
            V_reflect_minus_V_switch=dict(estimate=red, se=red_se,
                                          ci95=[red - zc * red_se, red + zc * red_se]),
            V_switch_over_V_reflect=float(qS.mean() / qR.mean()),
            greedy_sum_over_total_reduction=tot_m / red,
            greedy_sum_along_reflected=float(gR.sum(1).mean()),
            final_only_gain_after_reflection=float(gR[:, -1].mean()),
            final_only_share_of_total_reduction=float(gR[:, -1].mean() / red)),
        note='whole-line quantities; V_k closed forms from eq:covdist with the Gaussian '
             'smoothing of the later stages; intervals are nominal 95%: delta-method for '
             'variances and ratios over independent trajectories, t over batches in (b)',
        seconds=time.perf_counter() - t0)
    json.dump(res, open(OUT / 'heat.json', 'w'), indent=1)
    print(f"\n   {time.perf_counter() - t0:.0f}s   saved -> {OUT / 'heat.json'}")


if __name__ == '__main__':
    main()
