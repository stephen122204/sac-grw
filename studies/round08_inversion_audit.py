import sys, numpy as np
sys.path.insert(0,'.')
from studies import twopulse_reference as TP

def run(N, h, nu=0.1, T=1.0, seed=7):
    """Reproduces study_round07_regime.probe trajectories EXACTLY, and scores
    both the original (array-order) and corrected (spatial-order) diagnostics."""
    xi, mi, ul, _ = TP.initialize(N)
    K = round(T/h)
    rng = np.random.default_rng(seed)
    o = np.argsort(xi, kind='stable')
    x, m = np.asarray(xi,float)[o].copy(), np.asarray(mi,float)[o].copy()
    v = ul + np.cumsum(m)
    sd = np.sqrt(2.0*nu*h)
    orig, corr, vjump_err = [], [], []
    for _ in range(K):
        x_pre = x.copy()
        x = x + v*h
        # ---- ORIGINAL diagnostic: adjacent ARRAY entries, one direction only
        db, da = np.diff(x_pre), np.diff(x)
        orig.append(float(np.mean((db > 0) & (da < 0))))
        # ---- CORRECTED: adjacent CURRENT SPATIAL neighbours, both directions
        s = np.argsort(x_pre, kind='stable')          # neighbour indices only
        a, b = s[:-1], s[1:]                          # spatial neighbours
        corr.append(float(np.mean(x[a] > x[b])))      # reversed by transport
        # ---- velocity-jump identity check for CURRENT spatial neighbours
        vjump_err.append(float(np.max(np.abs((v[b]-v[a]) - m[b]))))
        o = np.argsort(x, kind='stable'); x, m = x[o], m[o]
        v = ul + np.cumsum(m)
        x = x + sd*rng.standard_normal(len(x))
    return np.mean(orig), np.mean(corr), np.max(vjump_err)

print(f"{'N':>6} {'h':>8} {'original (array order)':>24} {'corrected (spatial)':>21} {'Codex':>9} {'max |v-jump - m_next|':>22}")
codex = {(1600,0.005):0.05562,(1600,0.0025):0.02183,(6400,0.005):0.14138,(6400,0.0025):0.07106}
for N in (1600,6400):
    for h in (0.005,0.0025):
        a,b,e = run(N,h)
        print(f"{N:>6} {h:>8g} {a:24.4e} {b:21.5f} {codex[(N,h)]:9.5f} {e:22.4f}")
