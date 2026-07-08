"""Produce the two must-fix corrected figures in figuresv3_regen/.

Reads the same CSVs the study scripts produce and writes _fixed.pdf next
to the frozen regens.

Fix 1: production_gbmc_bias_spread_total_vs_N_fixed.pdf
  Drop the misleading E_bias slope=-0.511 label; requalify it with
  R^2 = 0.94 and CI width [-0.65, -0.42], noting that the bias column
  is non-monotone.

Fix 2: heat_bias_spread_total_vs_N_fixed.pdf
  Retitle to headline the spread exponent (-0.468) rather than the
  total exponent (-0.457). Prose in §4.2 and the abstract use spread as
  the headline; the plot should match.
"""
import os, csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGEN = os.path.join(BASE, '..', 'regen_output')
OUT   = os.path.join(BASE, '..', 'figuresv3_regen')

# Fix 1: production_gbmc_bias_spread_total_vs_N
def slope_ci(x, y, n_boot=10000, seed=42):
    """Log-log slope, R^2, and bootstrap 95% CI."""
    lx, ly = np.log(x), np.log(y)
    s, b = np.polyfit(lx, ly, 1)
    R2 = 1 - np.sum((ly - (s*lx + b))**2) / np.sum((ly - ly.mean())**2)
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        while len(set(idx)) < 2:
            idx = rng.integers(0, len(x), size=len(x))
        slopes.append(np.polyfit(lx[idx], ly[idx], 1)[0])
    slopes = np.array(slopes)
    return float(s), float(R2), (float(np.quantile(slopes, 0.025)),
                                  float(np.quantile(slopes, 0.975)))


rows = list(csv.DictReader(open(f'{REGEN}/gbmc_production/per_N_summary.csv')))
N  = np.array([int(r['N']) for r in rows])
Eb = np.array([float(r['E_bias']) for r in rows])
Es = np.array([float(r['E_spread']) for r in rows])
Et = np.array([float(r['E_total']) for r in rows])

s_t, r2_t, ci_t = slope_ci(N, Et)
s_s, r2_s, ci_s = slope_ci(N, Es)
s_b, r2_b, ci_b = slope_ci(N, Eb)

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(N, Et, 's-', color='black',    lw=2, ms=7,
          label=f'$E_\\mathrm{{total}}$  slope $={s_t:.3f}$, $R^2 = {r2_t:.3f}$')
ax.loglog(N, Es, 'o-', color='steelblue', lw=2, ms=7,
          label=f'$E_\\mathrm{{spread}}$ slope $={s_s:.3f}$, $R^2 = {r2_s:.3f}$')
ax.loglog(N, Eb, '^-', color='firebrick',
          lw=2, ms=7,
          label=(f'$E_\\mathrm{{bias}}$ (small, non-monotone; '
                 f'slope reported for reference only)'))
N_grid = np.array([N.min(), N.max()])
ax.loglog(N_grid, Es[0] * (N_grid/N[0])**(-0.5),
          'k:', lw=1.2, label=r'$N^{-1/2}$ reference')
ax.set_xlabel('N'); ax.set_ylabel('Error')
ax.set_title('RB--GBMC production: bias--spread--total decomposition')
ax.grid(True, which='both', alpha=0.3)
ax.legend(loc='lower left', fontsize=8, framealpha=0.9)

txt = (f'Note: $E_\\mathrm{{bias}}$ column is non-monotone\n'
       f'(0.00261 at $N=3200$; 0.00336 at $N=6400$).\n'
       f'Bias fit: slope $= {s_b:.3f}$, $R^2 = {r2_b:.3f}$, '
       f'95%~CI $= [{ci_b[0]:.3f}, {ci_b[1]:.3f}]$ (wide;\n'
       f'not treated as headline evidence).')
ax.text(0.98, 0.98, txt, transform=ax.transAxes,
        ha='right', va='top', fontsize=7.5,
        bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.85))

fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT,
        f'production_gbmc_bias_spread_total_vs_N_fixed.{ext}'),
        dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'wrote production_gbmc_bias_spread_total_vs_N_fixed.pdf')


# Fix 2: heat_bias_spread_total_vs_N headlines spread, not total
rows = list(csv.DictReader(open(f'{REGEN}/t4_heat_extended/summary_by_N.csv')))
N  = np.array([int(r['N']) for r in rows])
Eb = np.array([float(r['E_bias']) for r in rows])
Es = np.array([float(r['E_spread']) for r in rows])
Et = np.array([float(r['E_total']) for r in rows])

s_t, r2_t, ci_t = slope_ci(N, Et)
s_s, r2_s, ci_s = slope_ci(N, Es)

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(N, Eb, 'bs-', lw=1.8, ms=6, label=r'$E_\mathrm{bias}$')
ax.loglog(N, Es, 'ro-', lw=1.8, ms=6, label=r'$E_\mathrm{spread}$')
ax.loglog(N, Et, 'g^-', lw=1.8, ms=6, label=r'$E_\mathrm{total}$')
N_ref = np.array([N.min(), N.max()])
c0 = Et[0] * N[0]**0.5
ax.loglog(N_ref, c0 * N_ref**(-0.5), 'k--', lw=1.2, label=r'$O(N^{-1/2})$')

ci_s_str = f'[{ci_s[0]:.3f},{ci_s[1]:.3f}]'
ci_t_str = f'[{ci_t[0]:.3f},{ci_t[1]:.3f}]'
ax.set_title(f'Heat GRW: bias--spread--total vs N\n'
             f'HEADLINE $E_\\mathrm{{spread}}$ slope $= {s_s:.3f}$, '
             f'95%~CI $= {ci_s_str}\\;$ '
             f'(secondary: $E_\\mathrm{{total}}$ slope $= {s_t:.3f}$, '
             f'CI $= {ci_t_str}$)', fontsize=9)
ax.set_xlabel('N'); ax.set_ylabel('Error ($L^2$)')
ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT,
        f'heat_bias_spread_total_vs_N_fixed.{ext}'),
        dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'wrote heat_bias_spread_total_vs_N_fixed.pdf')
