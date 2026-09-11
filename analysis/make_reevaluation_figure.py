"""Principal figure and summary table for the reevaluation study.

Left panel: the paired fitted-viscosity difference, in units of D_vel, against
dimensionless time T A^2/nu.  Curves are the two parameter-free predictions;
points are measurements.  Right panel: predicted against measured for every
configuration, including the held-out cases and the Gaussian transient.
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.predict import predict

BASE = 'output/reevaluation_2026_09'
RT = os.path.join(BASE, 'response_time')
OUT = os.path.join(BASE, 'figures')

# archived scaled-step multi-viscosity rows, mapped to their dimensionless time
ARCHIVED = [  # nu,   dt,        taut, half-window in shock widths
    (0.5,   0.0025,   1.0,  2.0), (0.25, 0.00125, 2.0,  4.0),
    (0.1,   0.0005,   5.0, 10.0), (0.05, 0.00025, 10.0, 12.0),
    (0.025, 0.000125, 20.0, 12.0)]


def load_A(at):
    p = os.path.join(RT, f'A_at{at:g}'.replace('.', 'p') + '_summary.json')
    d = json.load(open(p))
    return d['metadata'], d['summary']


def archived_g():
    d = json.load(open('output/final_prepublication_tests/'
                       'gbmc_multiviscosity_scaled_dt/summary.json'))
    out = {}
    for r in d['paired_nu_contrasts']:
        if r['contrast'].endswith('- cond_mean'):
            at = 2.0 if 'a2' in r['contrast'] else 4.0
            out[(round(r['nu'], 6), at)] = r['excess_over_D_label']
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    arch = archived_g()
    taut_curve = np.concatenate([np.linspace(0.1, 2, 20), np.linspace(2.2, 25, 24)])
    colors = {2.0: '#1f77b4', 4.0: '#7b3fa0'}

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
    ax = axes[0]
    table = []
    for at in (2.0, 4.0):
        c = colors[at]
        g2 = [predict(at, 0.005, t)['M2_g'] for t in taut_curve]
        g1 = [predict(at, 0.005, t)['M1_g'] for t in taut_curve]
        ax.plot(taut_curve, g2, '-', color=c, lw=1.8,
                label=f'derived model, $a/A={at:g}$')
        ax.plot(taut_curve, g1, '--', color=c, lw=1.1, alpha=0.75,
                label=f'constant $D_{{\\rm vel}}$, $a/A={at:g}$')
        meta, summ = load_A(at)
        t = np.array([r['taut'] for r in summ])
        g = np.array([r['g'] for r in summ])
        lo = np.array([r['g_lo'] for r in summ])
        hi = np.array([r['g_hi'] for r in summ])
        ax.errorbar(t, g, yerr=[g - lo, hi - g], fmt='o', ms=5, color=c,
                    capsize=3, lw=1.2, zorder=5,
                    label=f'Study A (new runs), $a/A={at:g}$')
        for r in summ:
            p = predict(at, 0.005, r['taut'])
            table.append(dict(study='A', at=at, dtt=0.005, taut=r['taut'],
                              g=r['g'], g_lo=r['g_lo'], g_hi=r['g_hi'],
                              M2=p['M2_g'], M1=p['M1_g']))
        # Archived scaled-step rows at the same dimensionless step.  These use
        # narrower evaluation windows at the two largest viscosities, so they
        # are NOT expected to lie on the 12-width curve; each point's own
        # window-matched prediction is drawn as a cross beside it.
        tt = [t_ for (_, _, t_, _) in ARCHIVED]
        ga = [arch[(nu, at)] for (nu, _, _, _) in ARCHIVED]
        gp = [predict(at, 0.005, t_, half_widths=hw)['M2_g']
              for (_, _, t_, hw) in ARCHIVED]
        ax.plot(tt, ga, 's', ms=6, mfc='none', mew=1.4, color=c, zorder=4,
                label=f'archived $\\nu$-sweep, $a/A={at:g}$')
        ax.plot(tt, gp, '+', ms=8, mew=1.4, color=c, zorder=4,
                label=f'its window-matched prediction, $a/A={at:g}$')
    ax.axhline(1.0, color='0.6', lw=0.8, ls=':')
    ax.set_xscale('log')
    ax.set_xlabel(r'dimensionless time $TA^{2}/\nu$')
    ax.set_ylabel(r'paired fitted-viscosity difference $/\ D_{\mathrm{vel}}$')
    ax.set_ylim(0, 1.62)
    ax.legend(fontsize=6.6, ncol=1, loc='upper left', framealpha=0.92)

    # held-out cases
    ax2 = axes[1]
    for name in ('C1', 'C2', 'C3', 'C4'):
        p = os.path.join(RT, f'{name}_summary.json')
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        m, r = d['metadata'], d['summary'][0]
        pr = predict(m['at'], m['dtt'], m['taut'])
        table.append(dict(study=name, at=m['at'], dtt=m['dtt'], taut=m['taut'],
                          g=r['g'], g_lo=r['g_lo'], g_hi=r['g_hi'],
                          M2=pr['M2_g'], M1=pr['M1_g']))
    tr = json.load(open(os.path.join(BASE, 'transient_prediction.json')))

    for row in table:
        mk = 'o' if row['study'] == 'A' else 'D'
        col = '#d62728' if row['study'] != 'A' else '#1f77b4'
        ax2.errorbar(row['M2'], row['g'],
                     yerr=[[row['g'] - row['g_lo']], [row['g_hi'] - row['g']]],
                     fmt=mk, ms=5, color=col, capsize=2, lw=1.0)
        ax2.plot(row['M1'], row['g'], 'x', ms=5, color='0.55', mew=1.1)
    lim = [0, 1.2]
    ax2.plot(lim, lim, '-', color='0.6', lw=0.9)
    ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.set_xlabel('predicted ratio')
    ax2.set_ylabel('measured ratio')
    from matplotlib.lines import Line2D
    ax2.legend(handles=[
        Line2D([], [], marker='o', ls='', color='#1f77b4', label='Study A, derived model'),
        Line2D([], [], marker='D', ls='', color='#d62728', label='held out, derived model'),
        Line2D([], [], marker='x', ls='', color='0.55', label='constant $D_{\\rm vel}$')],
        fontsize=8, loc='upper left')

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'reevaluation_response.{ext}'),
                    dpi=160, bbox_inches='tight')
    plt.close(fig)

    e2 = np.array([r['g'] - r['M2'] for r in table])
    e1 = np.array([r['g'] - r['M1'] for r in table])
    inside2 = sum(r['g_lo'] <= r['M2'] <= r['g_hi'] for r in table)
    inside1 = sum(r['g_lo'] <= r['M1'] <= r['g_hi'] for r in table)
    stats = dict(n_cells=len(table),
                 rms_derived=float(np.sqrt((e2 ** 2).mean())),
                 rms_constant=float(np.sqrt((e1 ** 2).mean())),
                 max_abs_derived=float(np.max(np.abs(e2))),
                 max_abs_constant=float(np.max(np.abs(e1))),
                 inside_ci_derived=int(inside2), inside_ci_constant=int(inside1),
                 transient=tr, cells=table)
    json.dump(stats, open(os.path.join(BASE, 'reevaluation_summary.json'), 'w'),
              indent=1)
    print(f"cells={len(table)}  RMS(derived)={stats['rms_derived']:.4f}  "
          f"RMS(constant)={stats['rms_constant']:.4f}  "
          f"inside 95% CI: derived {inside2}/{len(table)}, "
          f"constant {inside1}/{len(table)}")
    for r in table:
        print(f"  {r['study']:>2} a/A={r['at']:g} dt~={r['dtt']:g} T~={r['taut']:<5g} "
              f"meas={r['g']:.4f} [{r['g_lo']:.4f},{r['g_hi']:.4f}]  "
              f"M2={r['M2']:.4f} ({r['g']-r['M2']:+.4f})  "
              f"M1={r['M1']:.4f} ({r['g']-r['M1']:+.4f})")


if __name__ == '__main__':
    main()
