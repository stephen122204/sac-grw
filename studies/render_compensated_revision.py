"""Render revision tables and scientific figures from saved study results."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'output/compensated_accuracy_2026_09_09'
LABEL={'original':'Original','compensated':'Compensated','mean':'Mean transport'}
COLOR={'original':'#b54a2d','compensated':'#176da1','mean':'#21805a'}
MARK={'original':'o','compensated':'s','mean':'^'}
ORDER=['shock_a2_h0.02','shock_a2_h0.01','shock_a2_h0.005',
       'gaussian_a2_dt0.005','gaussian_a2_dt0.00125',
       'gaussian_a4_dt0.005','gaussian_a4_dt0.00125']

def main():
    p=argparse.ArgumentParser();p.add_argument('--paper-dir',type=Path,required=True)
    paper=p.parse_args().paper_dir
    figs=paper/'figures-2';figs.mkdir(exist_ok=True)
    data=[json.loads((DATA/n/'summary.json').read_text()) for n in ORDER]
    for d in data:
        for row in d['rows']:
            assert np.isclose(row['total']**2,row['bias']**2+row['spread']**2,rtol=1e-11)
    text=[r'\begin{table}[htbp]',r'\centering\small',
          r'\caption{Full-profile errors and median solver times. All cells use $N=6400$, $S=50$. $B$, $F$, and $E$ are defined in Eqs.~\eqref{eq:BF}--\eqref{eq:E}.}',
          r'\label{tab:accuracy}',r'\begin{tabular}{llrrrrrr}',r'\toprule',
          r'Problem & Method & $a$ & $\dt$ & $B$ & $F$ & $E$ & Time (s) \\',r'\midrule']
    for d in data:
        c=d['config']
        for i,r in enumerate(d['rows']):
            problem=('Shock' if c['problem']=='shock' else 'Gaussian') if i==0 else ''
            aa=f"{c['a']:g}" if i==0 else '';dd=f"{c['dt']:g}" if i==0 else ''
            text.append(f"{problem} & {LABEL[r['arm']]} & {aa} & {dd} & {r['bias']:.5f} & {r['spread']:.5f} & {r['total']:.5f} & {r['seconds']:.3f} \\\\")
        text.append(r'\addlinespace')
    text += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (paper/'tables-compensated-errors.tex').write_text('\n'.join(text)+'\n')
    text=[r'\begin{table}[htbp]',r'\centering\small',
          r'\caption{Compensated/original and compensated/mean ratios of total error $E$, with paired $95\%$ percentile bootstrap intervals. Ratios below one favor compensation.}',
          r'\label{tab:ratios}',r'\begin{tabular}{lrrll}',r'\toprule',
          r'Problem & $a$ & $\dt$ & Compensated/original & Compensated/mean \\',r'\midrule']
    for d in data:
        c=d['config']; vals=[]
        for b in ('original','mean'):
            r=d['compensated_ratios'][b];lo,hi=r['interval']
            vals.append(f"{r['ratio']:.3f} [{lo:.3f}, {hi:.3f}]")
        text.append(f"{c['problem'].capitalize()} & {c['a']:g} & {c['dt']:g} & {vals[0]} & {vals[1]} \\\\")
    text += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (paper/'tables-compensated-ratios.tex').write_text('\n'.join(text)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.labelsize':11,'legend.fontsize':9})
    fig,axs=plt.subplots(1,2,figsize=(9,3.35),layout='constrained')
    for ax,name,title in zip(axs,[ORDER[0],ORDER[5]],['Stationary shock','Gaussian transient']):
        z=np.load(DATA/name/'profiles.npz')
        for arm in LABEL:
            ax.plot(z['x'],z[arm].mean(axis=0)-z['reference'],label=LABEL[arm],color=COLOR[arm],lw=1.5)
        ax.axhline(0,color='0.4',lw=.6);ax.set(xlabel='$x$',ylabel='Ensemble-mean error',title=title)
        ax.grid(alpha=.16);ax.legend(frameon=False)
    axs[0].set_xlim(-2,6)
    fig.savefig(figs/'compensated_profiles.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
    for ax,subset,title in zip(axs,[data[:3],data[5:]],['Shock, $a=2$','Gaussian, $a=4$']):
        for arm in LABEL:
            rr=[next(r for r in d['rows'] if r['arm']==arm) for d in subset]
            ax.plot([r['seconds'] for r in rr],[r['total'] for r in rr],
                    marker=MARK[arm],color=COLOR[arm],label=LABEL[arm],lw=1)
        ax.set(xlabel='Median solver time (seconds)',ylabel='RMS profile error $E$',title=title)
        ax.legend(frameon=False);ax.grid(alpha=.18)
    fig.savefig(figs/'compensated_cost.pdf');plt.close(fig)
    response=json.loads((ROOT/'output/reevaluation_2026_09/reevaluation_summary.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for ax,a in zip(axs,[2,4]):
        cells=[d for d in response['cells'] if d['study']=='A' and d['at']==a]
        t=np.array([d['taut'] for d in cells]);g=np.array([d['g'] for d in cells])
        ax.plot(t,[d['M2'] for d in cells],color='#176da1',label='State-dependent model')
        ax.plot(t,[d['M1'] for d in cells],color='#b54a2d',ls='--',label='Constant model')
        ax.errorbar(t,g,yerr=np.array([g-np.array([d['g_lo'] for d in cells]),np.array([d['g_hi'] for d in cells])-g]),
                    fmt='o',ms=4,capsize=3,color='black',label='Particle measurement')
        ax.set(xlabel=r'Dimensionless time $\tau$',ylabel='Normalized response $g$',title=fr'$\alpha={a}$, $h=0.005$')
        ax.grid(alpha=.18);ax.legend(frameon=False)
    fig.savefig(figs/'response_overhaul.pdf');plt.close(fig)
    text=[r'\begin{table}[H]',r'\centering\small',
          r'\caption{All response cells: paired particle measurements and deterministic predictions for $g$. Each particle cell uses $S=400$. Intervals are pointwise $95\%$ bootstrap intervals; the time-series rows share trajectories.}',
          r'\label{tab:response}',r'\begin{tabular}{rrrlll}',r'\toprule',
          r'$\alpha$ & $h$ & $\tau$ & Particle $g$ [interval] & State-dependent & Constant \\',r'\midrule']
    for i,d in enumerate(response['cells']):
        if i in (7,14):text.append(r'\midrule')
        text.append(f"{d['at']:g} & {d['dtt']:g} & {d['taut']:g} & {d['g']:.4f} [{d['g_lo']:.4f}, {d['g_hi']:.4f}] & {d['M2']:.4f} & {d['M1']:.4f} \\\\")
    text += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (paper/'table-response-overhaul.tex').write_text('\n'.join(text)+'\n')
    print('Rendered 3 figures and 3 tables from 7 accuracy cells and 18 response cells.')

if __name__=='__main__':main()
