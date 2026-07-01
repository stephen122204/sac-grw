"""Table 6 confirmation: run T2 traveling-shock at S=30 (all other params default).

Manuscript §7.8, line 1086 pins:
  A=1, c=0.5, nu=0.5, a=2, T=1, dt=0.005, S=30
  N in {100, 200, 400, 800, 1600, 3200}

run_task2() defaults match everything except S (default 10).
Base seed = 42, same as T4/T5/T6 studies. Seeds are 42, 43, ..., 71.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_t2_traveling_shock as m

m.OUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'regen_output', 't2_traveling_S30')
os.makedirs(m.OUT_BASE, exist_ok=True)

t0 = time.perf_counter()
result = m.run_task2(
    nu=0.5,
    A=1.0,
    c=0.5,
    a=2.0,
    L=8.0,
    x0=3.0,
    output_times=[0.25, 0.5, 1.0],
    N_seq=[100, 200, 400, 800, 1600, 3200],
    dt_seq=[0.02, 0.01, 0.005, 0.0025],
    S=30,
    base_seed=42,
    run_sharp=False,           # save time; the sharp-layer nu=0.2 run is not in Table 6
)
print(f"\n=== T2 traveling-shock S=30 wall time: {time.perf_counter()-t0:.1f}s ===")
