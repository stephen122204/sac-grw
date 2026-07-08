"""Smoke-test wrapper: run the T6 production study at reduced N/S to time it."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_gbmc_production_n_refinement as m
m.N_SEQ = [100, 200]
m.S = 3
m.OUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'smoke_output', 'gbmc_production')
os.makedirs(m.OUT_BASE, exist_ok=True)
t0 = time.perf_counter()
rates = m.run_study()
print(f"\nSmoke wall time: {time.perf_counter()-t0:.1f}s")
if rates:
    print(f"Smoke spread slope: {rates.get('spread_slope', float('nan')):.4f}")
