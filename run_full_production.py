"""Full T6 production sweep: 7 N × 50 seeds, in-repo output dir."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_gbmc_production_n_refinement as m
m.OUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'regen_output', 'gbmc_production')
os.makedirs(m.OUT_BASE, exist_ok=True)
t0 = time.perf_counter()
rates = m.run_study()
print(f"\n=== Full T6 production wall time: {time.perf_counter()-t0:.1f}s ===")
if rates:
    print(f"spread slope: {rates.get('spread_slope', float('nan')):.4f}")
    print(f"spread CI:    [{rates.get('spread_ci_lo', float('nan')):.4f}, "
          f"{rates.get('spread_ci_hi', float('nan')):.4f}]")
