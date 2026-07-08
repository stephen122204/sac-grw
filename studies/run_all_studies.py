"""Sequentially run studies T2, T4, T5, T1, T3, each writing into ../regen_output/."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     '..', 'regen_output')
os.makedirs(REGEN, exist_ok=True)

for tag, mod_name in [
    ('T2 traveling-shock',     'study_t2_traveling_shock'),
    ('T4 heat_extended',       'study_t4_heat_extended'),
    ('T5 fhn_extended',        'study_t5_fhn_extended'),
    ('T1 dt-bias',             'study_t1_gbmc_dt_bias'),
    ('T3 cole-hopf plateau',   'study_t3_cole_hopf_plateau'),
]:
    print(f"\n=========== {tag} ===========")
    t0 = time.perf_counter()
    m = __import__(mod_name)
    if hasattr(m, 'OUT_BASE'):
        base = os.path.join(REGEN, mod_name.replace('study_', ''))
        m.OUT_BASE = base
        os.makedirs(base, exist_ok=True)
    fn = None
    for name in ('run_study', 'run_task1', 'run_task2', 'run_task3', 'run_task4', 'run_task5', 'main'):
        fn = getattr(m, name, None)
        if callable(fn):
            print(f"  invoking {mod_name}.{name}()")
            break
    if fn is None:
        print(f"  no run entry found in {mod_name}")
        continue
    try:
        rv = fn()
    except SystemExit as e:
        rv = f"SystemExit({e})"
    except Exception as e:
        rv = f"EXCEPTION: {type(e).__name__}: {e}"
    dt = time.perf_counter() - t0
    print(f"  {tag} wall time: {dt:.1f}s  return: {str(rv)[:200]}")
print("\n=== all studies done ===")
