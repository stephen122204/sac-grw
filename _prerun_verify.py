"""Pre-run verification checks for all 5 study scripts."""
import sys

checks = {}

with open('study_t1_gbmc_dt_bias.py') as f:
    src1 = f.read()
checks['t1_bias_formula']   = 'u_mean - u_exact' in src1
checks['t1_spread_formula'] = 'u_arr - u_mean' in src1
checks['t1_total_formula']  = 'u_arr - u_exact' in src1
checks['t1_identity_check'] = 'identity' in src1
checks['t1_no_smoothing']   = 'convolve' not in src1 and 'gradient' not in src1

with open('study_t2_traveling_shock.py') as f:
    src2 = f.read()
checks['t2_H1_transport']  = 'x_p = x_p + v * dt' in src2
checks['t2_H2_sort']       = 'argsort(x_p' in src2
checks['t2_H3_cumsum']     = 'u_inf + np.cumsum(m_p)' in src2
checks['t2_H5_subchar']    = 'subcharacteristic' in src2.lower() or 'max_u >= a' in src2
checks['t2_H7_switch']     = 'rng.random(N) < p_plus' in src2
checks['t2_H8_diffuse']    = 'rng.normal(0.0, sigma' in src2
checks['t2_no_reflect']    = 'reflect' not in src2.lower()
checks['t2_no_smoothing']  = 'convolve' not in src2 and 'gradient' not in src2
checks['t2_u_inf_c_plus_A'] = 'u_inf = c + A' in src2
checks['t2_mass_conservation'] = 'mass_change' in src2 or 'mass_outside' in src2

with open('study_t4_heat_extended.py') as f:
    src4 = f.read()
checks['t4_N_max_50000']   = '50000' in src4
checks['t4_default_S30']   = 'S=30' in src4
checks['t4_bias_decomp']   = 'E_bias' in src4 and 'E_spread' in src4 and 'E_total' in src4
checks['t4_no_dt_reason']  = 'exact in time' in src4 or 'no_dt_study_reason' in src4

with open('study_t5_fhn_extended.py') as f:
    src5 = f.read()
checks['t5_l1']          = "'l1'" in src5
checks['t5_l2']          = "'l2'" in src5
checks['t5_linf']        = "'linf'" in src5
checks['t5_rel_l2']      = "'rel_l2'" in src5
checks['t5_center_err']  = 'center_err' in src5
checks['t5_speed_err']   = 'speed_err' in src5
checks['t5_aligned']     = 'aligned' in src5
checks['t5_dt_study']    = 'dt_results' in src5 and 'dt_seq' in src5
checks['t5_no_smoothing']= 'convolve' not in src5

all_pass = True
for k, v in sorted(checks.items()):
    status = 'PASS' if v else 'FAIL'
    if not v:
        all_pass = False
    print(f'  {status}  {k}')

print()
if all_pass:
    print('ALL CHECKS PASSED — proceed to full runs')
else:
    print('SOME CHECKS FAILED — inspect before running')
sys.exit(0 if all_pass else 1)
