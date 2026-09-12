"""Recompute every number the coupling manuscript quotes, from archived evidence.

Nothing here reruns a simulation. Each entry records the archive it came from, so
the claim-to-evidence map can be checked mechanically. Two quantities are
computed here for the first time and are marked `new_here`:

  * bootstrap intervals for the nonlinear EMPIRICAL MSE ratios (the independent
    review reported the point estimates without uncertainty);
  * the FULL/WITHIN variance ratio for the cubic flux, from archived integrated
    variances.

Empirical MSE convention, stated because an earlier report used a different one:
    MSE_emp = mean_r (Q_r - Q_ref)^2 = bias^2 + s^2 (n-1)/n,
where s^2 is the ddof=1 sample variance. A plug-in `bias^2 + s^2` exceeds this by
s^2/n and is not reported.
"""

import csv, json
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
OUTP = HERE / "numbers.json"
CODE = HERE.parents[1]
ARCH = CODE / "output"
REVIEW = HERE.parent / "evidence"
BOOT, SEED = 6000, 20260910


def jload(p):
    return json.load(open(p))


def boot_ratio(num, den, rng, stat):
    """Whole-replicate percentile interval for stat(num)/stat(den), paired
    replicates resampled together (the two arms share replicate indices only in
    the sense that each arm is resampled on its own index set of equal size)."""
    n = len(num)
    idx = rng.integers(0, n, (BOOT, n))
    r = np.array(
        [
            stat(num[i]) / stat(den[j])
            for i, j in zip(idx, rng.integers(0, n, (BOOT, n)))
        ]
    )
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main():
    out = {
        "convention": {
            "empirical_mse": "mean_r (Q_r - Q_ref)^2 = bias^2 + s^2 (n-1)/n, s^2 with ddof=1",
            "interval_reading": "an interval containing 1 means UNRESOLVED at this sample size, "
            "not that the effect is absent",
            "bootstrap": f"{BOOT} whole-replicate percentile resamples, seed {SEED}",
        }
    }
    rng = np.random.default_rng(SEED)

    # ---------------------------------------------------------------- fields
    z = np.load(REVIEW / "replayed_fields.npz")
    x = z["x"]
    dx = float(x[1] - x[0])
    ref = z["reference"]
    PM = {a: 0.5 * (z[f"{a}_A"] + z[f"{a}_B"]) for a in ("RAW", "FULL", "WITHIN")}
    n = PM["RAW"].shape[0]
    ivar = {a: float(dx * np.sum(PM[a].var(0, ddof=1))) for a in PM}
    imse = {a: float(np.mean(dx * np.sum((PM[a] - ref) ** 2, axis=1))) for a in PM}

    def bs_field(a, b, kind):
        f = (
            (lambda M: float(dx * np.sum(M.var(0, ddof=1))))
            if kind == "var"
            else (lambda M: float(np.mean(dx * np.sum((M - ref) ** 2, axis=1))))
        )
        ia = rng.integers(0, n, (BOOT, n))
        ib = rng.integers(0, n, (BOOT, n))
        r = np.array([f(PM[a][i]) / f(PM[b][j]) for i, j in zip(ia, ib)])
        return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]

    out["field"] = {
        "source": "round22-independent-review/replayed_fields.npz (independent replay of "
        "output/round22_corrections_2026_09_10 at N=8192, h=0.005, nu=0.1, T=1, 48 pairs)",
        "N": 8192,
        "h": 0.005,
        "nu": 0.1,
        "T": 1.0,
        "replicates": n,
        "integrated_variance": ivar,
        "integrated_mse": imse,
        "ratios": {},
    }
    for a, b in (("WITHIN", "RAW"), ("FULL", "RAW"), ("FULL", "WITHIN")):
        out["field"]["ratios"][f"{a}/{b}"] = {
            "variance": ivar[a] / ivar[b],
            "variance_ci": bs_field(a, b, "var"),
            "mse": imse[a] / imse[b],
            "mse_ci": bs_field(a, b, "mse"),
        }

    # ------------------------------------------------------------ nonlinear
    def pg(M):
        return np.max(np.abs(np.gradient(np.atleast_2d(M), dx, axis=1)), axis=1)

    def lev(M, L=0.25):
        o = []
        for a in np.atleast_2d(M):
            i = int(np.argmax(a))
            if not np.any(a[i:] < L):
                o.append(np.nan)
                continue
            j = i + int(np.argmax(a[i:] < L))
            o.append(float(np.interp(-L, -a[j - 1 : j + 1], x[j - 1 : j + 1])))
        return np.array(o)

    nl = {
        "source": "same replay archive; observable definitions in the driver docstring",
        "rows": [],
        "ratios": [],
    }
    samples = {}
    for oname, fn in (("max|grad u|", pg), ("level x(u=0.25)", lev)):
        qref = float(fn(ref[None, :])[0])
        for a in ("RAW", "WITHIN", "FULL"):
            qm = fn(PM[a])
            qq = 0.5 * (fn(z[f"{a}_A"]) + fn(z[f"{a}_B"]))
            for est, v in (("Q_of_mean", qm), ("mean_of_Q", qq)):
                v = v[~np.isnan(v)]
                samples[(oname, est, a)] = (v, qref)
                b = float(v.mean() - qref)
                s2 = float(v.var(ddof=1))
                nl["rows"].append(
                    dict(
                        observable=oname,
                        estimator=est,
                        arm=a,
                        reference=qref,
                        mean=float(v.mean()),
                        bias=b,
                        variance=s2,
                        empirical_mse=b * b + s2 * (len(v) - 1) / len(v),
                        bias_share=b * b / (b * b + s2 * (len(v) - 1) / len(v)),
                        n=int(len(v)),
                        censored=int(48 - len(v)),
                    )
                )
        for est in ("Q_of_mean", "mean_of_Q"):
            for a, bb in (("FULL", "RAW"), ("FULL", "WITHIN"), ("WITHIN", "RAW")):
                vn, qr = samples[(oname, est, a)]
                vd, _ = samples[(oname, est, bb)]
                emp = lambda v: float(np.mean((v - qr) ** 2))
                ia = rng.integers(0, len(vn), (BOOT, len(vn)))
                ib = rng.integers(0, len(vd), (BOOT, len(vd)))
                r = np.mean((vn[ia] - qr) ** 2, axis=1) / np.mean(
                    (vd[ib] - qr) ** 2, axis=1
                )
                nl["ratios"].append(
                    dict(
                        observable=oname,
                        estimator=est,
                        comparison=f"{a}/{bb}",
                        empirical_mse_ratio=emp(vn) / emp(vd),
                        ci=[
                            float(np.percentile(r, 2.5)),
                            float(np.percentile(r, 97.5)),
                        ],
                        variance_ratio=float(vn.var(ddof=1) / vd.var(ddof=1)),
                        new_here="bootstrap interval computed here",
                    )
                )
    out["nonlinear"] = nl

    # ---------------------------------------------------------------- costs
    c22 = jload(ARCH / "round22_corrections_2026_09_10/costs.json")
    c17 = jload(ARCH / "round17_compare_2026_09_10/compare.json")["stats"]
    out["cost"] = {
        "source": [
            "output/round22_corrections_2026_09_10/costs.json",
            "output/round17_compare_2026_09_10/compare.json",
        ],
        "scope": "serial single-machine wall-clock, interleaved repetitions, solver only; "
        "an implementation result, not an intrinsic property of within-sign pairing",
        "per_pair_seconds": c22,
        "ratios_8192": {
            "FULL/RAW": c22["8192"]["FULL"] / c22["8192"]["RAW"],
            "FULL/WITHIN": c22["8192"]["FULL"] / c22["8192"]["WITHIN"],
            "WITHIN/RAW": c22["8192"]["WITHIN"] / c22["8192"]["RAW"],
        },
        "ratios_2048": {
            "FULL/RAW": c22["2048"]["FULL"] / c22["2048"]["RAW"],
            "FULL/WITHIN": c22["2048"]["FULL"] / c22["2048"]["WITHIN"],
            "WITHIN/RAW": c22["2048"]["WITHIN"] / c22["2048"]["RAW"],
        },
        "pair_over_single_8192": c17["8192_RAW"]["C"] / c17["8192_SINGLE"]["C"],
        "single_8192_s": c17["8192_SINGLE"]["C"],
        "pair_8192_s": c17["8192_RAW"]["C"],
    }
    vc = (
        out["field"]["ratios"]["FULL/WITHIN"]["variance"]
        * out["cost"]["ratios_8192"]["FULL/WITHIN"]
    )
    out["cost"]["variance_times_cost_FULL_over_WITHIN"] = vc

    # ------------------------------------------------------------- mechanism
    mech = jload(ARCH / "round23_mechanism_2026_09_10/mechanism.json")
    out["mechanism"] = {
        "source": "output/round23_mechanism_2026_09_10/mechanism.json",
        "N": mech["N"],
        "h": mech["h"],
        "replicates": mech["reps"],
        "note": "variance ratios use advance_pair_flux, which implements WITHIN correctly; "
        "the sep_ratio column of this archive is defective and is superseded by "
        "output/round24_separation_2026_09_10/separation.json",
        "rows": [
            {
                k: r[k]
                for k in (
                    "profile",
                    "sign_regions",
                    "T",
                    "within_over_raw",
                    "full_over_raw",
                    "full_over_within",
                    "ci",
                )
            }
            for r in mech["rows"]
        ],
    }
    sep = jload(ARCH / "round24_separation_2026_09_10/separation.json")
    out["separation"] = {
        "source": "output/round24_separation_2026_09_10/separation.json",
        "replicates": sep["reps"],
        "geometric_fact": "for two equally sized sets of reals, matching in sorted order "
        "minimises the total absolute matched distance over all bijections; "
        "within-sign matching is a constrained bijection, so every entry of "
        "the matched-state table is <= 1 by construction",
        "rows": sep["rows"],
    }

    # ---------------------------------------------------------- applicability
    app = jload(ARCH / "round21_applicability_2026_09_10/applicability.json")
    out["applicability"] = {
        "source": "output/round21_applicability_2026_09_10/applicability.json",
        "N": app["N"],
        "replicates": app["reps"],
        "rows": app["rows"],
    }
    div22 = jload(ARCH / "round22_corrections_2026_09_10/corrections.json")
    div25 = jload(ARCH / "round25_divergence_2026_09_10/divergence.json")
    out["divergence"] = {
        "source": [
            "output/round22_corrections_2026_09_10/corrections.json",
            "output/round25_divergence_2026_09_10/divergence.json",
        ],
        "round22_40_seeds": div22["divergence"],
        "round25_200_seeds": div25["rows"],
        "pooled_separated": div25["pooled_separated"],
    }
    out["configurations"] = {
        "source": "output/round22_corrections_2026_09_10/corrections.json",
        "rows": div22["configurations"],
    }

    # ------------------------------------------------------------- held-out 1
    c = jload(ARCH / "round17_compare_2026_09_10/compare.json")
    out["heldout_A"] = {
        "source": "output/round17_compare_2026_09_10/compare.json",
        "target": c["target"],
        "target_provenance": c["target_provenance"],
        "pilot_key": c["pilot_key"],
        "eval_key": c["eval_key"],
        "blocks": c["blocks"],
        "N": 8192,
        "arms": {},
    }
    for a in ("SINGLE", "RAW", "FULL", "RQMC"):
        B = c["plans"][a]["B"]
        out["heldout_A"]["arms"][a] = dict(
            B=B,
            trajectories=B * (1 if a in ("SINGLE", "RQMC") else 2),
            attained=c["realised"][a]["mean"],
            se=c["realised"][a]["se"],
            work_s=c["realised"][a]["work_s"],
        )
    tA = float(stats.t.ppf(1 - 0.05 / (2 * 4), c["blocks"] - 1))
    for a in out["heldout_A"]["arms"]:
        r = out["heldout_A"]["arms"][a]
        r["upper_95_pointwise"] = r["attained"] + 2.0395 * r["se"]
        r["upper_95_bonferroni"] = r["attained"] + tA * r["se"]
        r["attained_below_target"] = bool(r["upper_95_bonferroni"] < c["target"])
    out["heldout_A"]["t_pointwise"] = float(stats.t.ppf(0.975, 31))
    out["heldout_A"]["t_bonferroni"] = tA
    out["heldout_A"]["rqmc"] = c["rqmc"]

    # ------------------------------------------------------------- held-out 2
    w = jload(ARCH / "round23_mechanism_2026_09_10/worktarget.json")
    t = jload(ARCH / "round24_separation_2026_09_10/timing_check.json")
    out["heldout_B"] = {
        "source": [
            "output/round23_mechanism_2026_09_10/worktarget.json",
            "output/round24_separation_2026_09_10/timing_check.json",
        ],
        "target": w["tol"],
        "pilot_key": w["pilot_key"],
        "eval_key": w["eval_key"],
        "blocks": w["blocks"],
        "N": w["N"],
        "tuning_cost_s": w["tuning_cost_s"],
        "arms": {},
    }
    for a in ("RAW", "WITHIN", "FULL"):
        B = w["plans"][a]["B"]
        r = w["realised"][a]
        se = r["se"]
        out["heldout_B"]["arms"][a] = dict(
            B=B,
            trajectories=2 * B,
            attained=r["mean"],
            se=se,
            upper_95_pointwise=r["mean"] + float(stats.t.ppf(0.975, 31)) * se,
            work_sequential_s=r["work_s"],
            work_interleaved_s=t["interleaved"][a],
        )
    tB = float(stats.t.ppf(1 - 0.05 / (2 * 3), w["blocks"] - 1))
    for a in out["heldout_B"]["arms"]:
        r = out["heldout_B"]["arms"][a]
        r["upper_95_bonferroni"] = r["attained"] + tB * r["se"]
        r["attained_below_target"] = bool(r["upper_95_bonferroni"] < w["tol"])
    out["heldout_B"]["t_pointwise"] = float(stats.t.ppf(0.975, 31))
    out["heldout_B"]["t_bonferroni"] = tB
    out["heldout_B"]["work_ratios_interleaved"] = {
        "FULL/RAW": t["interleaved"]["FULL"] / t["interleaved"]["RAW"],
        "FULL/WITHIN": t["interleaved"]["FULL"] / t["interleaved"]["WITHIN"],
    }

    # ------------------------------------------------------------------ flux
    rows = list(
        csv.DictReader(open(ARCH / "round09_costs_control_2026_09_10/rows.csv"))
    )
    iv = {(r["flux"], r["arm"]): float(r["int_var"]) for r in rows}
    out["flux_transfer"] = {
        "source": "output/round09_costs_control_2026_09_10/rows.csv (N=1600, h=0.005, 64 seeds)",
        "rows": [
            {
                "flux": f,
                "FULL/RAW": iv[(f, "FULL")] / iv[(f, "RAW")],
                "WITHIN/RAW": iv[(f, "WITHIN")] / iv[(f, "RAW")],
                "FULL/WITHIN": iv[(f, "FULL")] / iv[(f, "WITHIN")],
                "new_here": "FULL/WITHIN formed here from the archived integrated variances",
            }
            for f in ("burgers", "cubic")
        ],
        "controls": [
            {"flux": f, "arm": a, "int_var_over_RAW": iv[(f, a)] / iv[(f, "RAW")]}
            for f in ("burgers", "cubic")
            for a in ("RAW+RB", "RAW+CVF(c*)", "FULL+RB", "FULL+CVF(c*)", "FULL")
        ],
    }

    # -------------------------------------------------------- control arms
    zc = np.load(ARCH / "round09_costs_control_2026_09_10/fields.npz")
    xg = zc["x_grid"]
    dxc = float(xg[1] - xg[0])
    arms = [
        "RAW",
        "RAW+RB",
        "RAW+CVH(c=1)",
        "RAW+CVH(c*)",
        "RAW+CVF(c=1)",
        "RAW+CVF(c*)",
        "WITHIN",
        "FULL",
        "FULL+RB",
        "FULL+CVH(c=1)",
        "FULL+CVH(c*)",
        "FULL+CVF(c=1)",
        "FULL+CVF(c*)",
    ]
    cost = {(r["flux"], r["arm"]): (float(r["cost"]), float(r["oneoff"])) for r in rows}
    ctrl = {
        "source": "output/round09_costs_control_2026_09_10/{fields.npz,rows.csv} "
        "(N=1600, h=0.005, nu=0.1, T=1, 64 seeds, arms share seed streams)",
        "paired_bootstrap": "same resampled replicate indices in both arms",
        "rows": [],
    }
    for flux in ("burgers", "cubic"):
        F = {a: zc[f"{flux}_{a}"].astype(np.float64) for a in arms}
        iv = {a: float(dxc * np.sum(F[a].var(0, ddof=1))) for a in arms}
        nrep = F["RAW"].shape[0]
        ridx = rng.integers(0, nrep, (BOOT, nrep))
        bsv = {
            a: np.array([dxc * np.sum(F[a][i].var(0, ddof=1)) for i in ridx])
            for a in arms
        }
        for a in arms:
            c, one = cost[(flux, a)]
            cR, _ = cost[(flux, "RAW")]
            r_raw = bsv[a] / bsv["RAW"]
            row = dict(
                flux=flux,
                arm=a,
                int_var=iv[a],
                ratio_vs_RAW=iv[a] / iv["RAW"],
                ci_vs_RAW=[
                    float(np.percentile(r_raw, 2.5)),
                    float(np.percentile(r_raw, 97.5)),
                ],
                cost_s=c,
                one_off_s=one,
                var_times_cost_vs_RAW=(iv[a] / iv["RAW"]) * (c / cR),
            )
            if a in ("FULL", "FULL+RB", "FULL+CVH(c*)", "FULL+CVF(c*)"):
                r_w = bsv[a] / bsv["WITHIN"]
                row["ratio_vs_WITHIN"] = iv[a] / iv["WITHIN"]
                row["ci_vs_WITHIN"] = [
                    float(np.percentile(r_w, 2.5)),
                    float(np.percentile(r_w, 97.5)),
                ]
            ctrl["rows"].append(row)
    out["controls"] = ctrl

    # -------------------------------------------------------------- bias/work
    jc = jload(ARCH / "round07_joint_cell_2026_09_09/joint_cell.json")
    out["bias_regime"] = {
        "source": "output/round07_joint_cell_2026_09_09/joint_cell.json",
        "rows": jc["rows"],
        "caveat": jc["caveat"],
    }

    # ------------------------------------------------------------- final step
    fs = jload(ARCH / "round06_finalstep_2026_09_09/finalstep.json")
    fsv = jload(ARCH / "round25_finalstage_2026_09_10/finalstage.json")
    out["final_stage_check"] = {
        "source": "output/round25_finalstage_2026_09_10/finalstage.json",
        "statement": fsv["statement"],
        "draws": fsv["draws"],
        "conditional_checks": fsv["conditional_checks"],
        "share": fsv["share"],
        "note": fsv["note"]
        + " Historical key guaranteed_share means an estimated final-only/full-history reduction ratio; it is not a guaranteed share of terminal improvement.",
    }
    out["final_stage"] = {
        "source": "output/round06_finalstep_2026_09_09/finalstep.json",
        "rows": [
            {
                k: r[k]
                for k in (
                    "problem",
                    "N",
                    "h",
                    "K",
                    "seeds",
                    "V_RAW",
                    "V_FINAL",
                    "V_FULL",
                    "E_Delta",
                    "E_Delta_se",
                    "final_step_share",
                )
            }
            for r in fs["rows"]
        ],
    }

    # ------------------------------------------------------------ calibration
    cal = jload(ARCH / "round25_calibration_2026_09_10/calibration.json")
    out["calibration"] = {
        "source": "output/round25_calibration_2026_09_10/calibration.json",
        "superseded": "output/round22_corrections_2026_09_10/calibration.json "
        "used one standardised shape for both synthetic arms",
        "rows": cal["rows"],
        "shapes": cal["shapes"],
        "construction": cal["construction"],
        "limitation": cal["limitation"],
    }

    # ------------------------------------------------------ marginal witness
    aud = jload(ARCH / "round14_audit_2026_09_10/audit.json")
    out["law_witness"] = {
        "source": "output/round14_audit_2026_09_10/audit.json",
        "single_vs_production": aud["fixtures"]["single_vs_production"],
        "seed_deterministic": aud["fixtures"]["seed_deterministic"],
    }

    # ------------------------------------------------------- counterexample
    ce = jload(ARCH / "round07_counterexample_2026_09_09/counterexample.json")
    out["counterexample"] = {
        "source": "output/round07_counterexample_2026_09_09/counterexample.json",
        "config": ce["config"],
        "delta_U": ce["delta_U"],
        "log_exceptional_probability": ce["log_exceptional_probability"],
        "regime": ce["regime"],
        "scope": ce["scope"],
    }

    # ---------------------------------------------------------- diagnostic
    tr = jload(ARCH / "round19_transfer_2026_09_10/transfer.json")
    au = jload(ARCH / "round19_audit_2026_09_10/audit.json")
    out["diagnostic"] = {
        "source": [
            "output/round19_transfer_2026_09_10/transfer.json",
            "output/round19_audit_2026_09_10/audit.json",
        ],
        "accuracy": tr["accuracy"],
        "rows": len(tr["rows"]),
        "exponent_cut_sensitivity": jload(
            ARCH / "round19_transfer_2026_09_10/work.json"
        )["cut_robustness"],
        "placebo_shared": au["placebo_shared"],
        "artefact_if_constant": au["artefact_if_constant"],
        "status": "subordinate; the exponent is not stable under the tail cut "
        "and is not reported as a scaling law",
    }

    # ----------------------------------------------------------- observables
    obs18 = jload(ARCH / "round18_observable_2026_09_10/observable.json")
    fi = jload(ARCH / "round18_observable_2026_09_10/f_intervals.json")
    out["locations"] = {
        "source": [
            "output/round18_observable_2026_09_10/observable.json",
            "output/round18_observable_2026_09_10/f_intervals.json",
        ],
        "key": obs18["key"],
        "N": obs18["N"],
        "replicates": obs18["reps"],
        "metric": "FULL/RAW variance ratio, F(n-1,n-1) intervals",
        "rows": fi,
        "quintiles": obs18["quintiles"],
        "integrated_pointwise": obs18["integrated_pointwise"],
        "median_pointwise": obs18["median_pointwise"],
    }

    json.dump(out, open(OUTP, "w"), indent=1)
    print(f"wrote {OUTP}")
    for k in out:
        print("  ", k)


if __name__ == "__main__":
    main()
