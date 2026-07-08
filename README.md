<!-- TODO(license): add a LICENSE file and update CITATION.cff before public release. -->

# RB-GBMC companion code — Heat, Burgers, FitzHugh–Nagumo

Companion code for:

> **Gradient Random Walk Methods for Diffusive PDEs and a Relaxation–Brownian
> Particle Scheme for Viscous Burgers' Equation.**
> Stephen Abkin and Prabir Daripa, 2026. arXiv: TBD.

Gradient random walk (GRW) particle solvers for the heat equation and the
FitzHugh–Nagumo (FHN) system, and a relaxation–Brownian gradient Monte Carlo
(RB-GBMC) particle scheme for viscous Burgers' equation, together with the
studies behind every table and figure in the paper.

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
pip install -r requirements.txt    # numpy, matplotlib; scipy for full reruns

# Check the checked-in paper numbers without rerunning anything (seconds):
python reproduce.py verify

# Rerun an individual study with the paper configuration (needs scipy; ~10-20 min each):
python reproduce.py t6      # production GBMC N-refinement (Tables 5-6)
python reproduce.py --help  # all targets and what they produce
```

`MANIFEST.md` maps each manuscript table/figure to its study, script, and
checked-in data.

---

## WARNING: function defaults are exploration settings

The default arguments of the study functions (`run_task1` … `run_task5`,
etc.) are **exploration settings, not the paper configurations**. The paper
configurations live in the `reproduce.py` entry points (and the wrapper
scripts they call). In particular, the **T2 traveling-shock paper run is
S=30 seeds** — `run_task2()`'s S=10 default is exploration only; use
`python reproduce.py t2` (which calls `studies/run_t2_S30.py`).

---

## Directory layout

| Path | Contents |
|------|----------|
| `reproduce.py` | Entry points for the paper runs + `verify` (no-rerun check against `expected_values.json`) |
| `expected_values.json` | Pinned expected values backing `reproduce.py verify` |
| `MANIFEST.md` | Manuscript table/figure → study → data mapping |
| `studies/study_t1_gbmc_dt_bias.py` … `studies/study_t5_fhn_extended.py` | Pre-publication studies T1–T5 |
| `studies/study_gbmc_production_n_refinement.py` | Production GBMC N-refinement (T6) |
| `studies/run_t2_S30.py`, `studies/run_prepublication_studies.py`, `studies/run_all_studies.py` | Wrapper / master runners (rest of `studies/`: earlier convergence/N-refinement runners) |
| `simulation.py`, `relaxation_gbmc.py` | Solvers: GRW dispatchers and the RB-GBMC Burgers scheme |
| `config.py`, `configs/`, `main.py` | Exploration CLI: run one simulation from a JSON config |
| `verify_solver.py`, `tests/verify_grw.py`, `tests/test_relaxation_gbmc.py` | Verification harnesses and unit tests |
| `tools/` | Manifest / paper-table / figure-fix generators |
| `output/final_prepublication_tests/` | Checked-in study outputs backing the manuscript |
| `output/convergence_study/` | Earlier exploration outputs (not used by the paper; see `RELEASE_NOTES.md`) |
| `regen_data/` | Corrected tanh-fit regeneration (Table 6 fitted columns, dt-study viscosities, three `_fixed` draft figures); see `regen_data/PROVENANCE.md` |

---

## Seed scheme

All ensemble studies use **base seed 42** with consecutive seeds per ensemble
member (seed_i = 42 + i), and the **same seed list is paired across N** (run
`i` at every N uses the same seed), so N-refinement comparisons are paired
rather than independent. The production T6 study uses S=50 (seeds 42–91);
T1 uses S=40; T2/T4/T5 use S=30; T3 uses S=10.

---

## Exploration CLI

Single simulations from JSON configs (interactive prompts if no config given):

```bash
python main.py configs/burgers_stationary_shock.json
python verify_solver.py --equation all      # solver verification harness
python tests/verify_grw.py                  # heat-only GRW checks
```

`config_template.jsonc` documents the config fields (strip `//` comments
before use).

---

## References

The GRW method builds on G. S. Lindstrom's TAMU thesis, available from the
Texas A&M OAKTrust repository, handle
[`1969.1/ETD-TAMU-1993-THESIS-L7533`](https://hdl.handle.net/1969.1/ETD-TAMU-1993-THESIS-L7533).
