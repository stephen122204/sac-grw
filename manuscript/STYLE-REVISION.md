# Writing and attribution revision

Historical record of the first prose pass. The subsequent
[BPC presentation pass](BPC-STYLE-REVISION.md) supersedes its color and
section-placement choices; the initialization correction remains in force.

September 11, 2026. This pass applies the supplied editing prompts and the
error-attribution paper's prose conventions to the coupled-estimation paper.
The title and scientific subject remain unchanged.

## Changes

- Rewrote the abstract around the problem, construction, findings, and limits.
  It has 206 words and leaves detailed numerical intervals in the results.
- Added citations beside the signed-gradient representation, diffusion update,
  reflection/synchronization rule, Gaussian law-preservation argument,
  conditional expectation, control variates, and bootstrap procedure.
- Separated long formulas into displays and supplied intermediate steps for
  the differentiated equation, cumulative-field product identity, conditional
  field, control means and coefficient, initialization, and Cole--Hopf reference.
- Distinguished single trajectories, paired replicates, complete evaluation
  blocks, integrated field variance, and squared bias. The statistical appendix
  now states the sampling correction, resampling unit, and held-out upper bound.
- Removed internal-review language, rhetorical qualifications, repeated
  conclusions, and commentary inside the construction/geometry/bias plots.
  Captions retain the explanations. Plot data and experiments are unchanged.
- Combined discussion and conclusions. Preserved the limitations of the
  final-stage theorem, selected-plan work comparison, and numerical scope.
- Used blue section headings and pink citations. Added protection against
  section headings being stranded at the bottom of a page.

## Source attribution at the point of use

| Material | Attribution or derivation |
| --- | --- |
| Signed-gradient representation | Ghoniem–Sherman, Roberts, Lécot, Section 2.1 |
| Gaussian diffusion increment | Ghoniem–Sherman, Section 2.2 |
| Reflection/synchronization and sign-dependent switches | Hammersley–Morton and Nüsken–Pavliotis, Section 2.3 |
| Adapted transformations preserving Gaussian noise | Nüsken–Pavliotis, Appendix A, cited in Section 3.1; discrete proof supplied here |
| Conditional expectation and control variates | Glasserman, Chapter 4, Section 4.2; estimator formulas derived here |
| Whole-realization bootstrap | Efron (1981), Sections 4.1 and Appendix D |
| Sorted matching for absolute distance | Villani, Section 3.5 |
| Quasi-random diffusion adaptation | Lécot, equation (45), Section 4.4.1 |
| Cole–Hopf reference | Hopf and Cole, Appendix C |

The Nüsken–Pavliotis primary text, the locally saved Lécot RIMS paper,
Efron's original article, and the Glasserman chapter listing/abstract were
consulted for the added attribution. The existing bibliography metadata was
retained. These citations do not attribute the present baseline's exact
transport convention or its numerical findings to those predecessors.

## Initialization qualification found during the writing check

The main two-pulse initializer enforces equal mass magnitudes across signs.
The profile/time initializer instead samples the derivative on 200,001 points
in [-6,6], forms separate cumulative rectangle sums, and assigns each class
its own mass. The previous blanket equal-magnitude description was too broad.
For the oscillatory profile at N=2048, its total signed mass is
-2.339898222125658e-7. The manuscript now states both constructions and keeps
the zero-total-mass theorem separate from these empirical profile comparisons.
This is a correction to the description, not a change to solver data or results.
The direct initializer outputs are saved in `evidence/style-initialization.json`.

## Checks and scope

- 79 printed-value consistency checks passed after the revision.
- 62 independent archived-evidence checks passed, including held-out block
  errors, uncertainty bounds, field statistics, and figure-state rank ordering.
- The coefficient formula was checked against `study_round09_costs_control.py`.
  Initializers were checked against `twopulse_reference.py` and
  `study_round19_transfer.py`.
- All figures were rebuilt. No trajectory study or timing campaign was rerun.
- The final PDF has 22 pages. The build has no undefined references/citations,
  overfull/underfull boxes, or LaTeX warnings. Rendered pages were inspected.
- A scoped prose scan excluding algorithms, table cells, and LaTeX identifiers
  found none of the checked retired expressions (`headline`, `buys`,
  `in the sense of`, `output grid`, `geometric slack`, and bare `resolution`).
  Mathematical punctuation and algorithm separators are retained.

The source and PDF before this pass are saved under
`/Users/Stephen/Documents/GRW Methods/reviews/2026-09-11-paper2-style-pass/before/`.
This pass does not certify publication acceptance or establish new nonlinear
variance guarantees. Journal-template conversion remains a separate task.
