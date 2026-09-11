# Direct manuscript revision and scientific audit

September 10, 2026. The subject is **Sign-Aware Antithetic Coupling for Signed
Gradient-Particle Methods**, on the mean-transport baseline. The earlier
compensated two-speed manuscript remains a separate historical artifact.

## Scientific decision

The manuscript now leads with the additional field-estimation benefit over
within-sign pairing. Reflection/synchronization, preservation of marginals,
sorted matching, and gradient reconstruction have established predecessors.
The contribution is their specified realization and evaluation for the signed
transient field, with an exact conditional final-stage identity and explicit
limits. The paper does not claim invention of antithetic switching, optimal
matching for variance, or universal nonlinear terminal improvement.

A reader who already uses this solver can implement the coupling and evaluate
its additional work/error benefit. The main comparison gives variance ratio
0.7389 and empirical MSE ratio 0.8193 against within-sign pairing. One separate
held-out target is attained with four versus six trajectories at a measured
work ratio of 0.5986. This is a selected-plan result, not optimized equal-error
performance or a general complexity improvement. The one-sign equality,
weak-benefit cases, bias-dominated outputs, and scalar one-dimensional scope
remain visible. Publication significance is a reviewer judgment, not certified
by test counts or this audit.

## Corrections that changed the manuscript

| Issue | Correction |
|---|---|
| Post-diffusion state described as sorted | State chronology now distinguishes pre-diffusion spatial rank from post-diffusion storage. The carried velocity remains associated until the next transport. |
| Construction figure treated terminal indices as ranks | Rebuilt from a captured sorted pre-diffusion state, with sortedness checks and archived coordinates. |
| Marginal preservation used to imply unrelated convergence rates | Removed the imported-rate claim. Only the baseline's law and expectations are preserved; nonlinear functions of a pair mean need not preserve expectation. |
| Off-diagonal covariance independence overstated | Restricted to the two rank-matched rules in the final-stage proof. |
| `guaranteed_share` interpreted as a bound on full-history gain | Explained as a descriptive ratio of estimates. The exact comparison is RAW versus switching only its final stage. |
| Monotonicity example reported an exact expectation change of 0.5 | Added a self-contained zero-noise trace and Gaussian-tail argument; corrected transport displacement. The strict inequality, not equality to 0.5, is proved. |
| Conditional evaluation described as replacing noise by its mean | Corrected to the Gaussian CDF expectation of the terminal indicator field; added formulas for both known-mean controls and coefficient training. |
| Pilot allocation equation did not match the drivers | Corrected to bias_hi + variance_hi/B <= target/1.3. RQMC uses its own bias estimate. |
| Bias plot used equal pair counts on a work axis | Rebuilt as an explicit equal-continuous-work plug-in model, with shared estimated bias and separate costs. |
| Global seed/bootstrap statements exceeded the archives | Separated independent versus shared streams and per-experiment resample counts. |
| Scalar coverage calibration generalized to MSE/field intervals | Restricted its interpretation; nominal bootstrap and t intervals remain approximate. |
| Different batches blended in the spatial explanation | Removed the extra batch's 0.117/quintile figures from the headline field discussion. |
| Rounded historical costs used as source precision | Use original held-out costs for historical metadata; current work uses the interleaved timing check. |
| Prior-art scope and incomplete bibliographic authors | Corrected Nüsken–Pavliotis matching scope, supplied Netterdon et al. names, and included the recent McKean–Vlasov QMC construction with limited claims. |
| Internal review narrative dominated the article | Moved development history to the research archive; rewrote abstract, introduction, discussion and conclusion around the final contribution. |

## Verification performed

- Recomputed integrated variances and empirical MSEs directly from float64 partner
  fields; the central ratios survive unchanged.
- Independently recomputed both held-out block means, standard errors, nominal
  Bonferroni bounds and trajectory counts. All reported bounds remain below the
  corresponding targets.
- Compared the extracted shared core against its saved original function in 24
  cells: N=12, 200, 8192; Burgers/cubic; RAW/FULL/WITHIN/FINAL; 200 steps. Returned
  positions and masses match exactly. This is a refactoring equivalence check.
- Separately compared both replicas against the original complete-state solver
  under prescribed noise, including mixed signs, ties, and velocity association.
- Checked the final-stage formula through independent indicator-covariance
  quadrature, including unequal mass magnitudes and separated particles.
- Recomputed the Burgers spectral reference: zero difference from the archived
  reference; maximum difference 5.5664e-14 from Cole–Hopf. The Cole–Hopf routine
  emitted an IntegrationWarning. This agreement is a numerical cross-check,
  not a certified quadrature error bound; the warning is recorded in evidence.
- 53 tests pass, including six coupling/option tests; all 164 legacy numerical
  checks pass. The latter concern historical studies, not the new contribution.
- 62 independent evidence checks and 79 printed-value checks pass. The printed
  scan is only a consistency aid, not an independent scientific validation.
- All five figures are regenerated from data or captured states. The PDF is
  compiled and visually checked, with no undefined citations/references or
  overfull boxes in the final log.

Original timing experiments were not rerun to select more favorable numbers.
No new benchmark campaign, inverse-paper edits, submission, or external messages
were performed. Existing historical study outputs are retained, not silently
rewritten when a claim is corrected.

## Source check and limits

Primary sources inspected for the decisive positioning include:

- Nüsken–Pavliotis, [arXiv:1806.11026](https://arxiv.org/abs/1806.11026):
  Proposition 54, special cases, Remark 55, and repeated matching in Section 5.2.
- Lécot, [RIMS 1240, 103–113](https://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/1240-11.pdf):
  signed gradient representation and reordered quasi-random diffusion; the
  comparator remains an adaptation, not a reproduction of its transport.
- Bertaglia–Pareschi–Caflisch, [JSC 100, 60](https://link.springer.com/article/10.1007/s10915-024-02614-1):
  gradient-based relaxation methods, including systems. This provides context,
  not a theorem for our method or evidence of acceptance.
- Ben Rached et al., [arXiv:2409.09821](https://arxiv.org/abs/2409.09821):
  a distinct QMC interacting-particle construction and antithetic multilevel
  estimator; weak convergence under additive-noise assumptions.
- Netterdon et al., [arXiv:2601.14461](https://arxiv.org/abs/2601.14461):
  Fokker–Planck array-RQMC context. We make no family-wide superiority claim.

Crossref metadata was checked where available; several queries were rate-limited,
recorded in `evidence/bibliography-metadata.json`. The pre-existing bibliography
was not replaced by guesses when access failed. The Lécot–Ogawa 2002 companion
is not represented as a full-text reproduction. This is a focused claim audit,
not an exhaustive proof of priority across all particle literatures.

The remaining limitations are the paper's scientific scope: no full-history
variance ordering, no optimized work-to-tolerance theorem, no generalization to
systems/higher dimensions, and no guarantee that approximate intervals attain
nominal coverage. These are stated rather than converted into new unsupported
headline mechanisms.

## Disposition of the collaborator handoff

The missing full-history inequality, isolated causal mechanism, fine integer
allocation, combined five-arm benchmark, broader transfer, and parallel timings
are explicit limitations/possible future research, not results this revision
pretends to establish. A stricter target, not a looser one, would generally
require more replicates and reduce allocation granularity.

The orphan `nonlinear_mse_corrected.json` remains unchanged but is prominently
excluded by a README in its own directory. The missing historical residual is
not cited. The scalar calibration does not justify a claim that all bootstrap
intervals under-cover. The new tests distinguish replay equivalence from an
independent complete-state comparison and indicator-level mathematical check.
The uncited Rubinstein entry is not relied on for an optimality claim; the text
uses a standard monotonicity sufficient condition and a self-contained
counterexample. Sadr–Hadjiconstantinou remains outside the relied-upon evidence.
The two Lécot publications remain distinct. The exploratory predictor is removed
from the article. The retained preprint format is for review; journal-specific
submission formatting can be performed once the submission destination is fixed.
