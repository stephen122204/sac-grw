# Presentation aligned with BPC

September 11, 2026. Direct manuscript revision following the user's request.

## Published model consulted

Bertaglia, Pareschi and Caflisch, *Gradient-Based Monte Carlo Methods for
Relaxation Approximations of Hyperbolic Conservation Laws*, Journal of
Scientific Computing 100, 60 (2024).
https://doi.org/10.1007/s10915-024-02614-1

The published PDF was inspected visually, including displayed derivations,
algorithm boxes, Table 1, the scalar initial-data examples, and the appendix.
Its presentation retains defining functions and sampling rules in the main
text, places test initial conditions alongside numerical examples, uses
restrained heading/caption emphasis and ordinary table values, and puts
supporting reconstruction estimates in an appendix before the references.
These are presentation precedents, not evidence for our method's validity.

## Changes applied

- Black bold sans-serif headings, a mixed-case bold title without heavy title
  bars, a left-aligned abstract heading, and blue reference/citation links.
  These supersede the earlier blue-heading/pink-citation overlay.
- Bold figure/table labels with ordinary caption text. Removed bold winning
  table entries and rhetorical italics, preserving the formal definition.
- Combined the introductory background, closest predecessors, contribution,
  scope, and roadmap into one continuous introduction.
- Kept the PDE, representation, Gaussian law, two algorithms, switching rule,
  exact statements, quantitative comparisons and their limitations in the main
  text. Separated the paired algorithm's transport, sorting, and velocity
  update into explicit steps without changing their order or definition.
- Displayed the bump and oscillatory initial conditions beside their results.
  The numerical sign-count threshold and initialization details remain in
  Appendix C. An initial condition is part of the scientific test, not merely
  an implementation helper.
- Moved the supporting control-estimator formulas and training rule to the
  new Appendix D. The main text retains their meaning, role, comparison and
  computational limitations, with a direct appendix reference.
- Kept the RQMC rank/increment map in the main text; moved engine settings,
  scrambling details, clipping and implementation checks to Appendix D.2.
  Its distinct interacting law and separately measured bias remain explicit
  where its result is discussed.
- Moved final-stage identity verification particulars to Appendix A.4 while
  retaining the local-versus-full-history limitation in the main text.
- Placed references after the appendices. Statistics and evidence provenance
  are now Appendices E and F.

The existing preprint page geometry is retained. This is a BPC-inspired
presentation, not a claim to reproduce Springer's typeset article or to have
converted to the journal's submission class. No publisher logo, DOI, volume,
acceptance date, or publication status was added.

## Verification

- 79/79 printed-value checks passed.
- 62 independent archived-evidence checks passed.
- The LaTeX build has no warnings, undefined citations/references, or
  overfull/underfull boxes. All 22 rendered pages were visually inspected.
- Experimental arrays, figures, production solver, and original two-speed
  manuscript are unchanged. No new trajectory or timing study was run.

The pre-edit source and PDF are preserved at
`/Users/Stephen/Documents/GRW Methods/reviews/2026-09-11-paper2-bpc-style/before/`.
