# Release notes — Paper 2 (RB-GBMC) code release prep

Decisions recorded during release preparation (branch `paper2-release-prep`).

## License: pending

No LICENSE file yet. `CITATION.cff` carries `license: NOASSERTION` and the
README carries an HTML-comment `TODO(license)`. **Must be resolved before the
repository is made public.**

## Tracked exploration outputs: quarantine-or-exclude, NOT deleted

`output/convergence_study/**` (19 tracked files) are earlier *exploration*
outputs that do not back any manuscript artifact. They are **flagged for
quarantine or exclusion from the public release** but have deliberately
**not been deleted** here, to preserve history until a final call is made.
Everything the paper uses lives under `output/final_prepublication_tests/`
and `regen_data/` (see `MANIFEST.md`).

## Corrected fit-protocol study code lives in GRW-part2

The corrected tanh-fit protocol (scipy `curve_fit` throughout, fixing the
contaminated fallback fits in the original T6 production run) was implemented
and run in the **GRW-part2** repository at commit `fb18341`; that study code
has not been merged into this repository yet. **`regen_data/` carries the
regenerated data in the meantime** (bit-identical E-columns, corrected
fitted columns; see `regen_data/PROVENANCE.md`). Before or shortly after
publication, the corrected study code should be ported here so that
`reproduce.py t6` reproduces the corrected fits directly.

## Suggested tag

Tag the released commit as **`paper2-code-v1`**.

## Comment-cleanup guardrail

The comment cleanup performed on this branch is constrained to
**comments, docstrings, and whitespace only**: for every touched `.py` file,
`git diff` was inspected to contain no code changes and the file was
re-verified with `py_compile`. Every function keeps at least a 1–3 line
docstring summary; longer blocks were kept only where they document behavior
the code cannot express (e.g. BPC switching semantics, subcharacteristic
rationale, seed/RNG contracts, fit caveats). No study was rerun; no data
file was modified.
