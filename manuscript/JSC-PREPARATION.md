# JSC submission-format preparation

September 11, 2026. The scientific content has been transferred from the
custom preprint layout to the official Springer author class specified by JSC.
This is a prepared manuscript and source package, not a submitted article.

## Template and build

- Journal guidance: https://link.springer.com/journal/10915/submission-guidelines#Text
- Official package: https://media.springer.com/full/springer-instructions-for-authors-assets/zip/468198_LaTeX_DL_468198_240419.zip
- Class: `svjour3`, options `smallextended,nospthms`. The second option prevents
  duplicate theorem definitions while retaining the existing amsthm counters.
- Files `svjour3.cls`, `svglov3.clo`, and `spmpsci.bst` are unmodified copies
  from the official package. Bibliography uses its mathematics/physical-science
  style with numbered citations. Explicit author names replace natbib's
  textual-citation command, which this bibliography style does not supply.
- Native title, authors, affiliations, abstract, keywords, headings, captions,
  running headers, and page geometry replace the custom preprint styling.
- MSC 2020 codes: 65C05 (Monte Carlo methods), 65C35 (stochastic particle
  methods), checked against the AMS classification database.
- No receipt/acceptance dates or assigned manuscript number were invented.
  The manuscript-number placeholder is supplied by the class itself.

Build from the manuscript directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main-2-coupling.tex
```

## Layout and verification

Wide tables were reflowed with multiline headings and native text-sized
columns rather than scaled as images. The controls table retains every value
in separate Burgers/cubic panels. The provenance table can continue across
pages. Algorithms float within the available page space.

Figures were regenerated with the same archived data at the narrower journal
width. Field plots and held-out work comparisons now use vertically arranged
panels; captions and cross-references were updated accordingly. Text is
embedded as vector text, using sans-serif figure lettering. The standalone
bias plot is displayed larger to retain readable labels.

The converted PDF has 33 pages. The LaTeX build has no warnings, unresolved
references/citations, or overfull/underfull boxes. All pages were rendered and
visually inspected, with close checks of the algorithm, dense controls table,
and held-out work plot. The 79 printed-value checks and 62 independent
archived-evidence checks passed. No new scientific trajectory or timing
campaign was run, and the production solver remains unchanged.

## Author details still needed before submission

- Corresponding author: requested in the conversation; not guessed.
- Funding and competing-interest declarations: obtain the authors' accurate
  statements rather than inserting an unsupported 'none'.
- Substantive AI assistance: the journal distinguishes copyediting from
  assistance in content creation. This project's history includes assistance
  with code, analysis, and drafting, so the authors should confirm a factual
  disclosure of that use and their review responsibilities before submission.
  No statement falsely claiming completed human verification has been added.
- Reference-version audit: some existing entries cite preprints, including
  Nuesken--Pavliotis and Bencheikh--Jourdain. Prefer final published metadata
  where available, while checking that theorem/equation locators still match;
  this conversion has not silently changed the versions supporting those
  locators. Confirm the treatment of still-unpublished references against the
  journal guidance.

Existing author names, affiliations, and email addresses were retained.
The layout is prepared; these factual author details and reference-version
checks remain before an upload-ready submission can be certified.

## Preserved version

The preceding source and PDF are saved in:
`/Users/Stephen/Documents/GRW Methods/reviews/2026-09-11-paper2-jsc-template/before/`.
