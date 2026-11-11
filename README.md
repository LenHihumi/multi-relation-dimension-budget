# Multi-relation dimension budget: measurements and code

One PCA basis, several ways of defining relevance, one question: how many leading
coordinates does each relation need before its retrieval quality falls below a
stated tolerance, and when is a second graph worth its bytes?

This repository ships the measurements, the fitted parameters, the deployment
measurements and the code that produced and reduces them. It is a data package:
`bash repro.sh` regenerates the tables that are a function of the data, and nothing
in the repository contains an expected value to compare them against. It does not
read any manuscript and it asserts nothing about what a paper claims.

```bash
git clone <this repository> && cd <repository>
pip install -r requirements.txt
bash repro.sh            # python3 + numpy. No corpora, no GPU, no faiss, no network.
```

**Not shipped.** No images, no embeddings, no corpus file lists, no splits:
nothing from which a corpus could be reconstructed. The three corpora are
third-party and are not redistributable (§4 re-runs the pipeline from the original
downloads). `data/` holds derived statistics only.

## Layout

```
data/metric_std_parts/   19 cells x 23-31 ranks of mAP@10 (the core measurement)
data/grid/               19 cells x 2 cutoffs (recall@1, recall@10) x 18-31 ranks
data/mesh/               per-query AP@10 at every rank, 1000 queries per cell (npz)
data/theory/             eigenvalues and eta^2 per (encoder, corpus), 12 files
data/results/            every aggregate the reduction scripts quote, as JSON (43 files)
data/proj/               5 projeval JSON: PCA prefix against a projection trained on the relation
data/sensitivity/        two-point placement sweep, quantisation, group and label sweeps
data/fidelity/           per-query arrays for the query-count study (npz)
data/textbudget/         the text corpus and the deployment measurements
tables/                  20 CSVs/JSONs: 11 rebuilt by repro.sh, 9 measured (see §3)
scripts/                 105 Python scripts and 12 shell drivers
repro.sh                 one command: rebuild tables/ and check them against data/
requirements.txt         pinned dependency set, split by path (see section 5)
LICENSE                  CC BY-NC 4.0, this repository as a whole
scripts/LICENSE          CC BY 4.0, the code (see LICENSE_NOTE.md for why they differ)
```

## 1. What the numbers are

A **cell** is one (encoder, corpus, relation). Nineteen cells carry mAP@10; the
recall grid adds a second cutoff, for 38 curves.

Two truncation operators exist and they are not the same operation. `sub` keeps the
`r` leading coordinates of `X U_r`, so the mean component inside `span(U_r)`
survives. `cen` uses the PCA scores `(X - mu) U_r`, which removes the whole mean.
Both are L2-renormalised before scoring. Every curve says which one it used.

| quantity | definition |
|---|---|
| `fidelity@k(r)` | `|topk_r ∩ topk_full| / k`: agreement with the full-dimension ranking, label-free, ceiling 1 |
| `recall@k(r)` | `|topk_r ∩ R_q| / min(k, |R_q|)`: retrieval quality against the true relevant set |
| `rho(r)` | share of the cell's task energy that survives truncation at rank `r`: `1 - mean(squared loss) / total` |
| `Delta(r)` | `1 - rho(r)`, the share destroyed; the response surface is fitted on this |
| `r*(eps)` | the budget: `min r` with `curve(r) >= 1 - eps`, or, with `relative=True`, `>= (1 - eps) * curve(full)` |
| `bytes/vector` | `4 * r` at fp32 unless a table says otherwise |
| `p50` | single-threaded per-query latency in milliseconds, median over queries |

Fidelity always uses the absolute form because its ceiling is 1. Recall of a hard
task can have a ceiling well below 1, so its budget is only stated relative to that
cell's own full-dimension value. The two grids (`data/grid/` and
`data/metric_std_parts/`) are therefore not interchangeable, and a budget without
its metric, cutoff and epsilon is not a number.

The rank grid runs over `8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160,
192, 224, 258, 320, 384, 448, 512, 640, 768, 896, 1075, 1280, 1536, 1792, 2048,
3072, 4096`; a cell carries the subset its encoder's width allows, which is why
cells have 23 to 31 points and the largest measured rank can sit above the encoder's
nominal dimension once the mean is restored.

## 2. Data

| path | what it holds |
|---|---|
| `data/metric_std_parts/*.json` | one file per (encoder, corpus): mAP@10 at each rank, retention, `rho`, n_query, per-relation cells |
| `data/grid/*.json` | recall@1 and recall@10 at each rank, plus label-free fidelity |
| `data/theory/*.json` | eigenvalues, `eta^2` per direction, `rho0` and `s` of the fitted surface |
| `data/mesh/*.npz` | `grid` (ranks), `perq` (rank x query) AP@10, `full` (query) at full dimension |
| `data/results/*.json` | the aggregates: fit sensitivity, effective dimension, controlled bins, label-free rules, sequential arms, intervals, cross-cell margins |
| `data/proj/*.json` | PCA prefix against a relation-trained projection, five encoders, same tolerance |
| `data/sensitivity/*.json` | 688 admissible two-point placements, quantisation at 8/6/4 bits, group-count and label-noise sweeps |
| `data/fidelity/*.npz` | per-query agreement at several ranks and query subsamples, three cells |
| `data/textbudget/*.json` | the text corpus: the same grids, the multi-relation deployment, the mixed-workload measurements, build and latency repeats |
| `tables/*.csv` | the reduced views; `tables/` is the interface a reader quotes from |

Every JSON carries the parameters that produced it in a `meta`/`params` block:
encoder checkpoint, corpus size, relation definitions, epsilon, operator, and the
script that wrote it. No file contains a path, host address or user name from the
machine the study ran on.

## 3. What `repro.sh` does, and which tables it cannot rebuild

It is a build. `make_tables.py` writes the tables that are a function of `data/`,
`arms_by_branch.py` and `make_proj_table.py` the two that reduce those, and the two
figure scripts redraw the figures when matplotlib is present. No script compares a
result against an expected value: the evidence that the tables are the data's image
is that a run **leaves the working tree unchanged**, which CI asserts on every push
(`.github/workflows/repro.yml`). Where a table and the JSON ever disagreed, the
rebuild would rewrite the committed file and the diff would be the report.

| rebuilt from `data/` (11) | measured, needs a corpus (9) |
|---|---|
| `map10_19cells.csv`, `recall38_cells.csv`, `fits_19cells.csv`, `rules_table.csv`, `rules_table_recall10.csv`, `lodo_arms.csv`, `two_point_placement.csv`, `arms_by_branch.csv`, `arms_bootstrap.csv`, `proj_budget.csv`, `FIG2_response_law.json` | `mix_workload.csv`, `crosscell_pairs.csv`, `coverage_bytes.csv`, `quant_bits.csv`, `group_sensitivity.csv`, `perquery_steepness.csv`, `probe_cells.csv`, `supervised_reg.json`, `tune_baselines.json` |

The nine on the right are reduced from run output by `sb_mix_p50.py`,
`sb_crosscell.py`, `sb_quant_budget.py`, `review_tables.py`,
`make_sensitivity_tables.py`, `perquery_steepness.py` and `sb_probe.py`; regenerate
them by re-running those scripts against the corpora (§4), or take the shipped files
as the measurements they are.

The assertion suites that hold this project's expected values -- the numeric
cross-check, the table attestation, the closure audit -- are not part of this
repository. They are the answer key, and this package publishes data.

## 4. Re-running the pipeline from scratch

The corpora and the embeddings are third-party and large, so neither is shipped.
Everything downstream of them is. Each stage writes JSON that the next one reads, and
each script is idempotent (it checks for its output before doing work).

**Paths.** Every script derives its directories from its own location and can be
redirected by environment. `SB_ROOT` is the working root (corpora, caches, outputs),
`SB_OUT` the run directory, `SB_CACHE` where embeddings live, `SB_MODELS` where model
weights live:

```bash
export SB_ROOT=$HOME/sb
export SB_CACHE=$SB_ROOT/.cache/sb
export SB_MODELS=$HOME/models
```

**Stage 0 -- corpus to gallery/queries.** `sb_build_data.py` builds the three image
galleries. In-shop keeps every row of its parquet (52,591); SOP keeps the first four
images per item over 22,634 items (78,227 of the release's 120,053); the
Oxford/Paris and ILIAS galleries are built the same way.
`sb_build_data_instre.py` builds INSTRE as the first forty images of each of 200
classes (8,000 of 28,543), with a fixed keyword table mapping the 200 directory names
to 24 categories -- both caps put every gallery in the same 10^4--10^5 encoding
bracket, and they were set before any budget was measured (`--per-obj` / `per_item`
are the knobs). `sb_text_prep.py` / `sb_text_embed.py` do the same for a text corpus
against any OpenAI-compatible embedding endpoint; `imagenet_prep.py` +
`imagenet_encode.py` + `merge_emb.py` build the scale sweep.

**Stage 1 -- the grids.** `sb_grid.py --encoder E --dataset D` writes the fidelity
curve per cell; `sb_metric_std.py --pair E D` rescores the same runs with mAP@10,
nDCG@10 and recall@10; `sb_retrieval.py` adds the exact-search baseline;
`sb_perquery.py` dumps per-query AP@10.

**Stage 2 -- fits and mechanisms.** `sb_sigmoid.py` fits
`Delta = 1/(1 + (rho0/rho)^s)` per cell; `sb_effdim.py` tests the form (logit
linearity, curvature) and the mechanism (effective dimension, task energy);
`sb_controlled.py` bins the confounders; `sb_shrinkage.py`, `sb_eta_stability.py` and
`sb_sdeff.py` run the stability and shape controls.

**Stage 3 -- the rules.** `sb_2pt_sensitivity.py` sweeps every admissible two-point
placement (253 rank pairs, 435 rho-target pairs); `sb_probe.py` runs the
three-branch rule cell by cell; `sb_bisect.py` the bracket-bisection alternative;
`sb_acceptance.py` / `sb_wrapup.py` the capped sampler and the tables;
`sb_sequential.py` the sequential arms.

**Stage 4 -- predictors.** `sb_learned.py` (ridge on spectral features),
`sb_labelpred.py` (the label-aware sibling), `sb_lodo_true.py`
(leave-one-corpus-out), `sb_tune_baselines.py` (the 55-configuration budget).

**Stage 5 -- the index.** `sb_ann.py` builds the HNSW graphs (needs `faiss-cpu`);
`shared_index.py` measures one graph per cell; `sb_mix_p50.py` and `sb_mix_image.py`
the mixed workloads; `sb_multi_e2e.py` / `sb_multi3.py` the multi-relation
deployments; `sb_ann_scale.py` the 10^4--10^6 sweep.

**Stage 6 -- the projection.** `proj_rank.py` trains the relation projection (nested
InfoNCE, 200 Adam steps, half the entities held out); `proj_eval.py` re-evaluates it
against the PCA prefix with a separate implementation and refuses to report unless
its PCA column reproduces the measured budget; `make_proj_table.py` reduces the five
JSON to `tables/proj_budget.csv`.

**Stage 7 -- tables and figures.** What `repro.sh` runs, and it needs no corpus:
`make_tables.py`, `arms_by_branch.py`, `make_proj_table.py`, the two figure
scripts.

Total CPU time on 128 cores: about 6 minutes for the mAP grid, 40 minutes for the
full recall grid, 2 minutes for everything else.

## 5. Environment

| Path | Needs |
|---|---|
| `bash repro.sh` | python3 (3.10--3.13: CI runs 3.10 and 3.12, measured on 3.12.3) and numpy (pinned 2.2.6; 2.5.4 also reproduces the tables byte for byte) |
| `scripts/sb_ann.py` | above, plus faiss-cpu 1.15.1, plus the embedding caches |
| `scripts/sb_wrapup.py --mode figures` | above, plus matplotlib |

Pinned set: `requirements.txt`; only the first group is required, the other two are
commented out because they need data this repository does not contain. No GPU is
required -- the measurements used CPU BLAS with 24 threads; the exception is
`imagenet_encode.py`, which needs torch and a GPU. Matplotlib is needed only to
rebuild the two figures, and their inputs are checked either way, so a run without it
still verifies every number.

## 6. Caveats the data records

Four traps that cost a re-run each, recorded here because they change how a column
may be read.

1. **The `p50` of a mixture is not the mean of the components.**
   `w * p50(item) + (1 - w) * p50(cat)` is an interpolation, and at a 90/10 mix it is
   off by a factor of about 1.5. `tables/mix_workload.csv` holds the measured
   sequence medians, and the interpolation is not shipped as if it were a
   measurement.
2. **`tables/group_sensitivity.csv` reads as a shape, not as budgets.** Its absolute
   ranks sit at about half of `fits_19cells.csv` because the sweep rebuilds the
   relevance sets and query sample locally rather than reusing the pipeline's. The
   trend is what it supports.
3. **`tables/quant_bits.csv`'s headline is a grid-step fraction, not coverage.** It
   counts how many cells move by at most one grid position out of 34, which is a
   different quantity from the covered fraction of a target.
4. **Per-query budgets are not resolvable at this sample size.** Per-query fits are
   flatter than the cell fit on 19 of 19 cells (median ratio 0.76) and their position
   spread reaches 33x within one cell, mostly because a relevance set of median size
   3 admits only four distinct AP values. A budget is stated per cell because that is
   the finest unit the data supports.

## 7. Synthetic control

`scripts/sb_toy.py` generates a Gaussian corpus with two planted partitions and
writes `data/results/toy_synthetic.json`: one spectrum, two budgets a factor of five
apart, with no third-party data at all. It is the smallest reproduction of the
central phenomenon, and the only one that runs without a corpus download.

## 8. Licence

Code is CC BY 4.0 (`scripts/LICENSE`); the measured data, tables and figures are
CC BY-NC 4.0 (`LICENSE`), because the DeepFashion release agreement restricts derived
data to non-commercial use. `LICENSE_NOTE.md` gives the per-corpus reasoning and
names the one decision the corresponding author has to confirm.

## 9. Releasing this deposit

`make_release.sh` builds the Zenodo zip. It stages the file list into a temporary
directory and runs `repro.sh` **there**, not in this tree: an artifact that does not
reproduce as packaged is not packaged.

```
./make_release.sh v1        # stages, runs repro.sh on the stage, writes ../artifact_v1.zip
```

It refuses if any of these is still a placeholder, because each one is a claim the
DOI would make on the authors' behalf:

| placeholder | where | what it is |
|---|---|---|
| `FILL:` in creators | `.zenodo.json` | author names, ORCIDs, affiliations |
| `FILL:` in related_identifiers | `.zenodo.json` | the paper's DOI, if it has one at upload time |
| `zenodo.XXXXXXX` | `../sections/01_intro.tex` (`\artifactdoi`) | the concept DOI Zenodo assigns |

Order of operations, because the DOI and the paper are circular:

1. upload the zip as a **new version** and let Zenodo *reserve* the DOI;
2. paste the reserved **concept** DOI into `\artifactdoi`, rebuild the paper;
3. publish the deposit and confirm the DOI resolves to the version you built;
4. for later edits upload a new version: the concept DOI keeps resolving, which is
   why the paper gets the concept DOI and not the version DOI.
