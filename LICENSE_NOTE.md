# What is in this artifact, and on what terms

Two licences apply, because the deposit mixes our own work with statistics
derived from three third-party corpora.

| Part | Contents | Licence |
|---|---|---|
| `scripts/` | the 105 measurement and reduction scripts | CC BY 4.0 (`scripts/LICENSE`) |
| `data/`, `tables/`, `figs/` | measured curves, fitted parameters, aggregates, figures | CC BY-NC 4.0 |
| this repository, as a whole | — | CC BY-NC 4.0 (`LICENSE`) |

The code is offered more permissively than the data because we hold the
copyright on the code and do not hold it on the corpora. Re-users who only
want the analysis code can take it under CC BY 4.0.

## Per-corpus terms

`data/` holds **derived measurements only**: per-cell retrieval curves at each
truncation rank, the eigenvalues and `eta^2` of each corpus, and the fitted
parameters. There are no images, no embedding vectors, no file lists and no
splits — nothing from which a corpus could be reconstructed. That is still not
the same as being unrestricted: the three corpora have different terms, and one
of them names derived data explicitly.

| Corpus | Released by | Terms at release | Files shipped | How we treat derived statistics |
|---|---|---|---|---|
| In-shop Clothes Retrieval (DeepFashion) | CUHK MMLAB; Liu et al., CVPR 2016 | Signed Release Agreement ([text](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/DeepFashionAgreement.pdf)): (1) non-commercial research use only; (3) agree not to "reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data"; (4) agree not to "further copy, publish or distribute any portion of the DeepFashion dataset" | 5 of 11 `metric_std_parts` files (CLIP, DINOv3, Qwen3-VL, SigLIP2, WEMM) | Clause 3 names "derived data" inside its non-commercial restriction. Twenty-three aggregate numbers per cell cannot reconstruct an image, but the clause does not carve aggregates out, so we ship them **NC**, not CC BY. This is the reason the whole deposit is NC. |
| Stanford Online Products | Song et al., CVPR 2016; distributed from the `Deep-Metric-Learning-CVPR16` repository | The repository README states the MIT licence, and the dataset is distributed with it | 3 files (DINOv3, SigLIP2, WEMM) | No restriction on derived statistics. |
| INSTRE | Wang & Jiang, *ACM TOMM* 11(3), 2015; ICT CAS VIPL ([resource page](https://vipl.ict.ac.cn/homepage/jsqgrzy/zy/)) | No formal licence published with the release; mirrors (e.g. OpenDataLab) state that unauthorised bulk redistribution of the dataset is prohibited | 3 files (CLIP, DINOv3, SigLIP2) | Redistribution of the dataset is prohibited; derived statistics are not addressed either way. Treated as research-only, same as In-shop. |

The three corpora are not redistributed in any form. Anyone re-running the
pipeline obtains them from the original sources and re-embeds them with the
frozen checkpoints listed in the paper.

## The decision that belongs to the corresponding author

If clause 3 of the DeepFashion agreement is read as covering aggregate
statistics, the coherent response is to withhold
`data/metric_std_parts/*{inshop,instre}*.json`, the matching
`data/theory/*{inshop,instre}*.json`, and the affected rows of `tables/`.
The cost of that fallback is now bounded, and this is what the synthetic control
in `scripts/sb_toy.py` exists for. Three things survive the strictest reading:
the code, the SOP-derived files (MIT), and the synthetic corpus, which reproduces
the central phenomenon — one spectrum, two budgets a factor of five apart — with
no third-party data at all. What is lost is bounded: the In-shop and INSTRE
curves would no longer be recomputable from this deposit, and the synthetic
control would be the only path to the central measurement. The alternative — one
permission request to MMLAB — preserves both. We ship under NC and record the reasoning here rather
than leave the choice implicit.

## What would make this note wrong

If a corpus's terms change, or if a mirror's statement is not the release
statement, the row above is wrong and the affected files must be pulled. The
scripts in `scripts/` are sufficient for a reader to regenerate every number
locally, so withholding files costs reproducibility, not correctness.
