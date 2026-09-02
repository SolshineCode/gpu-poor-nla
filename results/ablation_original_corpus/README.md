# results/ablation_original_corpus — the matched control for the relabeled-corpus run

Added 2026-09-02. Question answered here: is the large content-reading jump in
`../relabel_v1/` caused by the corrected labels, or by the retraining pipeline itself
(fresh LoRA initialisation, freshly computed prior-deviation weights, fp32 training on
different hardware)? To find out, the *original* training corpus was retrained through
the identical pipeline with everything else held fixed, in two independent training
seeds, and scored on the same 580 held-out documents at the same steps.

## Files

| File | What it is |
|---|---|
| `SUMMARY.json` | Summary of all runs: n, unique outputs, top-1, chance, worst-case p. |
| `baseline_n580_{tfidf,semantic}.json` | The paper's published base decoder (step 5000), rescored under the same code for a same-file anchor. Matches `../step_005000_*_use3_K4_n580_fixedinj.json`. |
| `seed{0,1}_step0{05000,10000,14000}_outs_n580_{tfidf,semantic}_use3_K4.json` | Exact-document top-1 among 580 (`obs_doc_top1`, chance 1/580) with a 4-seed permutation null and its worst case (`worst_case.max_p_value`). |
| `seed{0,1}_step014000_shuffle_outs_n580_*` | Shuffled-activation control at the endpoint. |
| `seed{0,1}_step014000_noinj_outs_n24_*` | No-injection control (24 documents, chance 1/24). |
| `*_intradomain_diag.json` | Within-domain diagnostics, raw. |
| `smoke_n120/` | The earlier 120-document smoke-scale pass of the same comparison (`ablation/` for these two seeds, `relabel_v1/` for the relabeled run and its baseline anchor, with `subset_indices.json` naming the 120 rows). Kept because the full-scale result corrected one of its readings; see below. |

Generated descriptions (pickles) live in the source repository under
`experiments/v8_nla_local/results/nla_local/f194_ablation_eval_n580_20260829/` and
`.../f194_ablation_eval_20260828/`.

## What it shows, stated carefully

- The tf-idf channel is null throughout: 1–2x chance at every step for both seeds, every
  worst-case p above 0.26, baseline included.
- The semantic channel is not a perfectly clean null at full power. Seed 1 grows from 2x
  to a significant 7x chance by step 10000 (p = 0.0003) and holds it at 14000
  (p = 0.0010); its shuffle control falls back to 1x. Seed 0 does not replicate this: 4x
  at steps 5000 and 10000, receding to 2x (p = 0.27) by 14000. At the 120-document
  smoke scale neither seed had cleared significance, which is why the full-scale rerun
  is kept alongside it.
- Under worst-case-across-seeds reporting the ablation's anchoring number is seed 0's
  2x at step 14000. Its best case anywhere is seed 1's 7x. The relabeled run's *lowest*
  number at its *earliest* checkpoint (141x, step 5000, semantic) is about 20x larger
  than that best case, and its step-14000 numbers (273–314x) are roughly two orders of
  magnitude larger.

Conclusion: the pipeline is not inert, but it accounts for at most a 7x effect in one
seed; the corpus change accounts for the rest.
