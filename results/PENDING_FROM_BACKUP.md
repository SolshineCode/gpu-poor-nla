# Artifacts the paper cites that are not yet in this repository

Written 2026-09-02. On 2026-08-24 the source repository moved its bulk NLA results off
GitHub to a private provenance backup on Hugging Face
(`DarkStarDeleeuw/nla-research-bulk-provenance`, a dataset repo) and stopped tracking
`experiments/v8_nla_local/results/`. The round-2 and follow-up artifacts below were
therefore never copied here. `verify_claims.py` reads this table and reports each
absent file as a FAIL, so the public check cannot pass until they land.

Copy procedure (needs a login with read access to the backup; run from the source repo
on the homelab, where local copies also exist):

```
hf download DarkStarDeleeuw/nla-research-bulk-provenance --repo-type dataset \
  --include "experiments/v8_nla_local/results/nla_local/wco_use3/*" \
           "experiments/v8_nla_local/results/nla_local/causal_t2/t2_summary_*" \
           "experiments/v8_nla_local/results/content_aware_eval/*" --local-dir /tmp/prov
# then cp the files listed below into results/<destination> and run python verify_claims.py
```

Source paths are relative to the backup's mirror of the source repository; destinations
are relative to this repository's `results/`. "Unblocks" names the paper claim.

| Destination in `results/` | Source path on the backup | Unblocks |
|---|---|---|
| `dense1000_semantic_use3_K4_n580_fixedinj.json` | `experiments/v8_nla_local/results/nla_local/wco_use3/dense1000_semantic_use3_K4_n580_fixedinj.json` | round-2 continuation to 1000 steps, top-1 7x |
| `dense1000_tfidf_use3_K4_n580_fixedinj.json` | `.../wco_use3/dense1000_tfidf_use3_K4_n580_fixedinj.json` | round-2 lexical channel |
| `dense1000_outs_n580_fixedinj_ranksupp.json` | `.../wco_use3/dense1000_outs_n580_fixedinj_ranksupp.json` | mean own-document percentile 0.271 |
| `marginft200_semantic_use3_K4_n580_fixedinj.json` | `.../wco_use3/marginft200_semantic_use3_K4_n580_fixedinj.json` | warm-start worst-case-margin arm (fails its gates) |
| `marginft200_outs_n580_fixedinj_ranksupp.json` | `.../wco_use3/marginft200_outs_n580_fixedinj_ranksupp.json` | same, rank level |
| `paired_dense1000_outs_n580_fixedinj_vs_dense500_outs_n580_fixedinj.json` | `.../wco_use3/paired_dense1000_outs_n580_fixedinj_vs_dense500_outs_n580_fixedinj.json` | paired Wilcoxon 0.343 to 0.271, p = 1.5e-7 |
| `paired_marginft200_outs_n580_fixedinj_vs_dense500_outs_n580_fixedinj.json` | `.../wco_use3/paired_marginft200_outs_n580_fixedinj_vs_dense500_outs_n580_fixedinj.json` | margin arm regression |
| `paired_dense500_vs_step_005000_pure_ood_n236.json` | `.../wco_use3/paired_dense500_vs_step_005000_pure_ood_n236.json` | pure-OOD 236-row test (semantic p = 0.048, tf-idf p = 0.080) |
| `paired_dense500_outs_n295_fixedinj_vs_step_005000_outs_n295_fixedinj.json` | `.../wco_use3/paired_dense500_outs_n295_fixedinj_vs_step_005000_outs_n295_fixedinj.json` | pooled OOD paired test (p = 0.017 semantic, 0.030 tf-idf) |
| `format_matched_rescore_n580.json` | `.../wco_use3/format_matched_rescore_n580.json` | output-format control (0.349 vs 0.409, p = 2.3e-5) |
| `judge2afc_base_vs_dense500_n150_verdict.json` | `.../wco_use3/judge2afc_base_vs_dense500_n150_verdict.json` | judge grown to n = 150 |
| `judge2afc_base_vs_dense1000_verdict.json` | `.../wco_use3/judge2afc_base_vs_dense1000_verdict.json` | 88.1% win rate, p < 1e-15 |
| `judge2afc_dense500_vs_dense1000_verdict.json` | `.../wco_use3/judge2afc_dense500_vs_dense1000_verdict.json` | 88.0% win rate |
| `judge2afc_base_vs_dense500_HAIKUCONTROL_verdict.json` | `.../wco_use3/judge2afc_base_vs_dense500_HAIKUCONTROL_verdict.json` | judge-model dependence control (null) |
| `t2_summary_base5000_trace_n48_swap_vs_dense1000_trace_n48_swap.json` | `experiments/v8_nla_local/results/nla_local/causal_t2/t2_summary_base5000_trace_n48_swap_vs_dense1000_trace_n48_swap.json` | causal tracing gap 2.44 to 3.96 |
| `round_trip_v0_n50.json` | `experiments/v8_nla_local/results/round_trip_v0_n50.json` | round-trip gate 0.438 over 42 rows |
| `round_trip_null_floor_n50.json` | `experiments/v8_nla_local/results/round_trip_null_floor_n50.json` | content-blind mean-vector reconstructor 0.513 |
| `fc2afc_relabel_powered_eval_n580.json` | `experiments/v8_nla_local/results/content_aware_eval/fc2afc_relabel_powered_eval_n580.json` | base pair rescored under relabeled eval text: 0.529 to 0.557, McNemar p = 0.24 |
| `neuronpedia_comparison_dense_grpo_base.json` | `experiments/v8_nla_local/results/content_aware_eval/neuronpedia_comparison_dense_grpo_base.json` | anchoring vs the reference NLA (0.0/3 vs 0.58–0.83/3) |
| `neuronpedia_comparison_dense_grpo_dense500.json` | `.../content_aware_eval/neuronpedia_comparison_dense_grpo_dense500.json` | same |
| `neuronpedia_comparison_dense_grpo_dense1000.json` | `.../content_aware_eval/neuronpedia_comparison_dense_grpo_dense1000.json` | same |

Exact filenames were taken from the paper's artifact list; if a name differs on the
backup, copy the file that carries the claim and correct the row here (do not delete
rows, strike them through with the date).
