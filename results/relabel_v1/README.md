# results/relabel_v1 — the relabeled-corpus activation verbalizer, evaluated on the full 580-document held-out set

Added 2026-09-02. These are the evaluation outputs for an activation verbalizer trained
with exactly the same long-horizon recipe as the paper's supervised base decoder
(prior-deviation cross-entropy, LoRA rank 80, weight decay 0.3, learning rate 5e-5),
but on a training corpus whose labels were rewritten from short topic summaries into
feature-attribution text ("what must the model have integrated at this point to predict
the next token"). One training seed. Checkpoints at steps 5000, 10000 and 14000.

Every file here was produced by the same scoring code as the round-1 results in the
parent folder, on the same 580 held-out documents, so the numbers are directly
comparable to `../step_005000_*` (base) and `../dense500*` (the GRPO-improved model).

## Files

| File | What it is |
|---|---|
| `SUMMARY.json` | One-page summary of every run below: n, unique-output count, top-1 rate, chance, worst-case p. Its `reading` prose rounds chance to 0.0017 and so quotes multiples a few percent higher than the exact 1/580 used in the README and in `verify_claims.py`. |
| `relabel_step{005000,010000,014000}_outs_n580_{tfidf,semantic}_use3_K4.json` | Exact-document retrieval top-1 among 580 candidates (`obs_doc_top1`), chance = 1/580 = 0.001724, with a permutation null re-drawn under 4 seeds (`per_seed`) and the worst case across them (`worst_case.max_p_value`, floor 1/3001 = 0.0003). `tfidf` scores the generated description against the documents with tf-idf cosine; `semantic` with MiniLM sentence embeddings. |
| `relabel_step014000_shuffle_outs_n580_*` | Control: the same checkpoint scored with the injected activations shuffled across documents. A verbalizer that reads its input collapses to chance here; one that ignores it does not move. |
| `relabel_step014000_noinj_outs_n24_*` | Control: no activation injected at all (24 documents). Expected outcome for a working verbalizer is a single degenerate output; chance for this file is 1/24. |
| `*_intradomain_diag.json` | Within-domain diagnostics (how often the top-1 hit lands in the right topic neighbourhood vs. the right document). The summary-level routing decomposition for this run has not yet been extracted; treat these as raw. |

The raw generated descriptions (one pickle per run, ~400 KB each) are not copied here;
they are in the source repository at
`experiments/v8_nla_local/results/nla_local/f193_relabel_eval_n580_20260829/`.

## Headline, in words

At step 14000 the relabeled model identifies the exact source document for 54.1% of
held-out activations by tf-idf and 47.1% by semantic similarity, against a chance rate
of 0.17%: roughly 314x and 273x chance, with the permutation p-value at its floor at
every step. Shuffling the activations sends both numbers to chance, and removing the
injection collapses the output to one template, so the signal is activation-specific.
For comparison, the paper's best GRPO-trained model reaches 7–9x chance on the same
set, and a matched ablation that retrains on the *original* corpus through the identical
pipeline (`../ablation_original_corpus/`) tops out at 7x in one seed and 2x in the other.

## Caveats that travel with these numbers

- One training seed. A second seed is queued.
- The relabeled corpus (1356 rows) is not the original corpus with new labels; it
  overlaps the original (1757 rows) on about 81% of documents. The ablation therefore
  isolates "which corpus", not "label style alone". A row-matched relabel is queued.
- Routing vs. content decomposition (the NLAttack v2 split used elsewhere in the
  paper) has not been extracted for these checkpoints yet.
- The checkpoint itself is not yet on Hugging Face; release under the `Solshine`
  namespace is pending.
