# A working NLA for the GPU-poor

**The question this project asked. Can you train a generative Natural
Language Autoencoder on a 4 GB GTX 1650 Ti?**

**The answer is yes.** This repo holds the result data and figures behind
every claim. The full writeup is on Medium (link coming at publication) and
a methodology paper is in preparation.

![The trail and the field kit](figures/fig_nla_fieldguide_hero.png)

## What's here

An NLA is a pair of fine-tuned models. An Activation Verbalizer (AV) turns
a vector from inside a target model into a short description, and an
Activation Reconstructor maps the description back. This project trained
one for google/gemma-4-E2B on a used $700 laptop, evaluated it with
pre-committed gates and positive controls, rebuilt its benchmark when the
first eval turned out to be measuring the wrong thing, and then improved it
with GRPO for about $200 of decade-old GPUs.

Key numbers, each traceable to a JSON in `results/`.

- Round-trip faithfulness (v0, 42 held-out rows) mean cosine 0.438, worst
  row 0.313, every row above the pre-committed 0.30 gate.
- Exact-document retrieval among 580 held-out candidates went from 3x
  chance (base) to 9x chance after 500 GRPO steps (seed 0; the independent
  seed 1 reached 7x), and transferred to a lexical metric the reward never
  saw.
- A blind judge from a different model family preferred the trained model
  in 28 of 36 decided comparisons (p = 6e-4).
- On 295 never-seen documents across 4 unseen domains, a topic-level
  ranking signal survives (mean own-doc percentile 0.376 vs base 0.428 vs
  0.5 chance, paired dense-vs-base p = 0.008). This lead lives in the
  semantic-embedding metric; the lexical metric is flat out of distribution,
  and exact top-1 is at chance for both models. So topic-level ranking
  generalizes to unseen domains; word-for-word document identification does
  not.
- Causal patching places the improvement early, at the injection site,
  with late-layer processing becoming more distributed. Direction
  replicated across seeds.

## Weights

- Base AV/AR pair. [av-v0_0_1](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-v0_0_1) / [ar-v0_0_1](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1)
- GRPO-improved AV, seed 0. [av-grpo-dense500](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-grpo-dense500)
- GRPO-improved AV, seed 1. [av-grpo-dense500-s1](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-grpo-dense500-s1)
- More checkpoints under the [Solshine](https://huggingface.co/Solshine) namespace.

## Files

- `figures/` publication figures (hand-drawn infographics + data charts)
- `results/` evaluation JSONs. Naming. `step_005000_*` is the base model,
  `dense500*` the GRPO-improved model (s1 = second seed), `*_n580_*` the
  held-out set, `*_n295_*` the never-seen fresh set, `judge2afc_*` the
  blind-judge verdict, `t2_summary_*` the causal-patching contrasts.

## Verify the numbers yourself (no GPU)

Every headline number above traces to a JSON in `results/`. To confirm that,
with nothing but a Python install and no model download, run

```
python verify_claims.py
```

It re-derives all 21 headline claims from the committed data and prints a
pass/fail table. Regenerating the JSONs from scratch needs the HuggingFace
adapters plus the eval harness, which ships with the methodology paper; this
script checks that the published data is self-consistent with the published
claims.

Evaluation conventions worth copying. Every retrieval stat is reported as
the worst case across 4 negative-sampling seeds (max p-value, min effect).
Every harness carries a positive control. If your permutation null ever
comes back with zero variance, stop and check your metric.

Training code, the full eval harness, and the NLAttack v2 benchmark spec
release with the paper.

Questions or replications welcome. Find me on
[LinkedIn](https://www.linkedin.com/in/caleb-deleeuw-08509390/).
