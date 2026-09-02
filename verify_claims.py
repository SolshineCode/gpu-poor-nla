#!/usr/bin/env python3
"""verify_claims.py — reproduce every headline number in the README from the
committed result JSONs. No GPU, no model download, no dependencies beyond the
Python standard library. This is the "reproducible even by the GPU-poor" check:
it confirms that each claim in the README traces to a specific file in results/.

    python verify_claims.py

Exits 0 if every claimed number matches its artifact (within tolerance),
1 otherwise. Regenerating the JSONs themselves from the model weights needs the
HuggingFace adapters (Solshine namespace) plus the eval harness, which ships
with the methodology paper; this script verifies the published data is
self-consistent with the published claims.
"""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
TOL = 5e-4  # absolute tolerance for float comparisons


def load(name):
    return json.loads((R / name).read_text())


def approx(a, b, tol=TOL):
    return abs(a - b) <= tol


CHECKS = []


def check(desc, got, want, tol=TOL, source=""):
    ok = approx(got, want, tol) if isinstance(want, float) else (got == want)
    CHECKS.append((ok, desc, got, want, source))


def main():
    # --- Claim: retrieval 3x -> 9x chance (seed 0), 7x (seed 1); tfidf transfer ---
    chance_580 = 1 / 580
    d0 = load("dense500_semantic_use3_K4_n580_fixedinj.json")
    d1 = load("dense500s1_semantic_use3_K4_n580_fixedinj.json")
    base = load("step_005000_semantic_use3_K4_n580_fixedinj.json")

    def top1(doc):
        # doc-level retrieval top-1; the use3 JSONs store it under obs_doc_top1
        for key in ("obs_doc_top1", "doc_top1", "top1"):
            if key in doc:
                return float(doc[key])
        raise KeyError("top1 not found in " + json.dumps(list(doc.keys())))

    t0, t1, tb = top1(d0), top1(d1), top1(base)
    check("seed 0 semantic top-1 ~ 9x chance", round(t0 / chance_580, 1), 9.0, tol=0.6,
          source="dense500_semantic_use3_K4_n580_fixedinj.json")
    check("seed 1 semantic top-1 ~ 7x chance", round(t1 / chance_580, 1), 7.0, tol=0.6,
          source="dense500s1_semantic_use3_K4_n580_fixedinj.json")
    check("base semantic top-1 ~ 3x chance", round(tb / chance_580, 1), 3.0, tol=0.7,
          source="step_005000_semantic_use3_K4_n580_fixedinj.json")

    # --- Claim: blind judge 28/36 decided, p = 6e-4, 64 ties, no position bias ---
    j = load("judge2afc_base_vs_dense500_verdict.json")
    check("judge dense_wins", j["dense_wins"], 28, source="judge2afc_base_vs_dense500_verdict.json")
    check("judge base_wins", j["base_wins"], 8, source="judge2afc_base_vs_dense500_verdict.json")
    check("judge ties", j["ties"], 64, source="judge2afc_base_vs_dense500_verdict.json")
    check("judge decided = 36", j["dense_wins"] + j["base_wins"], 36,
          source="judge2afc_base_vs_dense500_verdict.json")
    check("judge one-sided binomial p = 6e-4", j["binomial_p_one_sided"], 0.0006, tol=1e-4,
          source="judge2afc_base_vs_dense500_verdict.json")
    check("judge model is a different family (qwen)", j["model"].startswith("qwen"), True,
          source="judge2afc_base_vs_dense500_verdict.json")
    # both display positions favor dense -> no position bias
    check("no position bias (both slots favor dense)",
          j["position_check"]["A"]["rate"] > 0.5 and j["position_check"]["B"]["rate"] > 0.5, True,
          source="judge2afc_base_vs_dense500_verdict.json")

    # --- Claim: OOD rank-level survives (0.376 vs base 0.428 vs 0.5), p=0.0003; top-1 at chance ---
    ood_d = load("dense500_outs_n295_fixedinj_ranksupp.json")["modes"]["semantic"]
    ood_b = load("step_005000_outs_n295_fixedinj_ranksupp.json")["modes"]["semantic"]
    check("OOD dense500 mean percentile 0.376", ood_d["obs"]["mean_percentile"], 0.376, tol=2e-3,
          source="dense500_outs_n295_fixedinj_ranksupp.json")
    check("OOD base mean percentile 0.428", ood_b["obs"]["mean_percentile"], 0.428, tol=3e-3,
          source="step_005000_outs_n295_fixedinj_ranksupp.json")
    check("OOD dense beats base (lower percentile)",
          ood_d["obs"]["mean_percentile"] < ood_b["obs"]["mean_percentile"], True,
          source="dense500/step_005000 n295 ranksupp")
    check("OOD dense worst-case perm p = 0.0003", ood_d["worst_case_max_p"]["p_mean_percentile"],
          0.0003, tol=1e-4, source="dense500_outs_n295_fixedinj_ranksupp.json")
    check("OOD top-1 near chance for both",
          ood_d["obs"]["top1"] < 0.01 and ood_b["obs"]["top1"] < 0.01, True,
          source="n295 ranksupp (both)")

    # --- Claim: causal patching, clean-corrupt gap 1.60 -> 2.44, positive control exact ---
    t2 = load("t2_summary_base5000_trace_n48_swap_vs_dense500_trace_n48_swap.json")
    check("causal base median gap 1.60", t2["median_gap8_a"], 1.60, tol=2e-2,
          source="t2_summary_base5000_..._vs_dense500_....json")
    check("causal dense median gap 2.44", t2["median_gap8_b"], 2.44, tol=2e-2,
          source="t2_summary_base5000_..._vs_dense500_....json")
    check("positive control E|marker recovery = 1.000",
          t2["cells_a"]["E|marker"]["mean_recovery"], 1.0, tol=1e-3,
          source="t2_summary (cells_a)")
    check("gap growth is significant (wilcoxon p < 0.05)",
          t2["gap_growth"]["wilcoxon_p"] < 0.05, True, source="t2_summary (gap_growth)")

    # --- Claim: seed-1 causal replication (direction), gap -> 2.29 ---
    t2s1 = load("t2_summary_base5000_trace_n48_swap_vs_dense500s1_trace_n48_swap.json")
    check("causal seed-1 dense median gap 2.29", t2s1["median_gap8_b"], 2.29, tol=3e-2,
          source="t2_summary_base5000_..._vs_dense500s1_....json")
    check("causal seed-1 gap grows over base",
          t2s1["median_gap8_b"] > t2s1["median_gap8_a"], True,
          source="t2_summary (s1)")

    # --- 2026-09-02 update: the relabeled-corpus run and its matched ablation ---
    # Same held-out set (580 documents), same K=4 permutation-null worst case.
    # Multiples of chance use the exact 1/580 = 0.001724 (the SUMMARY.json prose
    # in these folders rounds chance to 0.0017 and so prints slightly larger
    # multiples; the JSON fields are what is checked here).
    def mult(name):
        return round(top1(load(name)) / chance_580, 0)

    RL = "relabel_v1/relabel_step0%s_outs_n580_%s_use3_K4.json"
    for step, want_t, want_s in ((5000, 154.0, 141.0), (10000, 279.0, 245.0), (14000, 314.0, 273.0)):
        ss = f"{step:06d}"[1:]   # "05000", "10000", "14000" as in the filenames
        check(f"relabeled run step {step}: tf-idf top-1 ~{int(want_t)}x chance",
              mult(RL % (ss, "tfidf")), want_t, tol=1.0, source=RL % (ss, "tfidf"))
        check(f"relabeled run step {step}: semantic top-1 ~{int(want_s)}x chance",
              mult(RL % (ss, "semantic")), want_s, tol=1.0, source=RL % (ss, "semantic"))
        for mode in ("tfidf", "semantic"):
            w = load(RL % (ss, mode))["worst_case"]["max_p_value"]
            check(f"relabeled run step {step}: {mode} worst-case p at permutation floor (0.0003)",
                  w, 0.0003, tol=1e-4, source=RL % (ss, mode))
    sh_t = load("relabel_v1/relabel_step014000_shuffle_outs_n580_tfidf_use3_K4.json")
    sh_s = load("relabel_v1/relabel_step014000_shuffle_outs_n580_semantic_use3_K4.json")
    check("relabeled run, shuffled activations: tf-idf top-1 = 0 (chance)", top1(sh_t), 0.0, tol=1e-6,
          source="relabel_v1/relabel_step014000_shuffle_outs_n580_tfidf_use3_K4.json")
    check("relabeled run, shuffled activations: semantic top-1 = 1x chance", round(top1(sh_s) / chance_580, 0), 1.0,
          tol=0.5, source="relabel_v1/relabel_step014000_shuffle_outs_n580_semantic_use3_K4.json")
    check("relabeled run, shuffled activations: semantic p not significant",
          sh_s["worst_case"]["max_p_value"] > 0.05, True,
          source="relabel_v1/relabel_step014000_shuffle_outs_n580_semantic_use3_K4.json")
    noinj = load("relabel_v1/relabel_step014000_noinj_outs_n24_semantic_use3_K4.json")
    check("relabeled run, no injection: output collapses (top-1 = its own 1/24 chance)",
          round(top1(noinj) * 24, 2), 1.0, tol=0.05,
          source="relabel_v1/relabel_step014000_noinj_outs_n24_semantic_use3_K4.json")
    summ = load("relabel_v1/SUMMARY.json")["runs"]
    check("relabeled run step 14000: 497 of 580 outputs unique", summ["relabel_step014000"]["n_unique_outputs"], 497,
          source="relabel_v1/SUMMARY.json")
    check("relabeled run, no injection: 1 unique output of 24", summ["relabel_step014000_noinj"]["n_unique_outputs"], 1,
          source="relabel_v1/SUMMARY.json")

    AB = "ablation_original_corpus/seed%d_step0%s_outs_n580_%s_use3_K4.json"
    check("ablation (original corpus) seed 1 step 10000: semantic ~7x chance",
          mult(AB % (1, "10000", "semantic")), 7.0, tol=0.6, source=AB % (1, "10000", "semantic"))
    check("ablation seed 1 step 14000: semantic ~7x chance",
          mult(AB % (1, "14000", "semantic")), 7.0, tol=0.6, source=AB % (1, "14000", "semantic"))
    check("ablation seed 1 step 14000: semantic worst-case p = 0.0010",
          load(AB % (1, "14000", "semantic"))["worst_case"]["max_p_value"], 0.0010, tol=2e-4,
          source=AB % (1, "14000", "semantic"))
    check("ablation seed 0 step 14000: semantic ~2x chance (worst case across seeds)",
          mult(AB % (0, "14000", "semantic")), 2.0, tol=0.6, source=AB % (0, "14000", "semantic"))
    check("ablation seed 0 step 14000: semantic not significant (p = 0.27)",
          load(AB % (0, "14000", "semantic"))["worst_case"]["max_p_value"], 0.266, tol=2e-2,
          source=AB % (0, "14000", "semantic"))
    for seed in (0, 1):
        for step in ("05000", "10000", "14000"):
            check(f"ablation seed {seed} step {int(step)}: tf-idf null (<=2x chance, p > 0.26)",
                  mult(AB % (seed, step, "tfidf")) <= 2.0
                  and load(AB % (seed, step, "tfidf"))["worst_case"]["max_p_value"] > 0.26, True,
                  source=AB % (seed, step, "tfidf"))
    base_s = load("ablation_original_corpus/baseline_n580_semantic.json")
    check("baseline (step 5000) semantic ~3x chance, p = 0.08 (same anchor as the round-1 check)",
          round(top1(base_s) / chance_580, 0) == 3.0 and abs(base_s["worst_case"]["max_p_value"] - 0.0796) < 2e-3,
          True, source="ablation_original_corpus/baseline_n580_semantic.json")
    check("relabeled run beats the ablation's best case by >= 20x at every step",
          mult(RL % ("05000", "semantic")) / max(mult(AB % (s, st, m))
                                                for s in (0, 1) for st in ("05000", "10000", "14000")
                                                for m in ("tfidf", "semantic")) >= 20.0, True,
          source="relabel_v1/ vs ablation_original_corpus/")

    # --- files the README cites that are NOT yet in results/ (see results/PENDING_FROM_BACKUP.md) ---
    # These were moved from the source repository to a private provenance backup on
    # 2026-08-24 and have not been copied here yet. Each is a FAIL, not a skip, so that
    # this script cannot report a clean bill of health while a cited artifact is absent.
    pending_path = R / "PENDING_FROM_BACKUP.md"
    if pending_path.exists():
        for line in pending_path.read_text().splitlines():
            if line.startswith("| `") and "|" in line[3:]:
                name = line.split("`")[1]
                if not (R / name).exists():
                    check(f"cited artifact present: {name}", "MISSING (not yet copied from the private provenance backup)",
                          "present", source="results/PENDING_FROM_BACKUP.md")

    # --- report ---
    width = max(len(d) for _, d, _, _, _ in CHECKS)
    npass = sum(1 for ok, *_ in CHECKS if ok)
    print(f"\nVerifying {len(CHECKS)} README claims against results/*.json\n")
    for ok, desc, got, want, source in CHECKS:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {desc:<{width}}  got={got}  want={want}")
        if not ok:
            print(f"         source: {source}")
    print(f"\n{npass}/{len(CHECKS)} claims verified.")
    if npass != len(CHECKS):
        print("Some claims did not match their artifacts. See FAIL lines above.")
        return 1
    print("Every headline number in the README traces to a committed result file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
