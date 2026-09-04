"""
Experiment 6: Robust greedy selection.

  6a  inversion-robust     -- robust score uses single-lead inversions only
  6b  all-corruption robust -- inversions + limb reversals + chest swaps

Selection uses the VALIDATION fold (9) for both clean and corrupted evaluation,
per the guideline. The final selected subsets are retrained and reported on the
TEST fold (10).

    D(S)       = max over applicable corruptions of [F0(S) - Fc(S)]+
    robust(S)  = F0(S) - lambda * D(S)

lambda defaults to 1.0. Pass --lam to sweep it.

Outputs (variant in {6a, 6b}, per subset size)
  Experiment_{v}_candidates_size_N.csv   every candidate + corruption + class
  Experiment_{v}_predictions_size_N.csv
  Experiment_{v}_final_metrics_size_N.csv
"""
import json
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import config, common
from ecg import LEAD_ORDER, SWAP_REVERSAL_CORRUPTIONS


def corruption_list(subset, variant):
    """Applicable corruptions for a subset. 6a: inversions only. 6b: all."""
    evals = [("single_lead_polarity_inversion", l) for l in subset]
    if variant == "6b":
        evals += [(c, None) for c in SWAP_REVERSAL_CORRUPTIONS
                  if common.is_applicable(c, subset)]
    return evals


def score_candidate(name, subset, variant, lam, clean_res):
    """Clean F0, worst-case degradation D, robust score, and per-corruption rows."""
    f0 = common.macro_f1(clean_res)
    cn, cp = clean_res["class_names"], clean_res["probs"]
    drops, rows = [], []

    for corruption, lead in corruption_list(subset, variant):
        r = common.evaluate(name, leads=subset, split="val",
                            corruption=corruption, corruption_lead=lead)
        fc = common.macro_f1(r)
        drops.append(max(f0 - fc, 0.0))
        rows += common.build_metric_rows(
            r["labels"], r["probs"], cn, clean_probs=cp, test_loss=r["test_loss"],
            corruption=corruption, is_applicable=True,
            corrupted_lead_idx=common.lead_idx(lead), corrupted_lead_name=lead)

    D = max(drops) if drops else 0.0
    return f0, D, f0 - lam * D, rows


def greedy_robust(variant, lam=1.0, max_size=6, screening_epochs=None,
                  seed=config.SEED):
    screening_epochs = screening_epochs or config.SCREENING_EPOCHS
    selected, per_size_rows = [], {}

    for step in range(1, max_size + 1):
        best, rows_this_step = (-np.inf, None), []
        for cand in LEAD_ORDER:
            if cand in selected:
                continue
            trial = selected + [cand]
            name = f"exp{variant}_s{step}_{cand}"
            common.train_model(name, leads=trial, max_epochs=screening_epochs,
                               seed=seed + step)
            clean_res = common.evaluate(name, leads=trial, split="val")
            f0, D, score, crows = score_candidate(name, trial, variant, lam, clean_res)

            base = dict(subset_size=step, selected_before=str(selected),
                        candidate_added=cand, candidate_subset=str(trial),
                        clean_macro_f1=f0, degradation=D, robust_score=score,
                        lam=lam)
            # clean reference row
            rows_this_step += common.build_metric_rows(
                clean_res["labels"], clean_res["probs"], clean_res["class_names"],
                test_loss=clean_res["test_loss"], corruption=None,
                is_applicable=False, corrupted_lead_idx=None,
                corrupted_lead_name=None, **base)
            for r in crows:
                r.update(base)
            rows_this_step += crows

            if score > best[0]:
                best = (score, cand)
            common.cleanup(name)

        selected.append(best[1])
        for r in rows_this_step:
            r["selected_this_round"] = (r["candidate_added"] == best[1])
        per_size_rows[step] = rows_this_step
        print(f"  [{variant}] size {step}: +{best[1]} -> {selected} "
              f"(robust={best[0]:.4f})")

    return selected, per_size_rows


def run(variant="6b", lam=1.0, max_size=6, max_epochs=None,
        screening_epochs=None, seed=config.SEED):
    print(f"=== Experiment {variant}: robust greedy selection (lambda={lam}) ===")
    pl.seed_everything(seed, workers=True)
    path, per_size = greedy_robust(variant, lam, max_size, screening_epochs, seed)

    cand_cols = ["subset_size", "selected_before", "candidate_added", "candidate_subset",
                 "corruption", "is_applicable", "corrupted_lead_idx",
                 "corrupted_lead_name", "class", "auroc", "auprc", "f1", "sensitivity",
                 "specificity", "fnr", "probability_shift", "prediction_flip_rate",
                 "high_conf_fpr", "high_conf_fnr", "robust_score", "selected_this_round"]
    for n, rows in per_size.items():
        common.save(pd.DataFrame(rows)[cand_cols],
                    f"Experiment_{variant}_candidates_size_{n}.csv")

    for n in range(1, max_size + 1):
        subset = path[:n]
        name = f"exp{variant}_final_{n}"
        common.train_model(name, leads=subset, max_epochs=max_epochs, seed=seed)
        clean = common.evaluate(name, leads=subset, split="test")
        cn, cp = clean["class_names"], clean["probs"]

        preds, rows = [], []
        base = dict(subset_size=n, selected_subset=str(subset))
        preds.append(common.predictions_frame(clean["sample_ids"], clean["labels"], cp, cn,
                                              mode="clean", corruption=None,
                                              is_applicable=False, **base))
        rows += common.build_metric_rows(clean["labels"], cp, cn,
                                         test_loss=clean["test_loss"], mode="clean",
                                         corruption=None, is_applicable=False, **base)
        for corruption, lead in corruption_list(subset, variant):
            r = common.evaluate(name, leads=subset, split="test",
                                corruption=corruption, corruption_lead=lead)
            meta = dict(mode="corrupted", corruption=corruption, is_applicable=True, **base)
            preds.append(common.predictions_frame(r["sample_ids"], r["labels"],
                                                  r["probs"], cn, **meta))
            rows += common.build_metric_rows(r["labels"], r["probs"], cn, clean_probs=cp,
                                             test_loss=r["test_loss"], **meta)

        common.save(pd.concat(preds, ignore_index=True),
                    f"Experiment_{variant}_predictions_size_{n}.csv")
        final_cols = ["subset_size", "selected_subset", "mode", "corruption",
                      "is_applicable", "class", "auroc", "auprc", "f1", "sensitivity",
                      "specificity", "fnr", "probability_shift", "prediction_flip_rate",
                      "high_conf_fpr", "high_conf_fnr"]
        common.save(pd.DataFrame(rows)[final_cols],
                    f"Experiment_{variant}_final_metrics_size_{n}.csv")
        print(f"  size {n}: {subset}")

    with open(f"{config.RESULTS_DIR}/robust_sets_{variant}.json", "w") as f:
        json.dump({str(k): path[:k] for k in range(1, max_size + 1)}, f, indent=2)
    return path


def run_6a(**kw): return run(variant="6a", **kw)
def run_6b(**kw): return run(variant="6b", **kw)


if __name__ == "__main__":
    import fire
    fire.Fire({"6a": run_6a, "6b": run_6b, "run": run})
