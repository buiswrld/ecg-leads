"""
Experiment 3: Clean greedy selection.
Goal: under perfect conditions, find the optimal lead subsets.

Selection scores candidates on the VALIDATION fold (9). Selected subsets are
then retrained at the full budget and reported on the TEST fold (10).

Outputs (per subset size 1..6)
  Experiment_3_predictions_size_N.csv
  Experiment_3_class_metrics_size_N.csv
  Experiment_3_selection_path.csv     the greedy path + candidate scores
"""
import json
import pandas as pd
import pytorch_lightning as pl
import config, common
from ecg import LEAD_ORDER


def greedy_clean(max_size=6, screening_epochs=None, seed=config.SEED):
    """Forward selection on clean validation macro-F1."""
    screening_epochs = screening_epochs or config.SCREENING_EPOCHS
    selected, candidate_log = [], []
    for step in range(1, max_size + 1):
        best = (-1.0, None)
        for cand in LEAD_ORDER:
            if cand in selected:
                continue
            trial = selected + [cand]
            name = f"exp3_s{step}_{cand}"
            common.train_model(name, leads=trial, max_epochs=screening_epochs,
                               seed=seed + step)
            f1 = common.macro_f1(common.evaluate(name, leads=trial, split="val"))
            candidate_log.append({"subset_size": step, "selected_before": str(selected),
                                  "candidate_added": cand, "candidate_subset": str(trial),
                                  "val_macro_f1": f1})
            if f1 > best[0]:
                best = (f1, cand)
            common.cleanup(name)
        selected.append(best[1])
        print(f"  size {step}: +{best[1]} -> {selected}  (val macro-F1={best[0]:.4f})")
    return selected, candidate_log


def run(max_size=6, max_epochs=None, screening_epochs=None, seed=config.SEED):
    print("=== Experiment 3: clean greedy selection (selects on fold 9) ===")
    pl.seed_everything(seed, workers=True)
    path, log = greedy_clean(max_size, screening_epochs, seed)
    common.save(pd.DataFrame(log), "Experiment_3_selection_path.csv")

    optimal_sets = {}
    for n in range(1, max_size + 1):
        subset = path[:n]
        optimal_sets[n] = subset
        name = f"exp3_final_{n}"
        common.train_model(name, leads=subset, max_epochs=max_epochs, seed=seed)
        r = common.evaluate(name, leads=subset, split="test")

        meta = dict(subset_size=n, selected_subset=str(subset))
        common.save(common.predictions_frame(r["sample_ids"], r["labels"], r["probs"],
                                             r["class_names"], **meta),
                    f"Experiment_3_predictions_size_{n}.csv")
        rows = common.build_metric_rows(r["labels"], r["probs"], r["class_names"],
                                        test_loss=r["test_loss"], **meta)
        cols = ["subset_size", "selected_subset", "class", "auroc", "auprc", "f1",
                "sensitivity", "specificity", "fnr"]
        common.save(pd.DataFrame(rows)[cols], f"Experiment_3_class_metrics_size_{n}.csv")
        allr = [x for x in rows if x["class"] == "ALL"][0]
        print(f"  size {n} {subset}: test F1={allr['f1']:.4f} AUROC={allr['auroc']:.4f}")

    with open(f"{config.RESULTS_DIR}/optimal_sets.json", "w") as f:
        json.dump({str(k): v for k, v in optimal_sets.items()}, f, indent=2)
    return optimal_sets


if __name__ == "__main__":
    run()
