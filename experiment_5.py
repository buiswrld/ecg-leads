"""
Experiment 5: Corruption-specific greedy selection.

For each swap/reversal corruption, greedily select the strongest 
corruption-affected leads based on clean validation performance. The 
resulting subsets are retrained at the full budget and evaluated on the 
TEST fold under both clean and corrupted conditions.

Selection scores candidates on the VALIDATION fold (9). Selected subsets 
are then retrained at the full budget and reported on the TEST fold (10).

Outputs (per corruption)
  Experiment_5_candidates_C.csv
  Experiment_5_predictions_C.csv
  Experiment_5_class_metrics_C.csv
"""
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import config, common
from ecg import SWAP_REVERSAL_CORRUPTIONS, CORRUPTION_AFFECTED_LEADS

def greedy_per_corruption(max_size=5, screening_epochs=None, seed=config.SEED, 
                          corruption=None):
    screening_epochs = screening_epochs or config.SCREENING_EPOCHS
    selected, candidate_log = [], []
    
    for step in range(1, max_size + 1):
        best = (-1.0, None)
        for cand in CORRUPTION_AFFECTED_LEADS[corruption]:
            if cand in selected:
                continue
            trial = selected + [cand]
            name = f"exp5_{'_'.join(trial)}"
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

    return selected, candidate_log

def run(max_epochs=None, screening_epochs=None, seed=config.SEED):
    print("=== Experiment 5: corruption-specific clean greedy selection ===")
    pl.seed_everything(seed, workers=True)

    for corruption in SWAP_REVERSAL_CORRUPTIONS:
        max_size = len(CORRUPTION_AFFECTED_LEADS[corruption])
        path, log = greedy_per_corruption(max_size, screening_epochs, seed, corruption)
        common.save(pd.DataFrame(log), f"Experiment_5_candidates_{corruption}.csv")

        preds, rows = [], []

        for n in range(1, max_size + 1):
            subset = path[:n]
            name = f"exp5_final_{'_'.join(subset)}"
            common.train_model(name, leads=subset, max_epochs=max_epochs, seed=seed)

            base = dict(subset_size=n, selected_subset=str(subset))

            clean = common.evaluate(name, leads=subset, split="test")
            cn, cp = clean["class_names"], clean["probs"]
            preds.append(common.predictions_frame(clean["sample_ids"], clean["labels"], cp, 
                                                  cn, mode="clean", corruption=None, **base))
            rows += common.build_metric_rows(clean["labels"], cp, cn, 
                                             test_loss=clean["test_loss"], mode="clean", 
                                             corruption=None, **base)

            corr = common.evaluate(name, leads=subset, split="test", corruption=corruption)
            meta = dict(mode="corrupted", corruption=corruption, is_applicable=True, **base)
            preds.append(common.predictions_frame(corr["sample_ids"], corr["labels"], 
                                                  corr["probs"], cn, **meta))
            rows += common.build_metric_rows(corr["labels"], corr["probs"], cn, clean_probs=cp, 
                                             test_loss=corr["test_loss"], **meta)

        common.save(pd.concat(preds, ignore_index=True), 
                    f"Experiment_5_predictions_{corruption}.csv")
        cols = ["subset_size", "selected_subset", "mode", "corruption", "class", "auroc", 
                "auprc", "f1", "sensitivity", "specificity", "fnr", "probability_shift", 
                "prediction_flip_rate", "high_conf_fpr", "high_conf_fnr"]
        common.save(pd.DataFrame(rows)[cols], 
                    f"Experiment_5_class_metrics_{corruption}.csv")

if __name__ == "__main__":
    run()