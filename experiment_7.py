"""
Experiment 7: Corruption augmentation.

Train the standard-selected one- and three-lead subsets with corruption 
augmentation. 

Outputs (per subset size 1...3)
  corruption_aug_p{prob}_predictions_size_{n}.csv
  corruption_aug_p{prob}_final_metrics_size_{n}.csv
"""
import pandas as pd
import pytorch_lightning as pl
import config, common
from ecg import SWAP_REVERSAL_CORRUPTIONS

def run(corruption_prob=0.25, max_size=3, max_epochs=None, seed=config.SEED):
    print("=== Experiment 7: corruption augmentation ===")
    pl.seed_everything(seed, workers=True)
    path = ['V5', 'V1', 'II']

    for n in range(1, max_size + 1):
        subset = path[:n]
        name = f"exp7_{n}"
        common.train_model(name, leads=subset, augment_corruptions=True, 
                           corruption_prob=corruption_prob, max_epochs=max_epochs, 
                           seed=seed)
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

        # Evaluate on all applicable corruptions
        evals = [("single_lead_polarity_inversion", l) for l in subset]
        evals += [(c, None) for c in SWAP_REVERSAL_CORRUPTIONS 
                  if common.is_applicable(c, subset)]
        for corruption, lead in evals:
            r = common.evaluate(name, leads=subset, split="test",
                                corruption=corruption, corruption_lead=lead)
            meta = dict(mode="corrupted", corruption=corruption, is_applicable=True, **base)
            preds.append(common.predictions_frame(r["sample_ids"], r["labels"],
                                                    r["probs"], cn, **meta))
            rows += common.build_metric_rows(r["labels"], r["probs"], cn, clean_probs=cp,
                                                test_loss=r["test_loss"], **meta)

        common.save(pd.concat(preds, ignore_index=True),
                    f"corruption_aug_p{corruption_prob}_predictions_size_{n}.csv")
        final_cols = ["subset_size", "selected_subset", "mode", "corruption",
                        "is_applicable", "class", "auroc", "auprc", "f1", "sensitivity",
                        "specificity", "fnr", "probability_shift", "prediction_flip_rate",
                        "high_conf_fpr", "high_conf_fnr"]
        common.save(pd.DataFrame(rows)[final_cols],
                    f"corruption_aug_p{corruption_prob}_final_metrics_size_{n}.csv")
        print(f"  size {n}: {subset}")


if __name__ == "__main__":
    run()
