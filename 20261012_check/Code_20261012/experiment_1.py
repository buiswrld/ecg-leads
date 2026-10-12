"""
Experiment 1: Full clean 12-lead baseline.
Goal: establish normal performance under perfect conditions.

Outputs
  Experiment_1_predictions.csv   sample_id | NORM..HYP | prob_* | pred_*
  Experiment_1_metrics.csv       class | auroc | auprc | f1 | sensitivity | specificity | fnr
"""
import pandas as pd
import config, common


def run(max_epochs=None, seed=config.SEED):
    print("=== Experiment 1: full clean 12-lead baseline ===")
    common.train_model("exp1", leads=None, max_epochs=max_epochs, seed=seed)
    res = common.evaluate("exp1", leads=None, split="test")

    common.save(common.predictions_frame(res["sample_ids"], res["labels"],
                                         res["probs"], res["class_names"]),
                "Experiment_1_predictions.csv")

    rows = common.build_metric_rows(res["labels"], res["probs"], res["class_names"],
                                    test_loss=res["test_loss"])
    cols = ["class", "auroc", "auprc", "f1", "sensitivity", "specificity", "fnr"]
    common.save(pd.DataFrame(rows)[cols], "Experiment_1_metrics.csv")

    allrow = [r for r in rows if r["class"] == "ALL"][0]
    print(f"  ALL: AUROC={allrow['auroc']:.4f} AUPRC={allrow['auprc']:.4f} "
          f"F1={allrow['f1']:.4f}")
    return res


if __name__ == "__main__":
    run()
