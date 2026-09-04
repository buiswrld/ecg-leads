"""
Experiment 2: Full-lead corruption sensitivity.
Goal: measure how much each corruption hurts the 12-lead model; compare corruptions.

Corruption is applied to the FULL 12-lead signal (Methods 3.3).

Outputs
  Experiment_2_predictions.csv   per-sample, per-corruption predictions
  Experiment_2_avg_metrics.csv   averaged metrics per corruption
  Experiment_2_class_metrics.csv per-class metrics per corruption
"""
import pandas as pd
import config, common
from ecg import LEAD_ORDER, SWAP_REVERSAL_CORRUPTIONS


def run(max_epochs=None, seed=config.SEED):
    print("=== Experiment 2: 12-lead corruption sensitivity ===")
    common.train_model("exp2", leads=None, max_epochs=max_epochs, seed=seed)

    clean = common.evaluate("exp2", leads=None, split="test")
    cn, cp = clean["class_names"], clean["probs"]

    preds, class_rows = [], []
    preds.append(common.predictions_frame(clean["sample_ids"], clean["labels"], cp, cn,
                                          mode="clean", corruption=None,
                                          corruption_lead_idx=None,
                                          corruption_lead_name=None))
    class_rows += common.build_metric_rows(clean["labels"], cp, cn,
                                           test_loss=clean["test_loss"], mode="clean",
                                           corruption=None, corruption_lead_idx=None,
                                           corruption_lead_name=None)

    evals = [(c, None) for c in SWAP_REVERSAL_CORRUPTIONS] + \
            [("single_lead_polarity_inversion", l) for l in LEAD_ORDER]

    for corruption, lead in evals:
        r = common.evaluate("exp2", leads=None, split="test",
                            corruption=corruption, corruption_lead=lead)
        meta = dict(mode="corrupted", corruption=corruption,
                    corruption_lead_idx=common.lead_idx(lead),
                    corruption_lead_name=lead)
        preds.append(common.predictions_frame(r["sample_ids"], r["labels"],
                                              r["probs"], cn, **meta))
        class_rows += common.build_metric_rows(r["labels"], r["probs"], cn,
                                               clean_probs=cp,
                                               test_loss=r["test_loss"], **meta)
        allr = common.build_metric_rows(r["labels"], r["probs"], cn, clean_probs=cp)[-1]
        print(f"  {corruption:32s} {str(lead):5s} AUROC={allr['auroc']:.4f} F1={allr['f1']:.4f}")

    common.save(pd.concat(preds, ignore_index=True), "Experiment_2_predictions.csv")

    cdf = pd.DataFrame(class_rows)
    class_cols = ["mode", "corruption", "corruption_lead_idx", "corruption_lead_name",
                  "class", "test_loss", "auroc", "auprc", "f1", "sensitivity",
                  "specificity", "fnr", "probability_shift", "prediction_flip_rate",
                  "high_conf_fpr", "high_conf_fnr"]
    common.save(cdf[class_cols], "Experiment_2_class_metrics.csv")

    avg_cols = ["mode", "corruption", "corruption_lead_idx", "corruption_lead_name",
                "test_loss", "auroc", "auprc", "f1", "sensitivity", "fnr",
                "probability_shift", "prediction_flip_rate",
                "high_conf_fpr", "high_conf_fnr"]
    common.save(cdf[cdf["class"] == "ALL"][avg_cols], "Experiment_2_avg_metrics.csv")
    return cdf


if __name__ == "__main__":
    run()
