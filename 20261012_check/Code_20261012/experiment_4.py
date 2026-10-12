"""
Experiment 4: Clean-greedy subsets vs single-lead inversion.
Goal: of the strongest leads, which are most impacted by inversion?

For each optimal subset from Experiment 3, invert each retained lead in turn.

Outputs (per subset size)
  Experiment_4_predictions_size_N.csv
  Experiment_4_class_metrics_size_N.csv
"""
import json
import pandas as pd
import config, common


def load_optimal_sets():
    with open(f"{config.RESULTS_DIR}/optimal_sets.json") as f:
        return {int(k): v for k, v in json.load(f).items()}


def run(optimal_sets=None, max_epochs=None, seed=config.SEED):
    print("=== Experiment 4: subsets vs single-lead inversion ===")
    optimal_sets = optimal_sets or load_optimal_sets()

    for n, subset in sorted(optimal_sets.items()):
        name = f"exp4_{n}"
        common.train_model(name, leads=subset, max_epochs=max_epochs, seed=seed)
        clean = common.evaluate(name, leads=subset, split="test")
        cn, cp = clean["class_names"], clean["probs"]

        preds, rows = [], []
        for lead in subset:
            r = common.evaluate(name, leads=subset, split="test",
                                corruption="single_lead_polarity_inversion",
                                corruption_lead=lead)
            meta = dict(subset_size=n, selected_subset=str(subset),
                        corruption="single_lead_polarity_inversion",
                        inverted_lead_idx=common.lead_idx(lead),
                        inverted_lead_name=lead)
            preds.append(common.predictions_frame(r["sample_ids"], r["labels"],
                                                  r["probs"], cn, **meta))
            rows += common.build_metric_rows(r["labels"], r["probs"], cn,
                                             clean_probs=cp,
                                             test_loss=r["test_loss"], **meta)
            allr = common.build_metric_rows(r["labels"], r["probs"], cn, clean_probs=cp)[-1]
            print(f"  size {n} {subset} invert {lead}: F1={allr['f1']:.4f}")

        common.save(pd.concat(preds, ignore_index=True),
                    f"Experiment_4_predictions_size_{n}.csv")
        cols = ["subset_size", "selected_subset", "corruption", "inverted_lead_idx",
                "inverted_lead_name", "class", "auroc", "auprc", "f1", "sensitivity",
                "specificity", "fnr", "probability_shift", "prediction_flip_rate",
                "high_conf_fpr", "high_conf_fnr"]
        common.save(pd.DataFrame(rows)[cols],
                    f"Experiment_4_class_metrics_size_{n}.csv")


if __name__ == "__main__":
    run()
