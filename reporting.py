import os
import pandas as pd
import torch

from metrics import ECGMetrics
CLASS_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"]
LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

def report_predictions(
    outputs,
    save_dir,
    leads=None,
    mode="clean",
    corruption=None,
):
    """
    Generate:
        CSV: per-sample predictions

    outputs:
        task.test_epoch_outputs
    """

    os.makedirs(save_dir, exist_ok=True)

    # Combine batches
    sample_ids = []
    labels = []
    probs = []
    preds = []

    for batch in outputs:
        sample_ids.extend(batch["sample_id"])

        labels.append(batch["labels"])
        probs.append(batch["probs"])
        preds.append(batch["preds"])

    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)
    preds = torch.cat(preds, dim=0)

    if leads is None:
        leads = LEAD_NAMES

    prediction_rows = []

    for i, sample_id in enumerate(sample_ids):

        row = {
            "sample_id": sample_id,
            "subset_size": len(leads),
            "selected_subset": '-'.join(leads),
        }

        for j, cls in enumerate(CLASS_NAMES):
            row[f"true_{cls}"] = int(labels[i, j])

        for j, cls in enumerate(CLASS_NAMES):
            row[f"prob_{cls}"] = float(probs[i, j])

        for j, cls in enumerate(CLASS_NAMES):
            row[f"pred_{cls}"] = int(preds[i, j])

        prediction_rows.append(row)

    pred_path = os.path.join(
        save_dir,
        f"Experiment_5_{corruption}_{mode}_inference.csv"
    )

    pd.DataFrame(prediction_rows).to_csv(
        pred_path,
        mode="a",
        header=not os.path.exists(pred_path),
        index=False,
    )


def report_per_class_metrics(
    scores,
    save_dir,
    leads=None,
    mode="clean",
    corruption=None,
    instability=None,
):
    """
    Generate:
        CSV: per-class metrics

    scores:
        task.test_epoch_results
    """

    os.makedirs(save_dir, exist_ok=True)

    if leads is None:
        leads = LEAD_NAMES

    per_class_rows = []

    for i, class_result in enumerate(scores["per_class"]):
        row = {
            "subset_size": len(leads),
            "selected_subset": '-'.join(leads),

            "class": class_result["class"],

            "auroc": class_result["auroc"],
            "auprc": class_result["auprc"],
            "f1": class_result["f1"],
            "sensitivity": class_result["sensitivity"],
            "specificity": class_result["specificity"],
            "fnr": class_result["fnr"],
        }

        if instability is not None:
            instability_result = instability[i]

            row.update({
                "probability_shift": instability_result["probability_shift"],
                "prediction_flip_rate": instability_result["prediction_flip_rate"],
                "high_conf_fpr": instability_result["high_conf_fpr"],
                "high_conf_fnr": instability_result["high_conf_fnr"]
            })

        per_class_rows.append(row)

    # Add ALL row
    all_scores = scores["overall"]

    all_row = {
        "subset_size": len(leads),
        "selected_subset": '-'.join(leads),

        "class": "ALL",

        "auroc": all_scores["auroc"],
        "auprc": all_scores["auprc"],
        "f1": all_scores["f1"],
        "sensitivity": all_scores["sensitivity"],
        "specificity": all_scores["specificity"],
        "fnr": all_scores["fnr"],
    }

    if instability is not None:
        instability_result = instability[5]

        all_row.update({
            "probability_shift": instability_result["probability_shift"],
            "prediction_flip_rate": instability_result["prediction_flip_rate"],
            "high_conf_fpr": instability_result["high_conf_fpr"],
            "high_conf_fnr": instability_result["high_conf_fnr"]
        })

    per_class_rows.append(all_row)

    per_class_path = os.path.join(
        save_dir,
        f"Experiment_5_{corruption}_{mode}_metrics.csv"
    )

    pd.DataFrame(per_class_rows).to_csv(
        per_class_path,
        mode="a",
        header=not os.path.exists(per_class_path),
        index=False,
    )