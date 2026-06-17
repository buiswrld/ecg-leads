import csv
import json
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from classifier import ECGClassificationTask
from dataset import PTBXLDataset
from metrics import compute_classification_metrics


ALL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
BASE_CORRUPTIONS = [
    ("ra_la_swap", None),
    ("ra_ll_swap", None),
    ("la_ll_swap", None),
    ("v1_v2_swap", None),
    ("v2_v3_swap", None),
]
POLARITY_CORRUPTION = "single_lead_polarity_inversion"
CHECKPOINT_MONITOR = "val_loss"
CHECKPOINT_MODE = "min"

TRIAL_FIELDS = [
    "mode",
    "step",
    "candidate",
    "selected_before",
    "trial_leads",
    "clean_f1",
    "mean_f1_drop",
    "score",
    "lam",
    "selected",
    "best_checkpoint",
]
CORRUPTION_FIELDS = [
    "mode",
    "step",
    "candidate",
    "trial_leads",
    "corruption",
    "corruption_lead_idx",
    "clean_f1",
    "corrupt_f1",
    "f1_drop",
]


def _lead_text(leads):
    return ",".join(leads)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(_json_safe(payload), handle, indent=2)


def _refresh_outputs(results_dir, trial_rows, corruption_rows, history, config):
    '''Persist every update so interrupted Colab runs still leave evidence.'''
    _write_csv(os.path.join(results_dir, "trials.csv"), TRIAL_FIELDS, trial_rows)
    _write_csv(os.path.join(results_dir, "corruptions.csv"), CORRUPTION_FIELDS, corruption_rows)
    _write_json(
        os.path.join(results_dir, "history.json"),
        {
            "config": config,
            "selected": history[-1]["leads_so_far"] if history else [],
            "history": history,
        },
    )


def _expand_corruptions(leads, corruption_limit=None):
    corruptions = list(BASE_CORRUPTIONS)
    for lead in leads:
        corruptions.append((POLARITY_CORRUPTION, ALL_LEADS.index(lead)))
    if corruption_limit is not None:
        return corruptions[: int(corruption_limit)]
    return corruptions


def _corruption_policy_snapshot(corruption_limit):
    corruptions = [
        {"corruption": name, "corruption_lead_idx": lead_idx}
        for name, lead_idx in BASE_CORRUPTIONS
    ]
    corruptions.append(
        {
            "corruption": POLARITY_CORRUPTION,
            "corruption_lead_idx": "each_trial_lead",
        }
    )
    return {
        "policy": "base_corruptions_plus_polarity_inversion_for_each_trial_lead",
        "corruption_limit": corruption_limit,
        "expanded_order": corruptions,
    }


def _test_loader(
    leads,
    mlb,
    corruption,
    corruption_lead_idx,
    csv_path,
    data_root,
    batch_size,
    num_workers,
):
    ds = PTBXLDataset(
        csv_path=csv_path,
        data_root=data_root,
        split="test",
        leads=leads,
        mlb=mlb,
        corruption=corruption,
        corruption_lead_idx=0 if corruption_lead_idx is None else corruption_lead_idx,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def _eval_f1(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["ecg"].to(device))
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"].cpu())

    if not all_logits:
        raise ValueError("Cannot score an empty dataloader.")

    metrics = compute_classification_metrics(torch.cat(all_logits), torch.cat(all_labels))
    val = metrics["f1"].item()
    return 0.0 if val != val else val


def train_model(
    leads,
    csv_path="processed_ptbxl_metadata.csv",
    data_root=".",
    batch_size=32,
    num_workers=0,
    learning_rate=1e-3,
    max_epochs=10,
    patience=3,
    accelerator="auto",
    ckpt_dir="greedy_ckpts",
    limit_train_batches=1.0,
    limit_val_batches=1.0,
    progress_bar=False,
):
    task = ECGClassificationTask(
        num_leads=len(leads),
        num_classes=5,
        learning_rate=learning_rate,
        csv_path=csv_path,
        data_root=data_root,
        leads=leads,
        batch_size=batch_size,
        num_workers=num_workers,
        corruption=None,
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor=CHECKPOINT_MONITOR,
        mode=CHECKPOINT_MODE,
        save_top_k=1,
        filename="best",
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        gradient_clip_val=0.5,
        callbacks=[
            EarlyStopping(monitor=CHECKPOINT_MONITOR, patience=patience, mode=CHECKPOINT_MODE),
            ckpt_callback,
        ],
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=progress_bar,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
    )
    trainer.fit(task)
    if not ckpt_callback.best_model_path:
        raise ValueError(f"No best checkpoint found for leads {leads}.")

    '''Evaluate the checkpoint selected by validation loss, not final epoch drift.'''
    best_task = ECGClassificationTask.load_from_checkpoint(
        ckpt_callback.best_model_path,
        num_leads=len(leads),
        num_classes=5,
        learning_rate=learning_rate,
        csv_path=csv_path,
        data_root=data_root,
        leads=leads,
        batch_size=batch_size,
        num_workers=num_workers,
        corruption=None,
    )
    return best_task, task._train_ds.mlb, ckpt_callback.best_model_path


def compute_score(
    model,
    mlb,
    leads,
    mode,
    lam,
    csv_path,
    data_root,
    batch_size,
    num_workers,
    device,
    corruption_limit=None,
):
    clean_loader = _test_loader(leads, mlb, None, None, csv_path, data_root, batch_size, num_workers)
    clean_f1 = _eval_f1(model, clean_loader, device)

    if mode == "clean":
        return {
            "clean_f1": clean_f1,
            "mean_f1_drop": 0.0,
            "score": clean_f1,
            "corruptions": [],
        }

    corruption_rows = []
    drops = []
    for corruption_name, lead_idx in _expand_corruptions(leads, corruption_limit):
        loader = _test_loader(
            leads,
            mlb,
            corruption_name,
            lead_idx,
            csv_path,
            data_root,
            batch_size,
            num_workers,
        )
        corrupt_f1 = _eval_f1(model, loader, device)
        drop = max(0.0, clean_f1 - corrupt_f1)
        drops.append(drop)
        corruption_rows.append(
            {
                "corruption": corruption_name,
                "corruption_lead_idx": "" if lead_idx is None else lead_idx,
                "clean_f1": clean_f1,
                "corrupt_f1": corrupt_f1,
                "f1_drop": drop,
            }
        )

    mean_drop = sum(drops) / len(drops) if drops else 0.0
    '''Score is clean performance minus the robustness penalty.'''
    return {
        "clean_f1": clean_f1,
        "mean_f1_drop": mean_drop,
        "score": clean_f1 - lam * mean_drop,
        "corruptions": corruption_rows,
    }


def greedy_select(
    mode="robust",
    max_leads=6,
    lam=0.5,
    csv_path="processed_ptbxl_metadata.csv",
    data_root=".",
    batch_size=32,
    num_workers=0,
    learning_rate=1e-3,
    max_epochs=10,
    patience=3,
    accelerator="auto",
    results_dir="greedy_results",
    candidate_limit=None,
    corruption_limit=None,
    limit_train_batches=1.0,
    limit_val_batches=1.0,
    progress_bar=False,
    seed=42,
):
    if mode not in {"clean", "robust"}:
        raise ValueError("mode must be 'clean' or 'robust'.")

    os.makedirs(results_dir, exist_ok=True)
    if seed is not None:
        pl.seed_everything(seed, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    remaining = list(ALL_LEADS)
    selected = []
    history = []
    trial_rows = []
    corruption_rows = []
    config = {
        "mode": mode,
        "target_k": max_leads,
        "max_leads": max_leads,
        "lambda": lam,
        "lam": lam,
        "seed": seed,
        "csv_path": csv_path,
        "data_root": data_root,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "patience": patience,
        "accelerator": accelerator,
        "candidate_limit": candidate_limit,
        "corruption_limit": corruption_limit,
        "limit_train_batches": limit_train_batches,
        "limit_val_batches": limit_val_batches,
        "checkpoint_monitor": CHECKPOINT_MONITOR,
        "checkpoint_mode": CHECKPOINT_MODE,
        "selection_metric": "f1",
        "score_formula": "clean_f1 - lambda * mean_f1_drop",
        "corruptions_used_for_selection": _corruption_policy_snapshot(corruption_limit),
        "device": str(device),
    }
    _refresh_outputs(results_dir, trial_rows, corruption_rows, history, config)

    for step in range(min(max_leads, len(ALL_LEADS))):
        print(f"\nStep {step + 1} | selected so far: {selected}")
        candidates = remaining[: int(candidate_limit)] if candidate_limit is not None else list(remaining)
        best_row = None
        step_candidates = []

        for candidate in candidates:
            trial_leads = selected + [candidate]
            ckpt_dir = os.path.join(results_dir, f"step{step + 1}_{candidate}_ckpts")
            model, mlb, best_checkpoint = train_model(
                leads=trial_leads,
                csv_path=csv_path,
                data_root=data_root,
                batch_size=batch_size,
                num_workers=num_workers,
                learning_rate=learning_rate,
                max_epochs=max_epochs,
                patience=patience,
                accelerator=accelerator,
                ckpt_dir=ckpt_dir,
                limit_train_batches=limit_train_batches,
                limit_val_batches=limit_val_batches,
                progress_bar=progress_bar,
            )
            model = model.to(device)
            result = compute_score(
                model,
                mlb,
                trial_leads,
                mode,
                lam,
                csv_path,
                data_root,
                batch_size,
                num_workers,
                device,
                corruption_limit=corruption_limit,
            )

            trial_row = {
                "mode": mode,
                "step": step + 1,
                "candidate": candidate,
                "selected_before": _lead_text(selected),
                "trial_leads": _lead_text(trial_leads),
                "clean_f1": result["clean_f1"],
                "mean_f1_drop": result["mean_f1_drop"],
                "score": result["score"],
                "lam": lam,
                "selected": False,
                "best_checkpoint": best_checkpoint,
            }
            trial_rows.append(trial_row)

            for corruption_result in result["corruptions"]:
                corruption_rows.append(
                    {
                        "mode": mode,
                        "step": step + 1,
                        "candidate": candidate,
                        "trial_leads": _lead_text(trial_leads),
                        **corruption_result,
                    }
                )

            step_candidates.append({**trial_row, "corruptions": result["corruptions"]})
            if best_row is None or trial_row["score"] > best_row["score"]:
                best_row = trial_row

            _refresh_outputs(results_dir, trial_rows, corruption_rows, history, config)
            print(
                f"  {trial_leads}: clean_f1={result['clean_f1']:.4f} "
                f"mean_drop={result['mean_f1_drop']:.4f} score={result['score']:.4f}"
            )

        if best_row is None:
            raise ValueError("No candidate lead was scored.")

        best_row["selected"] = True
        best_lead = best_row["candidate"]
        selected.append(best_lead)
        remaining.remove(best_lead)
        history.append(
            {
                "step": step + 1,
                "selected_lead": best_lead,
                "leads_so_far": list(selected),
                "score": best_row["score"],
                "candidates": step_candidates,
            }
        )
        _refresh_outputs(results_dir, trial_rows, corruption_rows, history, config)
        print(f"  --> selected {best_lead} | score={best_row['score']:.4f}")

    print(f"\nFinal lead order ({mode}): {selected}")
    return selected


if __name__ == "__main__":
    import fire

    fire.Fire(greedy_select)
