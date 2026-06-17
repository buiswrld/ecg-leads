"""
greedy_selection.py — Clean and robust greedy ECG lead selection.

Clean greedy:  score = clean_F1
Robust greedy: score = clean_F1 - lam * mean_F1_drop_across_corruptions

Usage:
    python greedy_selection.py                        # robust, up to 6 leads
    python greedy_selection.py --mode clean           # clean greedy
    python greedy_selection.py --max_leads 3 --lam 0.3
"""

import json
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from classifier import ECGClassificationTask
from dataset import PTBXLDataset
from metrics import compute_classification_metrics

ALL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# All 6 corruptions from the research proposal.
# Single-lead polarity inversion is applied to lead I (index 0).
CORRUPTIONS = [
    ("ra_la_swap",                    0),
    ("ra_ll_swap",                    0),
    ("la_ll_swap",                    0),
    ("v1_v2_swap",                    0),
    ("v2_v3_swap",                    0),
    ("single_lead_polarity_inversion", 0),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _test_loader(leads, mlb, corruption, corruption_lead_idx,
                 csv_path, data_root, batch_size, num_workers):
    ds = PTBXLDataset(
        csv_path=csv_path,
        data_root=data_root,
        split="test",
        leads=leads,
        mlb=mlb,
        corruption=corruption,
        corruption_lead_idx=corruption_lead_idx,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)


def _eval_f1(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["ecg"].to(device))
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"].cpu())
    metrics = compute_classification_metrics(
        torch.cat(all_logits), torch.cat(all_labels)
    )
    val = metrics["f1"].item()
    return 0.0 if val != val else val  # replace NaN with 0


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    leads: List[str],
    csv_path: str = "processed_ptbxl_metadata.csv",
    data_root: str = ".",
    batch_size: int = 32,
    num_workers: int = 0,
    learning_rate: float = 1e-3,
    max_epochs: int = 10,
    patience: int = 3,
    accelerator: str = "auto",
    ckpt_dir: str = "greedy_ckpts",
) -> Tuple[ECGClassificationTask, object]:
    """Train a model on *leads* and return (model, fitted_mlb)."""
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
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        gradient_clip_val=0.5,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
            ModelCheckpoint(dirpath=ckpt_dir, monitor="val_loss", mode="min",
                            save_top_k=1, filename="best"),
        ],
        logger=False,
        enable_progress_bar=True,
    )
    trainer.fit(task)
    mlb = task._train_ds.mlb
    return task, mlb


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_score(
    model: ECGClassificationTask,
    mlb,
    leads: List[str],
    mode: str,
    lam: float,
    csv_path: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict:
    """
    Evaluate model on clean test set and (if mode='robust') all corruptions.
    Returns dict with clean_f1, mean_f1_drop, and score.
    """
    clean_loader = _test_loader(leads, mlb, None, 0,
                                csv_path, data_root, batch_size, num_workers)
    clean_f1 = _eval_f1(model, clean_loader, device)

    if mode == "clean":
        return {"clean_f1": clean_f1, "mean_f1_drop": 0.0, "score": clean_f1}

    drops = []
    for corruption_name, lead_idx in CORRUPTIONS:
        loader = _test_loader(leads, mlb, corruption_name, lead_idx,
                              csv_path, data_root, batch_size, num_workers)
        corrupt_f1 = _eval_f1(model, loader, device)
        drops.append(max(0.0, clean_f1 - corrupt_f1))

    mean_drop = sum(drops) / len(drops)
    return {
        "clean_f1": clean_f1,
        "mean_f1_drop": mean_drop,
        "score": clean_f1 - lam * mean_drop,
    }


# ── Greedy loop ───────────────────────────────────────────────────────────────

def greedy_select(
    mode: str = "robust",
    max_leads: int = 6,
    lam: float = 0.5,
    csv_path: str = "processed_ptbxl_metadata.csv",
    data_root: str = ".",
    batch_size: int = 32,
    num_workers: int = 0,
    learning_rate: float = 1e-3,
    max_epochs: int = 10,
    patience: int = 3,
    accelerator: str = "auto",
    results_dir: str = "greedy_results",
) -> List[str]:
    """
    Run greedy lead selection and save results to results_dir.

    mode      : 'clean' or 'robust'
    max_leads : how many leads to select (1 to 12)
    lam       : robustness penalty weight (only used when mode='robust')
    """
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    remaining = list(ALL_LEADS)
    selected: List[str] = []
    history = []

    for step in range(min(max_leads, len(ALL_LEADS))):
        print(f"\n{'='*60}")
        print(f"Step {step + 1} | selected so far: {selected}")
        print(f"{'='*60}")

        best_lead = None
        best_score = float("-inf")
        step_candidates = []

        for candidate in remaining:
            trial_leads = selected + [candidate]
            print(f"\n  Trying leads: {trial_leads}")

            ckpt_dir = os.path.join(
                results_dir, f"step{step+1}_{candidate}_ckpts"
            )
            model, mlb = train_model(
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
            )
            model = model.to(device)

            result = compute_score(
                model, mlb, trial_leads, mode, lam,
                csv_path, data_root, batch_size, num_workers, device,
            )
            print(f"  clean_F1={result['clean_f1']:.4f}  "
                  f"mean_drop={result['mean_f1_drop']:.4f}  "
                  f"score={result['score']:.4f}")

            step_candidates.append({"candidate": candidate, **result})

            if result["score"] > best_score:
                best_score = result["score"]
                best_lead = candidate

        selected.append(best_lead)
        remaining.remove(best_lead)
        history.append({
            "step": step + 1,
            "selected_lead": best_lead,
            "leads_so_far": list(selected),
            "score": best_score,
            "candidates": step_candidates,
        })
        print(f"\n  --> Selected: {best_lead}  (score={best_score:.4f})")

    out_path = os.path.join(results_dir, f"greedy_{mode}_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "mode": mode,
            "lam": lam,
            "selected_leads": selected,
            "history": history,
        }, f, indent=2)

    print(f"\nDone. Results saved to {out_path}")
    print(f"Final lead order ({mode}): {selected}")
    return selected


if __name__ == "__main__":
    import fire
    fire.Fire(greedy_select)
