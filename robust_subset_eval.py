import os
import csv
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from classifier import ECGClassificationTask
from dataset import PTBXLDataset
from metrics import compute_classification_metrics
from greedy_lead import greedy_select, train_model

SUBSET_SIZES = [1, 3, 6]

def eval_clean(model, leads, mlb, csv_path, data_root, batch_size, num_workers, device):
    ds = PTBXLDataset(
        csv_path=csv_path,
        data_root=data_root,
        split="test",
        leads=leads,
        mlb=mlb,
        corruption=None,
        corruption_lead_idx=0,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["ecg"].to(device))
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"].cpu())
    results = compute_classification_metrics(
        torch.cat(all_logits), torch.cat(all_labels)
    )
    return results


def run_subsets(
    mode= "robust",
    lam= 0.5,
    csv_path= "processed_ptbxl_metadata.csv",
    data_root= ".",
    batch_size= 32,
    num_workers= 0,
    learning_rate= 1e-3,
    max_epochs= 10,
    patience= 3,
    accelerator= "auto",
    results_dir= "results/experiment2",
):
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Running greedy selection...")
    ordered_leads = greedy_select(
        mode=mode,
        max_leads=6,
        lam=lam,
        csv_path=csv_path,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        patience=patience,
        accelerator=accelerator,
        results_dir=results_dir,
    )

    print("\nTesting subsets on clean ECGs...")
    rows = []
    for size in SUBSET_SIZES:
        leads = ordered_leads[:size]
        print(f"\nSubset size {size}: {leads}")

        ckpt_dir = os.path.join(results_dir, f"subset_{size}_ckpts")
        model, mlb = train_model(
            leads=leads,
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

        metrics = eval_clean(model, leads, mlb, csv_path, data_root,
                             batch_size, num_workers, device)
        auroc = round(metrics['auroc'].item(), 4)
        auprc = round(metrics['auprc'].item(), 4)
        f1 = round(metrics['f1'].item(), 4)
        print(f"  AUROC={auroc}  AUPRC={auprc}  F1={f1}")
        rows.append({
            "subset_size": size,
            "leads": str(leads),
            "auroc": auroc,
            "auprc": auprc,
            "f1": f1,
        })

    out_path = os.path.join(results_dir, "greedy_selection_robust_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subset_size", "leads", "auroc", "auprc", "f1"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {out_path}")
    print("\nDone!")


if __name__ == "__main__":
    import fire
    fire.Fire(run_subsets)
