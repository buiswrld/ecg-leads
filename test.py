import glob
import os
import fire
import pytorch_lightning as pl

from classifier import ECGClassificationTask


def test(
    exp_name,
    save_dir="results",
    csv_path="processed_ptbxl_metadata.csv",
    data_root=".",
    leads=None,
    batch_size=32,
    num_workers=0,
    accelerator="auto",
    corruption=None,
    corruption_lead_idx=0,
):
    """Evaluate a saved checkpoint, optionally with a corruption applied."""
    ckpt_dir  = os.path.join(save_dir, exp_name, "ckpts")
    ckpt_path = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))

    if not ckpt_path:
        raise ValueError(f"No checkpoint found in {ckpt_dir}")
    ckpt_path = ckpt_path[0]

    # 1. Load checkpoint, then override corruption settings for this test run.
    task = ECGClassificationTask.load_from_checkpoint(
        ckpt_path,
        csv_path=csv_path,
        data_root=data_root,
        leads=leads,
        batch_size=batch_size,
        num_workers=num_workers,
        corruption=corruption,
        corruption_lead_idx=corruption_lead_idx,
    )

    # 2. Force Lightning to build the data loaders ahead of time for validation
    task.setup(stage="test")
    test_dataset = task._test_ds

    # 3. VERIFICATION HOOK: Confirm and visualize the observable difference
    print("\n" + "="*60)
    print(f"DATA VALIDATION HOOK | Applied Corruption: {corruption}")
    print("="*60)

    if len(test_dataset) > 0:
        # Pull the very first raw item from our freshly configured dataset
        sample_batch = test_dataset[0]
        ecg_tensor = sample_batch["ecg"] # Shape: (num_leads, num_samples)
        
        # Display baseline confirmation characteristics
        print(f"-> Data shape: {list(ecg_tensor.shape)}")
        
        if corruption == "single_lead_polarity_inversion":
            idx = corruption_lead_idx
            print(f"-> Sample values for Lead Index {idx} (First 5 steps):")
            print(f"   {ecg_tensor[idx, :5].tolist()}")
            print("-> Verification: Check if these sign values are inverted relative to baseline.")
            
        elif corruption in ["ra_la_swap", "ra_ll_swap", "la_ll_swap", "v1_v2_swap", "v2_v3_swap"]:
            print("-> Active substitution map applied.")
            print(f"-> Sample values for Lead Index 0 (First 5 steps): {ecg_tensor[0, :5].tolist()}")
            print(f"-> Sample values for Lead Index 1 (First 5 steps): {ecg_tensor[1, :5].tolist()}")
        else:
            print("-> Baseline run. Data passing into model completely clean.")
            print(f"-> Sample values for Lead Index 0 (First 5 steps): {ecg_tensor[0, :5].tolist()}")
            print(f"-> Sample values for Lead Index 1 (First 5 steps): {ecg_tensor[1, :5].tolist()}")
    else:
        print("Warning: Test dataset contains no samples.")
    print("="*60 + "\n")

    # 4. Hand execution tracking back over to the automated Lightning pipeline
    trainer = pl.Trainer(accelerator=accelerator)
    trainer.test(task)


if __name__ == "__main__":
    fire.Fire(test)