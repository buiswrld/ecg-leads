import os
import glob
import fire
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Local Imports
from datamodule import PTBXLDataModule  # Integrated DataModule
from classifier import ECGClassificationTask

def train(
    save_dir="results",
    exp_name="ecg_run",
    leads=None, # None = all 12 leads; e.g. ["I", "II", "V1"]
    csv_path="processed_ptbxl_metadata.csv",
    data_root=".",
    batch_size=32,
    num_workers=2, #0
    learning_rate=1e-3,
    max_epochs=10,
    num_classes=5, # NORM, MI, STTC, CD, HYP
    patience=5,
    gradient_clip_val=0.5,
    accelerator="auto",
    corruption=None,             # Terminal switch option for corruptions
    corruption_lead_idx=0        # Terminal option for targeted lead inversion
):
    """Run the pipeline end-to-end: Initialize DataModule, train and evaluate."""
    
    # -- Data Module --
    # Automatically manages train, val, and test splits along with binarizer sharing
    data_module = PTBXLDataModule(
        csv_path=csv_path,
        data_root=data_root,
        batch_size=batch_size,
        leads=leads,
        num_workers=num_workers,
        corruption=corruption,
        corruption_lead_idx=corruption_lead_idx
    )

    # -- Model --
    num_leads = 12 if leads is None else len(leads)
    task = ECGClassificationTask(
        num_leads=num_leads,
        num_classes=num_classes,
        learning_rate=learning_rate,
        max_epochs=max_epochs,       # Pass max_epochs here
        warmup_epochs=1,
        corruption=corruption,
        corruption_lead_idx=corruption_lead_idx
    )

    # -- Callbacks --
    ckpt_dir = os.path.join(save_dir, exp_name, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best",
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
    )

    # -- Trainer --
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        precision="16-mixed",
        gradient_clip_val=gradient_clip_val,
        callbacks=[early_stop_callback, ckpt_callback],
        enable_progress_bar=False, # in Colab the progress bar doesn't display correctly
        logger=False
    )
    
    # Run clean training and baseline validations
    trainer.fit(task, datamodule=data_module)
    
    # Evaluate model performance (applies corruption here if specified)
    trainer.test(task, datamodule=data_module)

def test(
    exp_name,
    save_dir="results",
    save_results=False,
    results_file="results.csv",
    ckpt_exp=None,  # if you want a checkpoint from a different experiment
    csv_path="processed_ptbxl_metadata.csv",
    data_root=".",
    leads=None,
    batch_size=32,
    num_workers=2, #0
    accelerator="auto",
    corruption=None,             # Terminal switch option for testing checks
    corruption_lead_idx=0        # Terminal option for testing checks
):
    """Run an isolated evaluation loop using an established checkpoint folder and a custom corruption."""
    if ckpt_exp is None:
        ckpt_exp = exp_name
    ckpt_dir = os.path.join(save_dir, ckpt_exp, "ckpts")
    ckpt_path = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpt_path:
        raise ValueError(f"No checkpoint found in {ckpt_dir}")
    ckpt_path = ckpt_path[0]
    
    # Load your pre-trained model weights cleanly
    task = ECGClassificationTask.load_from_checkpoint(ckpt_path)

    #task.hparams.corruption = corruption
    #task.hparams.corruption_lead_idx = corruption_lead_idx
    
    # Setup the isolated evaluation data environment with the corruption activated
    data_module = PTBXLDataModule(
        csv_path=csv_path,
        data_root=data_root,
        batch_size=batch_size,
        leads=leads,
        num_workers=num_workers,
        corruption=corruption,
        corruption_lead_idx=corruption_lead_idx
        #corruption=task.hparams.corruption,               # Pulled out of the classifier!
        #corruption_lead_idx=task.hparams.corruption_lead_idx
    )
    
    trainer = Trainer(accelerator=accelerator, precision="16-mixed", logger=False)
    results = trainer.test(task, datamodule=data_module)
    if save_results:
        metrics = results[0]

        row = {
            "mode": "clean" if corruption is None else "corrupted",
            "corruption": corruption,
            "corruption_lead_idx": (
                corruption_lead_idx
                if corruption == "single_lead_polarity_inversion"
                else None
            ),
            "subset_size": 12 if leads is None else len(leads),
            "selected_leads": str(leads) if leads is not None else "all",
            **metrics,
        }

        summary_path = os.path.join(save_dir, exp_name, results_file)
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)

        pd.DataFrame([row]).to_csv(
            summary_path,
            mode="a", # append
            header=not os.path.exists(summary_path), # write header only first time
            index=False
        )

    return task.test_epoch_results

if __name__ == "__main__":
    # Expose both train and test routines to Fire CLI
    fire.Fire({
        "train": train,
        "test": test
    })
