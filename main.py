import os
import glob
import fire
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
    num_workers=0,
    learning_rate=1e-3,
    max_epochs=10,
    num_classes=5, # NORM, MI, STTC, CD, HYP
    patience=5,
    gradient_clip_val=0.5,
    accelerator="auto",
    corruption=None,             # NEW: Terminal switch option for corruptions
    corruption_lead_idx=0        # NEW: Terminal option for targeted lead inversion
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
        corruption=corruption,               # Added here!
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
        gradient_clip_val=gradient_clip_val,
        callbacks=[early_stop_callback, ckpt_callback],
        logger=False
    )
    
    # Run clean training and baseline validations
    trainer.fit(task, datamodule=data_module)
    
    # Evaluate model performance (applies corruption here if specified)
    trainer.test(task, datamodule=data_module)

def test(
    exp_name,
    save_dir="results",
    csv_path="processed_ptbxl_metadata.csv",
    data_root=".",
    leads=None,
    batch_size=32,
    num_workers=0,
    accelerator="auto",
    corruption=None,             # NEW: Terminal switch option for testing checks
    corruption_lead_idx=0        # NEW: Terminal option for testing checks
):
    """Run an isolated evaluation loop using an established checkpoint folder and a custom corruption."""
    ckpt_dir = os.path.join(save_dir, exp_name, "ckpts")
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
    
    trainer = Trainer(accelerator=accelerator, logger=False)
    trainer.test(task, datamodule=data_module)

    return task.test_epoch_results

if __name__ == "__main__":
    # Expose both train and test routines to Fire CLI
    fire.Fire({
        "train": train,
        "test": test
    })
