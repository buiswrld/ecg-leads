import os
import glob
import fire
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

# Local Imports
from dataset import PTBXLDataset
from classifier import ECGClassificationTask

def train(
    save_dir="results",
    exp_name="ecg_run",
    leads=None,               # None = all 12 leads; e.g. ["I", "II", "V1"]
    csv_path="processed_ptbxl_metadata.csv",
    data_root=".",
    batch_size=32,
    num_workers=0,
    learning_rate=1e-3,
    max_epochs=10,
    num_classes=5,            # NORM, MI, STTC, CD, HYP
    patience=5,
    gradient_clip_val=0.5,
    accelerator="auto",
):
    """Run the pipeline end-to-end: Download, build loaders, train and evaluate."""

    # -- Model --
    num_leads = 12 if leads is None else len(leads)
    task = ECGClassificationTask(
        num_leads=num_leads,
        num_classes=num_classes,
        learning_rate=learning_rate,
    )

    # -- Logger -- 
    logger = CSVLogger(
        save_dir=save_dir,
        name=exp_name
    )

    # -- Callbacks --
    ckpt_dir = os.path.join(logger.log_dir, "ckpts")
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
        logger=logger,
        enable_progress_bar=False, # in Colab the progress bar doesn't display correctly
    )
    trainer.fit(task)

    best_path = ckpt_callback.best_model_path

    if not best_path:
        raise ValueError("No best checkpoint found. Check val_loss logging.")
    
    trainer.test(
        model=task,
        ckpt_path=best_path
    )

if __name__ == "__main__":
    fire.Fire(train)