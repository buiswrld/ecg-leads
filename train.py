import os
import glob
import fire
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

# Local Imports
from dataset import PTBXLDataset
from model import ECGClassificationTask

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

    # -- Datasets & loaders --
    train_dataset = PTBXLDataset(csv_path=csv_path, data_root=data_root, split="train", leads=leads)
    val_dataset   = PTBXLDataset(csv_path=csv_path, data_root=data_root, split="val",   leads=leads, mlb=train_dataset.mlb)
    test_dataset  = PTBXLDataset(csv_path=csv_path, data_root=data_root, split="test",  leads=leads, mlb=train_dataset.mlb)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

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
    trainer.fit(task, train_loader, val_loader)
    trainer.test(task, test_loader)

def test(
    exp_name,
    save_dir="results",
    csv_path="processed_ptbxl_metadata.csv",
    data_root=".",
    leads=None,
    batch_size=32,
    num_workers=0,
    accelerator="auto",
):
    """Run an isolated evaluation loop using an established checkpoint folder."""
    ckpt_dir  = os.path.join(save_dir, exp_name, "ckpts")
    ckpt_path = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))

    if not ckpt_path:
        raise ValueError(f"No checkpoint found in {ckpt_dir}")
    ckpt_path = ckpt_path[0]

    task = ECGClassificationTask.load_from_checkpoint(ckpt_path)

    train_dataset = PTBXLDataset(csv_path=csv_path, data_root=data_root, split="train", leads=leads)
    test_dataset  = PTBXLDataset(csv_path=csv_path, data_root=data_root, split="test",  leads=leads, mlb=train_dataset.mlb)
    test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    trainer = Trainer(accelerator=accelerator)
    trainer.test(task, test_loader)

if __name__ == "__main__":
    fire.Fire(train)