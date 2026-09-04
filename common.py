"""
Shared metrics, evaluation, and CSV reporting for all experiments.
"""

import os
import shutil
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import config
from ecg import LEAD_ORDER, LEAD_IDX, CORRUPTION_AFFECTED_LEADS
from datamodule import PTBXLDataModule
from classifier import ECGClassificationTask


# ---------------------------------------------------------------- metrics
def _safe_auroc(y, p):
    """AUROC via rank statistic; nan if only one class present."""
    pos, neg = y == 1, y == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _safe_auprc(y, p):
    """Average precision; nan if no positives."""
    if (y == 1).sum() == 0:
        return float("nan")
    order = np.argsort(-p)
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / ys.sum())


def class_metrics(y, p, threshold=0.5):
    """All per-class diagnostic metrics for one class."""
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    return {"auroc": _safe_auroc(y, p), "auprc": _safe_auprc(y, p), "f1": f1,
            "sensitivity": sens, "specificity": spec, "fnr": 1.0 - sens}


def high_conf_rates(y, p):
    """
    high_conf_fpr: of truly-negative labels, how often does the model output a
                   confident positive probability (>= HIGH_CONF_POS)?
    high_conf_fnr: of truly-positive labels, how often does the model output a
                   confident negative probability (<= HIGH_CONF_NEG)?
    """
    neg, pos = y == 0, y == 1
    fpr = float((p[neg] >= config.HIGH_CONF_POS).mean()) if neg.sum() else 0.0
    fnr = float((p[pos] <= config.HIGH_CONF_NEG).mean()) if pos.sum() else 0.0
    return fpr, fnr


def instability(clean_p, corr_p, threshold=0.5):
    """Paired clean-vs-corrupted stability metrics."""
    shift = float(np.abs(corr_p - clean_p).mean())
    flip = float(((clean_p >= threshold) != (corr_p >= threshold)).mean())
    return shift, flip


def build_metric_rows(labels, probs, class_names, clean_probs=None,
                      test_loss=None, **extra):
    """
    One row per class plus an ALL row (macro average), in config.CLASS_ORDER.
    If clean_probs is given, corruption metrics are included too.
    """
    rows, acc = [], {k: [] for k in
                     ["auroc", "auprc", "f1", "sensitivity", "specificity", "fnr",
                      "probability_shift", "prediction_flip_rate",
                      "high_conf_fpr", "high_conf_fnr"]}
    order = [c for c in config.CLASS_ORDER if c in class_names]
    for cname in order:
        j = class_names.index(cname)
        y, p = labels[:, j], probs[:, j]
        m = class_metrics(y, p)
        hf, hn = high_conf_rates(y, p)
        m["high_conf_fpr"], m["high_conf_fnr"] = hf, hn
        if clean_probs is not None:
            s, f = instability(clean_probs[:, j], p)
            m["probability_shift"], m["prediction_flip_rate"] = s, f
        else:
            m["probability_shift"] = m["prediction_flip_rate"] = 0.0
        row = {**extra, "class": cname}
        if test_loss is not None:
            row["test_loss"] = test_loss
        row.update(m)
        rows.append(row)
        for k in acc:
            acc[k].append(m[k])

    all_row = {**extra, "class": "ALL"}
    if test_loss is not None:
        all_row["test_loss"] = test_loss
    all_row.update({k: float(np.nanmean(v)) for k, v in acc.items()})
    rows.append(all_row)
    return rows


def predictions_frame(sample_ids, labels, probs, class_names, threshold=0.5, **extra):
    """Predictions list: true labels + probabilities (+ thresholded preds)."""
    order = [c for c in config.CLASS_ORDER if c in class_names]
    data = {"sample_id": sample_ids}
    data.update(extra)
    for c in order:
        data[c] = labels[:, class_names.index(c)].astype(int)
    for c in order:
        data[f"prob_{c}"] = probs[:, class_names.index(c)]
    for c in order:
        data[f"pred_{c}"] = (probs[:, class_names.index(c)] >= threshold).astype(int)
    return pd.DataFrame(data)


# ---------------------------------------------------------------- runner
def _dm(leads=None, corruption=None, corruption_lead=None):
    return PTBXLDataModule(csv_path=config.CSV_PATH, data_root=config.DATA_ROOT,
                           leads=leads, corruption=corruption,
                           corruption_lead=corruption_lead,
                           batch_size=config.BATCH_SIZE,
                           num_workers=config.NUM_WORKERS)


def _trainer(max_epochs=None, callbacks=None):
    """accelerator='auto' picks the Colab GPU when present, CPU otherwise."""
    kw = dict(accelerator="auto", devices="auto", logger=False,
              enable_progress_bar=False, enable_model_summary=False)
    if max_epochs is not None:
        kw.update(max_epochs=max_epochs, callbacks=callbacks or [],
                  gradient_clip_val=0.5)
    return pl.Trainer(**kw)


_CKPTS = {}


def train_model(exp_name, leads=None, max_epochs=None, seed=config.SEED):
    """Train on the clean training set. Returns the best checkpoint path."""
    max_epochs = max_epochs or config.MAX_EPOCHS
    pl.seed_everything(seed, workers=True)
    dm = _dm(leads=leads)
    dm.setup()
    task = ECGClassificationTask(num_leads=12 if leads is None else len(leads),
                                 num_classes=5, max_epochs=max_epochs,
                                 class_names=dm.class_names)
    ckpt_dir = os.path.join(config.CKPT_DIR, exp_name)
    cb = ModelCheckpoint(dirpath=ckpt_dir, filename="best",
                         monitor="val_loss", mode="min", save_top_k=1)
    es = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    _trainer(max_epochs, [cb, es]).fit(task, datamodule=dm)
    _CKPTS[exp_name] = cb.best_model_path
    return cb.best_model_path


def evaluate(exp_name, leads=None, split="test", corruption=None, corruption_lead=None):
    """
    Evaluate a trained checkpoint. split='val' (fold 9) for any SELECTION
    decision; split='test' (fold 10) only for final reporting.
    Returns dict with labels, probs, sample_ids, class_names, test_loss.
    """
    path = _CKPTS.get(exp_name)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint for '{exp_name}'; train first.")
    task = ECGClassificationTask.load_from_checkpoint(path)
    expected = 12 if leads is None else len(leads)
    if task.hparams.num_leads != expected:
        raise ValueError(f"'{exp_name}' has num_leads={task.hparams.num_leads}, "
                         f"{expected} requested.")
    dm = _dm(leads=leads, corruption=corruption, corruption_lead=corruption_lead)
    dm.setup()
    loader = dm.val_dataloader() if split == "val" else dm.test_dataloader()
    ds = dm.val_ds if split == "val" else dm.test_ds
    _trainer().test(task, dataloaders=loader, verbose=False)
    r = task.last_results
    return {"labels": r["labels"].numpy(), "probs": r["probs"].numpy(),
            "sample_ids": np.asarray(ds.df.index if ds.df.index.name else range(len(ds))),
            "class_names": dm.class_names, "test_loss": r.get("loss", float("nan"))}


def macro_f1(res):
    """Macro-F1 across classes -- the greedy selection score."""
    cn = res["class_names"]
    return float(np.mean([class_metrics(res["labels"][:, j], res["probs"][:, j])["f1"]
                          for j in range(len(cn))]))


def save(df, name):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    p = os.path.join(config.RESULTS_DIR, name)
    (df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)).to_csv(p, index=False)
    print(f"  wrote {p}")


def cleanup(exp_name):
    """
    Delete a screening checkpoint. Greedy search generates one per candidate
    (57 per path); without this a full run leaves >100 GB behind. Selected
    subsets are retrained from scratch, so screening checkpoints are disposable.
    """
    shutil.rmtree(os.path.join(config.CKPT_DIR, exp_name), ignore_errors=True)
    _CKPTS.pop(exp_name, None)


def lead_idx(name):
    return LEAD_IDX[name] if name is not None else None


def is_applicable(corruption, subset, corruption_lead=None):
    """Does this corruption alter any lead retained in the subset?"""
    if corruption is None:
        return True
    if corruption == "single_lead_polarity_inversion":
        return corruption_lead in subset
    return len(set(CORRUPTION_AFFECTED_LEADS[corruption]) & set(subset)) > 0
