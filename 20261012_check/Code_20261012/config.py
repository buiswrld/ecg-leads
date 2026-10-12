"""Shared configuration. Edit paths here for Colab."""
import os

SEED = 137
MAX_EPOCHS = 5           # full training budget
SCREENING_EPOCHS = 5     # candidate screening during greedy search
BATCH_SIZE = 32
NUM_WORKERS = 2          # Colab: 2 is usually the sweet spot

# Colab default: data in Drive. Override with env vars if you prefer.
DATA_ROOT = os.environ.get("PTBXL_ROOT", "/content/drive/MyDrive/ptbxl_project")
CSV_PATH  = os.environ.get("PTBXL_CSV", f"{DATA_ROOT}/processed_ptbxl_metadata.csv")

RESULTS_DIR = os.environ.get("ECG_RESULTS", "experiment_results")
CKPT_DIR    = os.environ.get("ECG_CKPTS", "checkpoints")

# Confidence thresholds for high-confidence FP / FN (ablatable).
HIGH_CONF_POS = 0.90     # confidently predicts disease
HIGH_CONF_NEG = 0.10     # confidently misses disease

CLASS_ORDER = ["NORM", "MI", "STTC", "CD", "HYP"]   # reporting order
