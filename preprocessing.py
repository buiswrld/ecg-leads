"""
PTB-XL preprocessing.

  - Records with no diagnostic superclass are excluded.
  - Waveforms are stored as float32 (halves memory vs. the float64 default).
"""

import ast
import numpy as np
import pandas as pd
import wfdb

VERSION = "1.0.3"
SAMPLING_RATE = 100


def aggregate_diagnostic(scp_codes, agg_df):
    out = set()
    for code in scp_codes:
        if code in agg_df.index:
            cls = agg_df.loc[code, "diagnostic_class"]
            if pd.notna(cls):
                out.add(cls)
    return sorted(out)


def build(path):
    df = pd.read_csv(f"{path}ptbxl_database.csv", index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    agg = pd.read_csv(f"{path}scp_statements.csv", index_col=0)
    agg = agg[agg.diagnostic == 1]

    df["diagnostic_superclass"] = df.scp_codes.apply(lambda c: aggregate_diagnostic(c, agg))

    n_before = len(df)
    df = df[df["diagnostic_superclass"].map(len) > 0].copy()
    print(f"Excluded {n_before - len(df)} of {n_before} records with no diagnostic "
          f"superclass ({100*(n_before-len(df))/n_before:.1f}%); {len(df)} remain.")

    files = df.filename_lr if SAMPLING_RATE == 100 else df.filename_hr
    X = np.array([wfdb.rdsamp(path + f)[0] for f in files], dtype=np.float32)

    np.save("X_numpy_ndarray.npy", X)
    df.to_csv("processed_ptbxl_metadata.csv")
    print(f"Saved waveforms {X.shape} (float32) and metadata for {len(df)} records.")
    return X, df
