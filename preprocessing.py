import pandas as pd
import numpy as np
import wfdb
import ast
import gdown
import zipfile
import os
from pathlib import Path


# ------------------------------------------------------------
# Download and Unzip the PTBXL dataset
# ------------------------------------------------------------

file_id = "1f4sk1pCJ6SKK8M-TRpEhEiVqvLTHV80p"
output_zip = "download_file.zip"
extract_folder = "unzipped_files"
url = f"https://drive.google.com/uc?id={file_id}"

gdown.download(url, output_zip, quiet = False)

os.makedirs(extract_folder, exist_ok=True)
with zipfile.ZipFile(output_zip, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

path = './unzipped_files/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
sampling_rate=100


# ------------------------------------------------------------
# Load and Aggregate Metadata
# ------------------------------------------------------------

def aggregate_diagnostic(y_dic):
    """
    Map SCP-ECG codes to PTB-XL diagnostic superclasses
    using scp_statements.csv lookup table.
    """
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')

# Convert SCP codes from string to Python dictionary
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Keep only diagnostic SCP statements used for label mapping
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

# Assign diagnostic superclasses to each ECG record
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

Y.to_csv("processed_ptbxl_metadata.csv")


# ------------------------------------------------------------
# Load ECG waveform data
# ------------------------------------------------------------

def load_raw_data(df, sampling_rate, path):
    """
    Load raw ECG signals into a NumPy array.

    Returns:
        numpy array of shape (num_samples, 12 leads, signal_length)
    """
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

X = load_raw_data(Y, sampling_rate, path)

np.save("X_numpy_ndarray.npy", X)
