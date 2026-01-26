"""
Flashback: Scaled Anomaly Penalization (SAP)
Author: Preethi Amasa

Description:
    This script loads the frozen text embeddings and applies Scaled Anomaly
    Penalization (SAP) only to anomalous embeddings produced after Repulsive Prompting.
    It scales the magnitude of anomalous embeddings by alpha = 0.95, as described in
    the Flashback paper, to reduce anomaly bias at retrieval time.
"""

import os
import yaml
import torch
import numpy as np

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
ALPHA = 0.95  # Scaled Anomaly Penalization factor (from Flashback paper)
BASE_DIR = "/home/jacks.local/pamasa/thesis/src/memory"
INPUT_TEXT_EMB = os.path.join(BASE_DIR, "flashback_text_embeddings.npy")
INPUT_TN = os.path.join(BASE_DIR, "prompt_batches/flashback_TN.yaml")
INPUT_TA = os.path.join(BASE_DIR, "prompt_batches/flashback_TA.yaml")
OUTPUT_SCALED_EMB = os.path.join(BASE_DIR, "flashback_text_embeddings_SAP.npy")

# -----------------------------------------------------
# LOAD EMBEDDINGS
# -----------------------------------------------------
text_embeddings = np.load(INPUT_TEXT_EMB)
print(f"[INFO] Loaded text embeddings → {text_embeddings.shape}")

# -----------------------------------------------------
# LOAD NORMAL & ANOMALOUS TEMPLATE COUNTS
# -----------------------------------------------------
with open(INPUT_TN) as f:
    normal_templates = yaml.safe_load(f)
with open(INPUT_TA) as f:
    anomalous_templates = yaml.safe_load(f)

n_normal = len(normal_templates)
n_anomalous = len(anomalous_templates)
print(f"[INFO] Normal templates: {n_normal}, Anomalous templates: {n_anomalous}")

# -----------------------------------------------------
# VERIFY ORDER (normal first, anomalous second)
# -----------------------------------------------------
if text_embeddings.shape[0] != (n_normal + n_anomalous):
    raise ValueError(
        f"Mismatch: Total embeddings ({text_embeddings.shape[0]}) "
        f"≠ Normal+Anomalous ({n_normal + n_anomalous}). Check order."
    )

# -----------------------------------------------------
# APPLY SCALED ANOMALY PENALIZATION
# -----------------------------------------------------
print(f"[INFO] Applying Scaled Anomaly Penalization (α = {ALPHA}) to anomalous embeddings...")

scaled_embeddings = text_embeddings.copy()

# Assume normal embeddings are first, anomalous next
scaled_embeddings[n_normal:] *= ALPHA

np.save(OUTPUT_SCALED_EMB, scaled_embeddings)
print(f"[SAVED] SAP-adjusted embeddings → {OUTPUT_SCALED_EMB}")
print(f"[INFO] Shape: {scaled_embeddings.shape}")
