import os, yaml, torch, numpy as np
from tqdm import tqdm
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind import data as imagebind_data 

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

INPUT_TN = "/home/jacks.local/pamasa/thesis/src/memory/prompt_batches/flashback_TN.yaml"
INPUT_TA = "/home/jacks.local/pamasa/thesis/src/memory/prompt_batches/flashback_TA.yaml"

OUTPUT_TXT = "/home/jacks.local/pamasa/thesis/src/memory/flashback_captions.txt"
OUTPUT_EMB = "/home/jacks.local/pamasa/thesis/src/memory/flashback_text_embeddings.npy"

# -----------------------------------------------------
# LOAD YAML CAPTIONS
# -----------------------------------------------------
def load_yaml_captions(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    captions = []
    for item in data:
        if "Normal Scene" in item:
            captions.extend(item["Normal Scene"])
        elif "Anomalous Scene" in item:
            captions.append(item["Anomalous Scene"]["Scene Description"])
    return captions

captions_normal = load_yaml_captions(INPUT_TN)
captions_anomalous = load_yaml_captions(INPUT_TA)
captions = captions_normal + captions_anomalous

print(f"[INFO] Loaded {len(captions)} total captions.")
with open(OUTPUT_TXT, "w") as f:
    f.write("\n".join(captions))
print(f"[SAVED] Captions written to: {OUTPUT_TXT}")

# -----------------------------------------------------
# INITIALIZE FROZEN IMAGEBIND MODEL
# -----------------------------------------------------
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(DEVICE)
print("[INFO] Loaded frozen ImageBind model.")

# -----------------------------------------------------
# ENCODE TEXT (FIXED)
# -----------------------------------------------------
BATCH_SIZE = 256
embeddings = []

for i in tqdm(range(0, len(captions), BATCH_SIZE), desc="Encoding text"):
    batch = captions[i:i+BATCH_SIZE]
    with torch.no_grad():
        inputs = {ModalityType.TEXT: imagebind_data.load_and_transform_text(batch, DEVICE)}
        text_emb = model(inputs)[ModalityType.TEXT]
        text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=-1)
        embeddings.append(text_emb.cpu())

embeddings = torch.cat(embeddings, dim=0)
np.save(OUTPUT_EMB, embeddings.numpy())

print(f"[SAVED] Text embeddings: {OUTPUT_EMB}")
print(f"[INFO] Embedding shape: {embeddings.shape}")
print("Text embedding extraction complete.")
