import json
import logging
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALPHA = 0.95  # SAP
CHECKPOINT_PATH = Path(".checkpoints") / "imagebind_huge.pth"


def load_imagebind(device):
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType
    from imagebind import data as imagebind_data

    logger.info("Loading ImageBind...")
    model = imagebind_model.imagebind_huge(pretrained=False)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    logger.info(f"ImageBind loaded from {CHECKPOINT_PATH}")
    return model, ModalityType, imagebind_data


def encode_texts(texts, model, ModalityType, imagebind_data, device, batch_size=256):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        inputs = {ModalityType.TEXT: imagebind_data.load_and_transform_text(batch, device)}
        with torch.no_grad():
            embeddings = model(inputs)
            embs = F.normalize(embeddings[ModalityType.TEXT], dim=-1)
        all_embeddings.append(embs.cpu().float().numpy())
        logger.info(f"  Encoded {min(i + batch_size, len(texts)):,}/{len(texts):,} texts...")
    return np.concatenate(all_embeddings, axis=0)


def encode_single_text(text, model, ModalityType, imagebind_data, device):
    arr = encode_texts([text], model, ModalityType, imagebind_data, device, batch_size=1)
    return arr[0]


def apply_sap(ZA, alpha=ALPHA):
    ZA_scaled = alpha * ZA
    logger.info(
        f"SAP applied: alpha={alpha}. "
        f"Mean L2 before: {np.linalg.norm(ZA, axis=1).mean():.4f}, "
        f"after: {np.linalg.norm(ZA_scaled, axis=1).mean():.4f}"
    )
    return ZA_scaled


def apply_sap_single(embedding, alpha=ALPHA):
    return (alpha * embedding).astype(np.float32)


def load_memory_json(memory_path):
    memory_path = Path(memory_path)
    with open(memory_path, encoding="utf-8") as f:
        return json.load(f)


def build_custom_anomaly_embedding(
    text,
    model,
    ModalityType,
    imagebind_data,
    device,
    alpha=ALPHA,
    apply_prompt_template=False,
    template_prefix="Anomalous event:"
):
    """
    Runtime helper for user-defined anomaly events.
    Encodes one user text using the frozen text encoder.
    """
    if apply_prompt_template:
        text_to_encode = f"{template_prefix} {text.strip()}"
    else:
        text_to_encode = text.strip()

    emb = encode_single_text(text_to_encode, model, ModalityType, imagebind_data, device)
    emb = apply_sap_single(emb, alpha=alpha)
    return emb, text_to_encode


def encode_memory(memory_path, output_dir, alpha=ALPHA, batch_size=256, device_str="cuda"):
    from RP import apply_repulsive_prompting

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info(f"Loading memory from '{memory_path}'...")
    memory = load_memory_json(memory_path)

    captions_normal      = memory["captions_normal"]
    captions_anomalous   = memory["captions_anomalous"]
    categories_normal    = memory["categories_normal"]
    categories_anomalous = memory["categories_anomalous"]
    NN = len(captions_normal)
    NA = len(captions_anomalous)
    logger.info(f"Memory: {NN:,} normal + {NA:,} anomalous captions.")

    logger.info("Applying Repulsive Prompting (RP)...")
    templated_normal, templated_anomalous = apply_repulsive_prompting(
        captions_normal, captions_anomalous,
        categories_normal, categories_anomalous
    )

    model, ModalityType, imagebind_data = load_imagebind(device)

    logger.info(f"Encoding {NN:,} normal captions → ZN...")
    ZN = encode_texts(templated_normal, model, ModalityType, imagebind_data, device, batch_size)
    logger.info(f"ZN shape: {ZN.shape}")

    logger.info(f"Encoding {NA:,} anomalous captions → ZA...")
    ZA = encode_texts(templated_anomalous, model, ModalityType, imagebind_data, device, batch_size)
    logger.info(f"ZA shape: {ZA.shape}")

    logger.info(f"Applying SAP (alpha={alpha})...")
    ZA_penalized = apply_sap(ZA, alpha=alpha)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    zn_path = out_path / "embeddings_normal.npy"
    za_path = out_path / "embeddings_anomalous.npy"
    np.save(zn_path, ZN.astype(np.float32))
    np.save(za_path, ZA_penalized.astype(np.float32))
    logger.info(f"Saved ZN → '{zn_path}'")
    logger.info(f"Saved ZA → '{za_path}'")

    meta = {
        "encoder": "imagebind",
        "alpha": alpha,
        "NN": NN,
        "NA": NA,
        "embedding_dim": int(ZN.shape[1]),
        "ZN_path": str(zn_path.resolve()),
        "ZA_path": str(za_path.resolve()),
        "memory_path": str(Path(memory_path).resolve()),
        "captions_normal": captions_normal,
        "captions_anomalous": captions_anomalous,
        "categories_normal": categories_normal,
        "categories_anomalous": categories_anomalous,
        "labels": [0] * NN + [1] * NA,
    }
    meta_path = out_path / "memory_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metadata → '{meta_path}'")
    logger.info(f"Total embedding storage: {(ZN.nbytes + ZA_penalized.nbytes) / 1e9:.2f} GB")
    return ZN, ZA_penalized, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory_path", type=str, default=str(Path(__file__).resolve().parent / "memory.json"))
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent / "embeddings" / "stage1"))
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    encode_memory(args.memory_path, args.output_dir, args.alpha, args.batch_size, args.device)


if __name__ == "__main__":
    main()