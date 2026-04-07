import json
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def merge_batches(input_dir: str, output_path: str, max_pairs: int = 1_000_000):
    in_path = Path(input_dir)
    batch_files = sorted(in_path.glob("flashback_batch*.json"))
    if not batch_files:
        batch_files = sorted(in_path.glob("ai_event.json"))

    if not batch_files:
        raise FileNotFoundError(
            f"No flashback_batch*.json or ai_event.json files found in '{input_dir}'"
        )

    logger.info(f"Found {len(batch_files)} batch files in '{input_dir}'.")

    # Accumulators 
    captions_normal:    list[str] = []   # CN
    captions_anomalous: list[str] = []   # CA
    categories_normal:  list[str] = []   # KN
    categories_anomalous: list[str] = [] # KA

    total_loaded = 0
    skipped = 0

    for bf in batch_files:
        if total_loaded >= max_pairs:
            break

        with open(bf, encoding="utf-8") as f:
            raw = f.read().strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                batch = parsed
            elif isinstance(parsed, dict):
                batch = parsed.get("descriptions", parsed.get("prompts", []))
            else:
                batch = []
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse {bf.name}: {e}")
            skipped += 1
            continue

        for entry in batch:
            if total_loaded >= max_pairs:
                break

            try:
                n_cat  = entry["normal"]["category"].strip()
                n_desc = entry["normal"]["description"].strip()
                a_cat  = entry["anomalous"]["category"].strip()
                a_desc = entry["anomalous"]["description"].strip()

                if not n_desc or not a_desc:
                    skipped += 1
                    continue

                captions_normal.append(n_desc)
                categories_normal.append(n_cat)
                captions_anomalous.append(a_desc)
                categories_anomalous.append(a_cat)
                total_loaded += 1

            except (KeyError, AttributeError):
                skipped += 1
                continue

    NN = len(captions_normal)    # Number of normal captions
    NA = len(captions_anomalous) # Number of anomalous captions

    logger.info(f"Loaded {NN} normal and {NA} anomalous captions. Skipped {skipped} invalid entries.")

    C = captions_normal + captions_anomalous      # All captions
    K = categories_normal + categories_anomalous  # All categories
    Y = [0] * NN + [1] * NA                       # Binary anomaly flags

    assert len(C) == len(K) == len(Y), "Mismatch in lengths after merge."

    memory = {
        "meta": {
            "total_pairs": total_loaded,
            "NN": NN,
            "NA": NA,
            "source_dir": str(in_path.resolve()),
        },
        "captions_normal":      captions_normal,      # CN
        "captions_anomalous":   captions_anomalous,   # CA
        "categories_normal":    categories_normal,    # KN
        "categories_anomalous": categories_anomalous, # KA
        "captions":   C,  # C  = CN ⊕ CA
        "categories": K,  # K  = KN ⊕ KA
        "labels":     Y,  # Y  = (0,...,0, 1,...,1)
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Merged memory saved to '{out}'. "
        f"Total captions: {len(C):,} ({NN:,} normal + {NA:,} anomalous)."
    )
    return memory


def print_stats(memory: dict):
    print("\n=== Pseudo-Scene Memory Statistics ===")
    print(f"  Normal captions   (NN): {memory['meta']['NN']:>10,}")
    print(f"  Anomalous captions(NA): {memory['meta']['NA']:>10,}")
    print(f"  Total captions       : {len(memory['captions']):>10,}")
    print(f"  Total labels         : {len(memory['labels']):>10,}")
    print(f"  Label check (0s/1s) : {memory['labels'].count(0):,} / {memory['labels'].count(1):,}")
    print()

    print("  --- Sample Normal Caption ---")
    if memory["captions_normal"]:
        print(f"  Category: {memory['categories_normal'][0]}")
        print(f"  Caption : {memory['captions_normal'][0]}")
    print()
    print("  --- Sample Anomalous Caption ---")
    if memory["captions_anomalous"]:
        print(f"  Category: {memory['categories_anomalous'][0]}")
        print(f"  Caption : {memory['captions_anomalous'][0]}")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(
        description="Merge GPT batch caption files into a unified pseudo-scene memory."
    )
    parser.add_argument(
        "--input_dir", type=str, default=str(Path(__file__).resolve().parent),
        help="Directory containing flashback_batch*.json or ai_event.json files."
    )
    parser.add_argument(
        "--output_path", type=str, default=str(Path(__file__).resolve().parent / "memory.json"),
        help="Output path for the merged memory JSON."
    )
    parser.add_argument(
        "--max_pairs", type=int, default=1_000_000,
        help="Maximum number of pairs to include (paper uses 1M)."
    )
    args = parser.parse_args()

    memory = merge_batches(
        input_dir=args.input_dir,
        output_path=args.output_path,
        max_pairs=args.max_pairs,
    )
    print_stats(memory)


if __name__ == "__main__":
    main()