import json, glob, os

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
INPUT_DIR = "/home/jacks.local/pamasa/thesis/src/memory/prompt_batches"
OUTPUT_FILE = os.path.join(INPUT_DIR, "flashback_memorymerged.json")

# -----------------------------------------------------
# MERGE ALL BATCH FILES (FORMAT PROMPT PF)
# -----------------------------------------------------
merged_descriptions = []

for path in sorted(glob.glob(os.path.join(INPUT_DIR, "flashback_batch*.json"))):
    try:
        with open(path) as f:
            data = json.load(f)
            if "descriptions" in data:
                merged_descriptions.extend(data["descriptions"])
        print(f"[MERGED] {os.path.basename(path)}")
    except Exception as e:
        print(f"[SKIP] {path}: {e}")

# -----------------------------------------------------
# SAVE MERGED FILE
# -----------------------------------------------------
merged_output = {"descriptions": merged_descriptions}

with open(OUTPUT_FILE, "w") as f:
    json.dump(merged_output, f, indent=2)

print(f"\nMerged {len(merged_descriptions)} pairs total.")
print(f"📘 Output saved to: {OUTPUT_FILE}")
