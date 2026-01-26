import json, yaml, os

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
INPUT_FILE = "/home/jacks.local/pamasa/thesis/src/memory/prompt_batches/flashback_memorymerged.json"
OUTPUT_DIR = "/home/jacks.local/pamasa/thesis/src/memory/prompt_batches"

# Output files (Flashback style)
OUTPUT_TN = os.path.join(OUTPUT_DIR, "flashback_TN.yaml")  # Normal Scene (T_N)
OUTPUT_TA = os.path.join(OUTPUT_DIR, "flashback_TA.yaml")  # Anomalous Scene (T_A)

# -----------------------------------------------------
# LOAD MERGED FILE
# -----------------------------------------------------
with open(INPUT_FILE) as f:
    data = json.load(f)

descriptions = data.get("descriptions", [])
print(f"[INFO] Loaded {len(descriptions)} pairs from merged file")

# -----------------------------------------------------
# APPLY REPULSIVE PROMPTING (Flashback style)
# -----------------------------------------------------
normal_templates = []
anomalous_templates = []

for d in descriptions:
    try:
        n_cat = d["normal"]["category"].strip()
        a_cat = d["anomalous"]["category"].strip()
        n_desc = d["normal"]["description"].strip()
        a_desc = d["anomalous"]["description"].strip()

        # Template: T_N (Normal Scene)
        tn_entry = {"Normal Scene": [f"Normal: {n_desc}"]}
        normal_templates.append(tn_entry)

        # Template: T_A (Anomalous Scene)
        ta_entry = {
            "Anomalous Scene": {
                "Action Category": a_cat,
                "Scene Description": f"Anomalous: {a_desc}"
            }
        }
        anomalous_templates.append(ta_entry)

    except Exception as e:
        print(f"[WARN] Skipping malformed entry: {e}")

# -----------------------------------------------------
# SAVE YAML FILES (Flashback kept them separate)
# -----------------------------------------------------
with open(OUTPUT_TN, "w") as f:
    yaml.dump(normal_templates, f, sort_keys=False, indent=4)

with open(OUTPUT_TA, "w") as f:
    yaml.dump(anomalous_templates, f, sort_keys=False, indent=4)

# -----------------------------------------------------
# LOG SUMMARY
# -----------------------------------------------------
print(f"\nRepulsive Prompting (Flashback-style) Complete")
print(f"Normal templates (T_N): {len(normal_templates)}  →  {OUTPUT_TN}")
print(f"Anomalous templates (T_A): {len(anomalous_templates)}  →  {OUTPUT_TA}")
