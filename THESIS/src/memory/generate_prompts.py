from openai import OpenAI
import os, time

# -----------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------
OUTPUT_DIR = "/home/jacks.local/pamasa/thesis/src/memory/prompt_batches"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "gpt-5"
PROMPTS_PER_BATCH = 5000        # number of pairs per batch
TOTAL_BATCHES = 50             # total batches to generate
SLEEP_BETWEEN_CALLS = 2.5       # delay between requests (rate-limit safety)

client = OpenAI()
print("[INFO] GPT-5 Flashback Prompt Generator Initialized")

# -----------------------------------------------------
# PROMPT TEMPLATE (PURE FLASHBACK VERSION)
# -----------------------------------------------------
CONTEXT_PROMPT = """
You are solving the video anomaly detection (VAD) problem
in a fancy way. As you know, anomalous events are rare but
their categories are diverse. You have to generate example
scene descriptions both for the anomalous events and normal
events. We will use these descriptions to decide if given
video clips contain anomalous events by choosing one of the
descriptions having the top similarity measured by a
multi-modal retrieval model. The descriptions should be short
and concise. The entire response should be in the provided
JSON format.
"""

FORMAT_PROMPT = """
{
  "descriptions": [
    {
      "normal": {
        "category": "<scene type>",
        "description": "<short normal scene>"
      },
      "anomalous": {
        "category": "<scene type>",
        "description": "<short anomalous scene>"
      }
    }
  ]
}
"""

# -----------------------------------------------------
# GENERATION FUNCTION
# -----------------------------------------------------
def generate_prompts(batch_id):
    """Generates a JSON batch of scene descriptions."""
    filename = os.path.join(OUTPUT_DIR, f"flashback_batch{batch_id:02d}.json")
    if os.path.exists(filename):
        print(f"[SKIP] {filename} already exists.")
        return

    instruction = (
        f"{CONTEXT_PROMPT}\n"
        f"Generate {PROMPTS_PER_BATCH} unique and diverse scene descriptions "
        f"for both normal and anomalous cases.\n"
        f"{FORMAT_PROMPT}"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a professional VAD prompt generator."},
                {"role": "user", "content": instruction}
            ]
        )

        text = response.choices[0].message.content.strip()
        with open(filename, "w") as f:
            f.write(text)

        print(f"[SAVED] {filename}")
    except Exception as e:
        print(f"[ERROR] Batch {batch_id}: {e}")

    time.sleep(SLEEP_BETWEEN_CALLS)

# -----------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------
for b in range(TOTAL_BATCHES):
    print(f"\n[RUN] 🚀 Generating Batch {b+1}/{TOTAL_BATCHES}")
    generate_prompts(b)

print("\n All batches completed. Safe to re-run; skips existing files.")
