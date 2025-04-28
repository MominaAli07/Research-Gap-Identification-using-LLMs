import os
import pandas as pd
import re
import subprocess
from collections import Counter

# ------------------------------
# File Paths
# ------------------------------
PAPERS_FILE = "processed/papers_with_topics.csv"
TOPIC_INFO_FILE = "results/topics_info.csv"
OUTPUT_DIR = "local_gap_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# Local models to compare
# ------------------------------
# MODELS = ["llama3.2:latest", "openchat:latest", "zephyr:latest"]
MODELS = [
    "llama3.2:latest",
    "openchat:latest",
    "zephyr:latest",
    "gemma:2b",
    "mistral:latest",
    "neural-chat:latest",
    "phi:latest"
]


# ------------------------------
# Load data
# ------------------------------
papers_df = pd.read_csv(PAPERS_FILE)
topic_info = pd.read_csv(TOPIC_INFO_FILE)

# ------------------------------
# Helper: Extract limitations and future work
# ------------------------------
def extract_limitations_and_future_work(text):
    limitations, future_work = [], []
    sentences = re.split(r'[.!?]+', text)
    for s in sentences:
        if re.search(r'\b(limit|drawback|weakness|however|restricted to)\b', s, re.IGNORECASE):
            limitations.append(s.strip())
        if re.search(r'\b(future work|next step|should investigate|further research)\b', s, re.IGNORECASE):
            future_work.append(s.strip())
    return limitations[:3], future_work[:3]

# ------------------------------
# Generate gap candidates
# ------------------------------
print("🔍 Extracting gap candidates...")
topic_samples = {}
topic_scores = Counter()
topic_keywords = topic_info.set_index('Topic')['Representation'].to_dict()

for _, row in papers_df.iterrows():
    topic_id = row['topic_id']
    if topic_id == -1 or pd.isna(row.get('full_text')):
        continue
    lims, futs = extract_limitations_and_future_work(row['full_text'])
    if lims or futs:
        topic_scores[topic_id] += 1
        text_block = " ".join(lims + futs)
        if topic_id not in topic_samples:
            topic_samples[topic_id] = []
        topic_samples[topic_id].append(text_block)

# ------------------------------
# Compose prompts and run locally
# ------------------------------
for topic_id, score in topic_scores.most_common():
    if score == 0 or topic_id not in topic_samples:
        continue

    topic_name = topic_info[topic_info["Topic"] == topic_id]["Name"].values[0]
    keywords = topic_keywords.get(topic_id, "No keywords")
    evidence = " ".join(topic_samples[topic_id][:3])[:1000]

    # Build the prompt
    prompt = f"""You are an expert research analyst.

Topic: {topic_name}
Keywords: {keywords}
Evidence from recent papers:
{evidence}

Based on this evidence, write a detailed and academically styled research gap statement. Focus on what is missing in the current literature and suggest what future research should address. Only output the gap statement."""

    # Run each model
    for model in MODELS:
        print(f"\n🔁 {model} generating gap for Topic {topic_id}...")
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True
        )
        output = result.stdout.strip()

        # Save output
        filename = f"{model.replace(':', '_')}_topic_{topic_id}.txt"
        with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
            f.write(output)

        print(f"✅ {model} output saved: {filename}")

print("\n🎉 All models completed research gap generation.")
print("📄 Loaded papers:", len(papers_df))
print("📄 Loaded topics:", len(topic_info))