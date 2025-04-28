import os
import pandas as pd

OUTPUT_FOLDER = "local_gap_outputs"

# Dictionary to store model outputs by topic
gap_data = {}

# Loop through all text files
for fname in os.listdir(OUTPUT_FOLDER):
    if fname.endswith(".txt") and "_topic_" in fname:
        parts = fname.replace(".txt", "").split("_topic_")
        model_name = parts[0]                # e.g., 'llama3.2_latest'
        topic_id = int(parts[1])             # e.g., 0 or 1

        with open(os.path.join(OUTPUT_FOLDER, fname), "r") as f:
            gap_text = f.read().strip()

        gap_data.setdefault(topic_id, {})[model_name] = gap_text

# Build a comparison table
rows = []
all_models = set()

for topic_id, model_outputs in gap_data.items():
    row = {"topic_id": topic_id}
    for model, gap_text in model_outputs.items():
        row[model] = gap_text
        all_models.add(model)
    rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)
df = df[["topic_id"] + sorted(all_models)]
df.sort_values("topic_id", inplace=True)

# Save to CSV
df.to_csv("gap_comparison.csv", index=False)
print("✅ Saved: gap_comparison.csv")
