import pandas as pd
import re

# Load the GPT-4 evaluations
df = pd.read_csv("gap_evaluations_gpt4.csv")

rows = []

for _, row in df.iterrows():
    topic_id = row['topic_id']
    eval_text = row['evaluation']

    # Split lines, find markdown table rows
    lines = eval_text.split("\n")
    table_lines = [l for l in lines if l.strip().startswith("|") and not re.search(r'---+', l)]

    # Skip malformed
    if len(table_lines) < 3:
        continue

    # Extract model rows (skip header and separator)
    for model_row in table_lines[1:]:
        parts = [p.strip() for p in model_row.strip("|").split("|")]
        if len(parts) < 5:
            continue
        model_name, clarity, relevance, originality, comment = parts
        try:
            rows.append({
                "topic_id": topic_id,
                "model": model_name,
                "clarity": int(clarity),
                "relevance": int(relevance),
                "originality": int(originality),
                "comment": comment
            })
        except ValueError:
            continue

# Save as CSV
scores_df = pd.DataFrame(rows)
scores_df.to_csv("gap_evaluation_scores.csv", index=False)
print("✅ Saved: gap_evaluation_scores.csv")
