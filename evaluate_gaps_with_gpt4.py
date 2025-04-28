import openai
import pandas as pd
import config
import time

client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

INPUT_CSV = "gap_comparison.csv"
OUTPUT_CSV = "gap_evaluations_gpt4.csv"

df = pd.read_csv(INPUT_CSV)
evaluations = []

# Get model columns dynamically (skip 'topic_id')
model_columns = [col for col in df.columns if col != "topic_id"]

# Loop through each topic row
for _, row in df.iterrows():
    topic_id = row["topic_id"]

    # Dynamically build gap statement list
    model_statements = []
    for i, model in enumerate(model_columns, 1):
        gap_text = row[model].strip() if pd.notna(row[model]) else "Not available."
        model_statements.append(f"{i}. **{model}**:\n{gap_text}\n")

    prompt = f"""
You are an expert academic reviewer.

The following research gap statements were generated for the same topic using different AI models.
Please rate each one on:
- Clarity (1–5)
- Relevance (1–5)
- Originality (1–5)

Then explain briefly why each one received the scores it did.

Here are the gap statements:

{''.join(model_statements)}

Respond with a markdown table of scores and a few comments for each model.
"""

    try:
        print(f"🧪 Evaluating topic {topic_id} with {len(model_columns)} models...")

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a senior academic peer reviewer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )

        eval_text = response.choices[0].message.content
        evaluations.append({
            "topic_id": topic_id,
            "evaluation": eval_text
        })

        time.sleep(1)  # Optional: slow down for rate limit
    except Exception as e:
        print(f"⚠️ Error evaluating topic {topic_id}: {e}")
        evaluations.append({
            "topic_id": topic_id,
            "evaluation": f"ERROR: {str(e)}"
        })

# Save results
eval_df = pd.DataFrame(evaluations)
eval_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ GPT-4 evaluations saved to: {OUTPUT_CSV}")
