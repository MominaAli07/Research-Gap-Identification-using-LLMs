# pairwise_evaluate_and_check_consistency.py

import pandas as pd
import itertools
import time
import config
import openai

# Initialize OpenAI client
client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

# Filepaths
INPUT_CSV   = "gap_comparison.csv"
OUTPUT1_CSV = "pairwise_run1.csv"
OUTPUT2_CSV = "pairwise_run2.csv"

# Prompt template
PROMPT_TEMPLATE = (
    "You are a senior academic reviewer evaluating two research gap statements on the same topic.\n"
    "Compare them based on:\n"
    "1. Clarity: How well is the gap articulated? Is the language precise and structured?\n"
    "2. Relevance: Does the gap accurately reflect missing directions in context?\n"
    "3. Originality: Does the statement propose a novel idea or perspective?\n\n"
    "For each dimension, choose which statement is better (Model A or Model B) and explain specifically why.\n\n"
    "Model A ({model_a}):\n"
    "{text_a}\n\n"
    "Model B ({model_b}):\n"
    "{text_b}\n\n"
    "Respond in this markdown table format exactly:\n"
    "| Dimension   | Better Model | Justification |\n"
    "|-------------|--------------|---------------|\n"
    "| Clarity     | A or B       | ...           |\n"
    "| Relevance   | A or B       | ...           |\n"
    "| Originality | A or B       | ...           |\n\n"
    "Then conclude with one sentence: 'Overall, Model X provides a more compelling research gap statement for this topic.'"
)

def parse_response(resp_text: str):
    """
    Given a GPT-4 response as markdown, extract the winners for clarity, relevance, originality.
    Returns (clarity_winner, relevance_winner, originality_winner).
    """
    clarity = relevance = originality = None
    for line in resp_text.splitlines():
        line = line.strip()
        if line.startswith("| Clarity"):
            parts = [c.strip() for c in line.strip("|").split("|")]
            clarity = parts[1]
        elif line.startswith("| Relevance"):
            parts = [c.strip() for c in line.strip("|").split("|")]
            relevance = parts[1]
        elif line.startswith("| Originality"):
            parts = [c.strip() for c in line.strip("|").split("|")]
            originality = parts[1]
    return clarity, relevance, originality

def evaluate_pairwise(df, output_csv):
    """
    Runs one pass of pairwise evaluation and saves results to output_csv.
    Returns a DataFrame of the results.
    """
    records = []
    model_cols = [c for c in df.columns if c != "topic_id"]

    for _, row in df.iterrows():
        topic_id = row["topic_id"]
        for a, b in itertools.combinations(model_cols, 2):
            text_a = row[a].strip()
            text_b = row[b].strip()
            if not text_a or not text_b:
                continue

            # Build prompt
            prompt = PROMPT_TEMPLATE.format(
                model_a=a,  text_a=text_a,
                model_b=b,  text_b=text_b
            )

            # Call GPT-4
            try:
                resp = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a senior academic reviewer."},
                        {"role": "user",   "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                reply = resp.choices[0].message.content
                c_win, r_win, o_win = parse_response(reply)
            except Exception as e:
                # On error, record None
                c_win = r_win = o_win = None
                reply = f"ERROR: {e}"

            records.append({
                "topic_id": topic_id,
                "model_a": a,
                "model_b": b,
                "clarity_winner": c_win,
                "relevance_winner": r_win,
                "originality_winner": o_win,
                "raw_response": reply
            })
            time.sleep(1)  # throttle

    result_df = pd.DataFrame(records)
    result_df.to_csv(output_csv, index=False)
    print(f"✅ Saved pairwise results to {output_csv}")
    return result_df

def compute_consistency(df1, df2):
    """
    Compute fraction of rows where clarity, relevance, originality winners match.
    Assumes df1 and df2 have the same shape and same ordering.
    """
    # Align by index
    assert len(df1) == len(df2), "Run outputs differ in length!"
    matches = (
        (df1["clarity_winner"]     == df2["clarity_winner"]) &
        (df1["relevance_winner"]   == df2["relevance_winner"]) &
        (df1["originality_winner"] == df2["originality_winner"])
    )
    consistency = matches.mean()
    print(f"🏅 Consistency across two runs: {consistency:.1%}")
    return consistency

def main():
    # Load input
    df = pd.read_csv(INPUT_CSV)

    # Run first evaluation
    df1 = evaluate_pairwise(df, OUTPUT1_CSV)

    # Run second evaluation
    df2 = evaluate_pairwise(df, OUTPUT2_CSV)

    # Compute and print consistency
    compute_consistency(df1, df2)

if __name__ == "__main__":
    main()
