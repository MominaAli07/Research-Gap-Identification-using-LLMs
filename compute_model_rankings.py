import pandas as pd
import networkx as nx

# compute_model_rankings.py
# Computes tournament rankings separately for two runs:
# - Borda Count
# - Elo Ratings
# - PageRank Centrality

# File paths for pairwise results and desired outputs
RUN_CONFIG = [
    ("pairwise_run1.csv", "model_tournament_rankings_run1.csv"),
    ("pairwise_run2.csv", "model_tournament_rankings_run2.csv"),
]

# Helper: normalize winner labels
def normalize_winner(row, col):
    w = str(row[col]).strip()
    if w in ("A", "Model A"):
        return row["model_a"]
    if w in ("B", "Model B"):
        return row["model_b"]
    return w

# Helper: Elo expected score
def expected_score(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))

# Core function to compute rankings for a single run
def compute_for_run(input_csv: str, output_csv: str):
    # Load the pairwise comparisons
    df = pd.read_csv(input_csv)

    # Normalize winners for each metric
    for metric in ("clarity", "relevance", "originality"):
        df[f"{metric}_winner"] = df.apply(lambda r: normalize_winner(r, f"{metric}_winner"), axis=1)

    # Identify unique model names
    models = sorted(set(df['model_a']).union(df['model_b']))

    # Technique 1: Borda Count (tally wins)
    borda = pd.Series(0, index=models, dtype=int)
    for metric in ("clarity", "relevance", "originality"):
        counts = df[f"{metric}_winner"].value_counts()
        for model, cnt in counts.items():
            if model in borda.index:
                borda[model] += cnt
    borda_df = borda.rename("borda_wins").to_frame()

    # Technique 2: Elo Ratings
    K = 32
    elo = {m: 1500.0 for m in models}

    # Build match list (winner, loser)
    matches = []
    for _, row in df.iterrows():
        for metric in ("clarity", "relevance", "originality"):
            winner = row[f"{metric}_winner"]
            if winner not in models:
                continue
            loser = row['model_b'] if winner == row['model_a'] else row['model_a']
            if loser not in models:
                continue
            matches.append((winner, loser))

    # Apply Elo updates
    for winner, loser in matches:
        r_w, r_l = elo[winner], elo[loser]
        e_w = expected_score(r_w, r_l)
        e_l = expected_score(r_l, r_w)
        elo[winner] += K * (1 - e_w)
        elo[loser]  += K * (0 - e_l)
    elo_df = pd.Series(elo, name="elo_rating").to_frame()

    # Technique 3: PageRank on directed win graph
    G = nx.DiGraph()
    G.add_nodes_from(models)
    for winner, loser in matches:
        if G.has_edge(loser, winner):
            G[loser][winner]['weight'] += 1
        else:
            G.add_edge(loser, winner, weight=1)
    pr = nx.pagerank(G, weight='weight')
    pr_df = pd.Series(pr, name="pagerank").to_frame()

    # Combine all rankings
    summary = pd.concat([borda_df, elo_df, pr_df], axis=1)
    summary.to_csv(output_csv)
    print(f"✅ Rankings for '{input_csv}' saved to '{output_csv}'")


if __name__ == '__main__':
    for input_csv, output_csv in RUN_CONFIG:
        compute_for_run(input_csv, output_csv)
