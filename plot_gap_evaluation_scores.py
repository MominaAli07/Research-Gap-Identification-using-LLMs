# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the scores
# df = pd.read_csv("gap_evaluation_scores.csv")

# # Compute average scores per model
# avg_scores = df.groupby("model")[["clarity", "relevance", "originality"]].mean().round(2)
# print("📊 Average Scores:\n", avg_scores)

# # Plot settings
# plt.figure(figsize=(10, 6))
# avg_scores.plot(kind="bar", figsize=(10, 6), rot=0)
# plt.title("Average Research Gap Evaluation Scores per Model")
# plt.ylabel("Average Score (1–5)")
# plt.ylim(0, 5)
# plt.xticks(rotation=0)
# plt.grid(axis="y", linestyle="--", alpha=0.7)
# plt.legend(title="Evaluation Aspect")
# plt.tight_layout()

# # Save and show
# plt.savefig("model_gap_scores.png")
# plt.show()





import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np
import os

# Load score data
df = pd.read_csv("gap_evaluation_scores.csv")

# Create output directory for plots
os.makedirs("visualizations", exist_ok=True)

# Pivot to Topic × Model × Metric format
def pivot_scores(metric):
    pivot = df.pivot(index="topic_id", columns="model", values=metric)
    return pivot

avg_scores = df.groupby("model")[["clarity", "relevance", "originality"]].mean().round(2)
print("📊 Average Scores:\n", avg_scores)

# Plot grouped bar chart
plt.figure(figsize=(10, 6))
avg_scores.plot(kind="bar", rot=0)
plt.title("Average Research Gap Evaluation Scores per Model")
plt.ylabel("Average Score (1–5)")
plt.ylim(0, 5)
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(title="Metric")
plt.tight_layout()

# Save to file
plt.savefig("visualizations/average_score_barplot.png")
plt.close()




# # ==========================
# # 1. HEATMAPS
# # ==========================
# for metric in ['clarity', 'relevance', 'originality']:
#     pivot = pivot_scores(metric)
#     plt.figure(figsize=(10, 6))
#     sns.heatmap(pivot, annot=True, cmap="YlGnBu", cbar_kws={'label': metric.title()})
#     plt.title(f"Heatmap of {metric.title()} Scores by Model and Topic")
#     plt.xlabel("Model")
#     plt.ylabel("Topic ID")
#     plt.tight_layout()
#     plt.savefig(f"visualizations/heatmap_{metric}.png")
#     plt.close()

# print("✅ Heatmaps saved.")



# ==========================
# 2. MODEL RANKING TABLE
# ==========================
# Add total score and optional weighted score
df["total_score"] = df["clarity"] + df["relevance"] + df["originality"]

# Optional: apply weights (e.g., 40% clarity, 30% relevance, 30% originality)
weights = {"clarity": 0.4, "relevance": 0.3, "originality": 0.3}
df["weighted_score"] = (
    df["clarity"] * weights["clarity"] +
    df["relevance"] * weights["relevance"] +
    df["originality"] * weights["originality"]
)

# Compute average scores by model
ranking = df.groupby("model")[["clarity", "relevance", "originality", "total_score", "weighted_score"]].mean().round(2)
ranking = ranking.sort_values("weighted_score", ascending=False)

# Save as table
ranking.to_csv("visualizations/model_score_ranking.csv")
print("✅ Model ranking table saved.")

# Plot horizontal bar chart
ranking["weighted_score"].plot(kind="barh", color="teal", figsize=(8, 5))
plt.xlabel("Average Weighted Score")
plt.title("Model Ranking by Weighted Score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("visualizations/model_ranking_bar.png")
plt.close()

# ==========================
# 3. PAIRWISE CORRELATION
# ==========================
# Pivot into wide format: each row = topic_id + model, columns = metrics
pivot_df = df.pivot_table(index=["topic_id"], columns="model", values=["clarity", "relevance", "originality"])

# Flatten multi-index columns
pivot_df.columns = [f"{metric}_{model}" for metric, model in pivot_df.columns]

# Compute correlation matrix
correlation_matrix = pivot_df.corr().round(2)

# Save correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Correlation'})
plt.title("Pairwise Correlation of Model Scores")
plt.tight_layout()
plt.savefig("visualizations/model_score_correlation_heatmap.png")
plt.close()

print("✅ Pairwise model correlation heatmap saved.")

# Optional: scatter matrix (commented out if too dense)
# scatter_matrix(pivot_df, figsize=(15, 15), diagonal="hist")
# plt.suptitle("Scatter Matrix of All Model Scores")
# plt.savefig("visualizations/scatter_matrix.png")
# plt.close()
