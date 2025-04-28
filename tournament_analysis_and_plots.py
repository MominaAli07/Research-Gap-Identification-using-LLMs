import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# tournament_analysis_and_plots.py
# Generates visualizations for each individual run and side-by-side comparison
os.makedirs("visualizations", exist_ok=True)

# Helper to load and normalize rankings
def load_and_norm(path):
    df = pd.read_csv(path, index_col=0)
    df.columns = ['Borda', 'Elo', 'PageRank']
    norm = (df - df.min()) / (df.max() - df.min())
    norm = norm.fillna(0)
    return df, norm

# Define file paths and labels for each run
runs = [
    ('model_tournament_rankings_run1.csv', 'Run 1'),
    ('model_tournament_rankings_run2.csv', 'Run 2'),
]

# Load data for each run
raw_data = []  # [(df, label)]
norm_data = []  # [(norm_df, label)]
for file, label in runs:
    df_run, norm_run = load_and_norm(file)
    raw_data.append((df_run, label))
    norm_data.append((norm_run, label))

# --- 1) Raw score bar charts side-by-side ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (df_run, label) in zip(axes, raw_data):
    df_run.plot(kind='bar', ax=ax, rot=45)
    ax.set_title(f'Raw Rankings ({label})')
    ax.set_ylabel('Score / Rating')
plt.tight_layout()
plt.savefig('visualizations/tournament_raw_bar_comparison.png')
plt.close()

# --- 2) Normalized bar charts side-by-side ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (norm_run, label) in zip(axes, norm_data):
    norm_run.plot(kind='bar', ax=ax, rot=45)
    ax.set_title(f'Normalized Rankings ({label})')
    ax.set_ylabel('Normalized Score')
    ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('visualizations/tournament_normalized_bar_comparison.png')
plt.close()

# --- 3) Heatmaps of normalized scores side-by-side ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (norm_run, label) in zip(axes, norm_data):
    sns.heatmap(norm_run, annot=True, cmap='YlOrBr', cbar=False, ax=ax)
    ax.set_title(f'Normalized Heatmap ({label})')
plt.tight_layout()
plt.savefig('visualizations/tournament_heatmap_comparison.png')
plt.close()

# --- 4) Radar plots of normalized scores side-by-side ---
metrics = norm_data[0][0].columns.tolist()
n_metrics = len(metrics)
angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)
angles_closed = np.concatenate((angles, [angles[0]]))

fig = plt.figure(figsize=(12, 6))
for idx, (norm_run, label) in enumerate(norm_data):
    ax = fig.add_subplot(1, 2, idx+1, projection='polar')
    for model in norm_run.index:
        values = norm_run.loc[model].tolist()
        values_closed = values + [values[0]]
        ax.plot(angles_closed, values_closed, label=model)
    ax.set_xticks(angles)
    ax.set_xticklabels(metrics)
    ax.set_title(f'Radar Plot ({label})')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig('visualizations/tournament_radar_comparison.png')
plt.close()

print("✅ Comparison visualizations for Run 1 and Run 2 saved to /visualizations.")
