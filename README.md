# Research-Gap-Identification-using-LLMs
## Automated Research Gap Identification System

This repository implements a fully automated pipeline to extract, model, and evaluate “research gap” statements from a corpus of scientific papers using both programmatic and LLM-driven methods. It then benchmarks seven local LLMs via bulk scoring and head-to-head tournaments to rank their output quality.

## 🚀 Quickstart
### Clone & install
```bash
_git clone https://github.com/yourusername/research-gap-identifier.git

cd research-gap-identifier

pip install -r requirements.txt
```

### Configure
```bash
cp config.py.example config.py
```
Edit config.py to set your OpenAI API key (for GPT-4 stages), adjust search keywords, and confirm directory paths.

### Run the full pipeline
```bash
python main.py
```
### Launch the Streamlit UI:
```bash
streamlit run app.py
```
## 📚 Pipeline Overview

### 1. Data Collection
**Script:** step1_data_collection.py

**Output:** processed/collected_metadata.csv

**What it does:** Fetches paper metadata from Semantic Scholar based on your keywords.

### 2. Text Processing
**Script:** step2_text_processing.py

**Output:** processed/processed_papers.csv

**What it does:** Cleans abstracts and full text, extracts key sentences, and prepares data for topic modeling.

### 3. Topic Modeling
**Script:** step3_4_topic_modeling.py

**Outputs:**
results/papers_with_topics.csv

results/topics_info.csv

**What it does:** Uses BERTopic to assign each paper to a topic and summarize topic keywords.
### 4. Citation Analysis
**Script:** step5_citation_analysis.py

**Output:** results/citation_network_analysis.csv

**What it does:** Builds and analyzes a citation graph of the collected papers.

### 5. Gap Identification
**Script:** step6_7_gap_identification.py

**Output:** results/potential_gaps.csv

**What it does:** Extracts “limitations & future work” sentences and combines them into candidate gap statements.

### 6. Bulk GPT-4 Evaluation
**Scripts:***

evaluate_gaps_with_gpt4.py → raw markdown tables

parse_evaluation_csv.py → gap_evaluation_scores.csv

plot_gap_evaluation_scores.py → figures in visualizations/

**Outputs:**

gap_evaluation_scores.csv

Bar charts, heatmaps, and correlation plots under visualizations/

**What it does:** Scores each model’s gap on clarity, relevance, originality, and visualizes results.
### 7. Pairwise & Tournament Analysis
**Scripts:**

generate_gaps_from_local_models.py → local_gap_outputs/*.txt

pairwise_evaluate_gaps.py → pairwise_run1.csv, pairwise_run2.csv

compute_model_rankings.py → model_tournament_rankings_run1.csv, model_tournament_rankings_run2.csv

tournament_analysis_and_plots.py → tournament plots in visualizations/

**What it does:** Conducts head-to-head GPT-4 matchups in two independent runs, then aggregates wins via Borda Count, Elo Ratings, and PageRank, producing rankings and visual comparisons.
