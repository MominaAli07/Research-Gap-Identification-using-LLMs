# import streamlit as st
# import pandas as pd
# import os
# import subprocess
# import config
# import main

# st.set_page_config(page_title="Research Gap Dashboard", layout="wide")

# st.title("📊 Automated Research Gap Identification System")

# # Sidebar for navigation
# page = st.sidebar.radio("Navigate", ["📂 Data Pipeline", "🤖 Model Outputs", "📈 Evaluations", "📊 Visualizations"])

# # ========== Page 1: Data Pipeline ==========
# if page == "📂 Data Pipeline":
#     st.header("Step 1–7: Full Paper Processing Pipeline")

#     keyword = st.text_input("🔍 Enter Search Keyword:", value="Machine learning healthcare")

#     if st.button("▶️ Run Full Pipeline"):
#         with st.spinner(f"Running pipeline for: {keyword}"):

#             # ✅ Clear old data to force Step 1 & 2 to re-run
#             try:
#                 os.remove("processed/collected_metadata.csv")
#                 os.remove("processed/processed_papers.csv")
#             except FileNotFoundError:
#                 pass

#             # ✅ Update config with new keyword
#             config.set_search_keyword(keyword)

#             # ✅ Run full pipeline
#             main.run_pipeline()

#         st.success("✅ Full pipeline completed!")

# # ========== Page 2: Model Outputs ==========
# elif page == "🤖 Model Outputs":
#     st.header("Step: Generate Gaps with Local Models")
#     models = ["llama3.2:latest", "openchat:latest", "zephyr:latest", "gemma:2b", "mistral:latest", "neural-chat:latest", "phi:latest"]

#     if st.button("🚀 Generate Gaps from Local Models"):
#         subprocess.run(["python3", "generate_gaps_from_local_models.py"])
#         st.success("Gaps generated for all models!")

#     if os.path.exists("local_gap_outputs"):
#         st.subheader("📄 Model Outputs (topic 0)")
#         for m in models:
#             file = f"local_gap_outputs/{m.replace(':', '_')}_topic_0.txt"
#             if os.path.exists(file):
#                 with open(file) as f:
#                     st.expander(f"🧠 {m}").write(f.read())

# # ========== Page 3: Evaluations ==========
# elif page == "📈 Evaluations":
#     st.header("Evaluate Local Model Gaps Using GPT-4")

#     if st.button("💬 Run GPT-4 Evaluations"):
#         subprocess.run(["python3", "evaluate_gaps_with_gpt4.py"])
#         subprocess.run(["python3", "parse_evaluation_csv.py"])
#         st.success("Evaluation completed!")

#     if os.path.exists("gap_evaluation_scores.csv"):
#         df_eval = pd.read_csv("gap_evaluation_scores.csv")
#         st.dataframe(df_eval)

# # ========== Page 4: Visualizations ==========
# elif page == "📊 Visualizations":
#     st.header("Visualization Dashboard")

#     if st.button("📊 Generate All Visualizations"):
#         subprocess.run(["python3", "advanced_gap_visualizations.py"])
#         st.success("Plots saved in /visualizations folder")

#     col1, col2 = st.columns(2)

#     for metric in ["clarity", "relevance", "originality"]:
#         plot_path = f"visualizations/heatmap_{metric}.png"
#         if os.path.exists(plot_path):
#             col1.image(plot_path, caption=f"{metric.title()} Heatmap", use_column_width=True)

#     if os.path.exists("visualizations/model_ranking_bar.png"):
#         col2.image("visualizations/model_ranking_bar.png", caption="Model Ranking", use_column_width=True)

#     if os.path.exists("visualizations/model_score_correlation_heatmap.png"):
#         st.image("visualizations/model_score_correlation_heatmap.png", caption="Model Correlation Heatmap", use_column_width=True)




import streamlit as st
import pandas as pd
import os
import subprocess
import config
import main

st.set_page_config(page_title="Research Gap Dashboard", layout="wide")

st.title("\U0001F4CA Automated Research Gap Identification System")

# Sidebar for navigation
page = st.sidebar.radio("Navigate", ["\U0001F4C2 Data Pipeline", "\U0001F916 Model Outputs", "\U0001F4C8 Evaluations", "📊 Visualizations"])

# ========== Page 1: Data Pipeline ==========
if page == "\U0001F4C2 Data Pipeline":
    st.header("Step 1–7: Full Paper Processing Pipeline")

    keyword = st.text_input("🔍 Enter Search Keyword:", value="Machine learning healthcare")

    if st.button("▶️ Run Full Pipeline"):
        with st.spinner(f"Running pipeline for: {keyword}"):

            # ✅ Clear old data to force Step 1 & 2 to re-run
            try:
                os.remove("processed/collected_metadata.csv")
                os.remove("processed/processed_papers.csv")
            except FileNotFoundError:
                pass

            # ✅ Update config with new keyword
            config.set_search_keyword(keyword)

            # ✅ Run full pipeline
            main.run_pipeline()

        st.success("✅ Full pipeline completed!")

    if os.path.exists("processed/papers_with_topics.csv"):
        st.subheader(f"📄 Processed Topics for keyword: *{keyword}*")
        df = pd.read_csv("processed/papers_with_topics.csv")
        st.dataframe(df.head())

    if os.path.exists("results/topics_info.csv"):
        st.subheader("📈 Topic Info Summary")
        df2 = pd.read_csv("results/topics_info.csv")
        st.dataframe(df2.head())

# ========== Page 2: Model Outputs ==========
elif page == "🤖 Model Outputs":
    st.header("Step: Generate Gaps with Local Models")
    models = ["llama3.2:latest", "openchat:latest", "zephyr:latest", "gemma:2b", "mistral:latest", "neural-chat:latest", "phi:latest"]

    if st.button("🚀 Generate Gaps from Local Models"):
        subprocess.run(["python3", "generate_gaps_from_local_models.py"])
        st.success("Gaps generated for all models!")

    if os.path.exists("local_gap_outputs"):
        st.subheader("📄 Model Outputs (topic 0)")
        for m in models:
            file = f"local_gap_outputs/{m.replace(':', '_')}_topic_0.txt"
            if os.path.exists(file):
                with open(file) as f:
                    st.expander(f"🧠 {m}").write(f.read())

# ========== Page 3: Evaluations ==========
elif page == "📈 Evaluations":
    st.header("Evaluate Local Model Gaps Using GPT-4")

    if st.button("💬 Run GPT-4 Evaluations"):
        subprocess.run(["python3", "evaluate_gaps_with_gpt4.py"])
        subprocess.run(["python3", "parse_evaluation_csv.py"])
        st.success("Evaluation completed!")

    if os.path.exists("gap_evaluation_scores.csv"):
        df_eval = pd.read_csv("gap_evaluation_scores.csv")
        st.dataframe(df_eval)

# ========== Page 4: Visualizations ==========
elif page == "📊 Visualizations":
    st.header("Visualization Dashboard")

    if st.button("📊 Generate All Visualizations"):
        subprocess.run(["python3", "advanced_gap_visualizations.py"])
        st.success("Plots saved in /visualizations folder")

    col1, col2 = st.columns(2)

    for metric in ["clarity", "relevance", "originality"]:
        plot_path = f"visualizations/heatmap_{metric}.png"
        if os.path.exists(plot_path):
            col1.image(plot_path, caption=f"{metric.title()} Heatmap", use_column_width=True)

    if os.path.exists("visualizations/model_ranking_bar.png"):
        col2.image("visualizations/model_ranking_bar.png", caption="Model Ranking", use_column_width=True)

    if os.path.exists("visualizations/model_score_correlation_heatmap.png"):
        st.image("visualizations/model_score_correlation_heatmap.png", caption="Model Correlation Heatmap", use_column_width=True)
