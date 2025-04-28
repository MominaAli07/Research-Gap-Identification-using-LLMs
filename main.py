# main.py
import step1_data_collection
import step2_text_processing
import step3_4_topic_modeling
import step5_citation_analysis
import step6_7_gap_identification
import config
import os
import pandas as pd

def run_pipeline(force_refresh=False):
    print("\n=== Starting Full Pipeline ===\n")

    # Set the path to save files
    metadata_path = os.path.join(config.PROCESSED_DIR, "collected_metadata.csv")
    processed_path = os.path.join(config.PROCESSED_DIR, "processed_papers.csv")
    
    if not force_refresh and os.path.exists(metadata_path) and os.path.exists(processed_path):
        print("Detected both collected_metadata.csv and processed_papers.csv.")
        print("Skipping Step 1 (Data Collection) and Step 2 (Text Processing)...")
        # Load cached results
        metadata_df = pd.read_csv(metadata_path)
        processed_df = pd.read_csv(processed_path)
    else:
        if not os.path.exists(metadata_path):
            print("collected_metadata.csv not found, running Step 1: Data Collection...")
            metadata_df = step1_data_collection.fetch_and_prepare_papers(
                config.SEARCH_KEYWORDS, 
                config.MAX_PAPERS_PER_SOURCE
            )
            if metadata_df.empty:
                print("No papers were collected. Please check your search keywords or try a different query.")
                return
        else:
            print("collected_metadata.csv found, skipping Step 1: Data Collection...")
            metadata_df = pd.read_csv(metadata_path)

        if not os.path.exists(processed_path):
            print("processed_papers.csv not found, running Step 2: Text Processing...")
            processed_df = step2_text_processing.process_papers(metadata_df)
            if processed_df.empty or 'abstract_cleaned' not in processed_df.columns:
                print("Processed data is empty or missing required columns. Exiting pipeline.")
                return
        else:
            print("processed_papers.csv found, skipping Step 2: Text Processing...")
            processed_df = pd.read_csv(processed_path)

    # # Step 1: Data Collection using Semantic Scholar
    # metadata_df = step1_data_collection.fetch_and_prepare_papers(config.SEARCH_KEYWORDS, config.MAX_PAPERS_PER_SOURCE)
    # if metadata_df.empty:
    #     print("No papers were collected. Please check your search keywords or try a different query.")
    #     return

    # # Step 2: Text Processing (includes PDF extraction using PyMuPDF and gap sentence extraction)
    # processed_df = step2_text_processing.process_papers(metadata_df)
    # if processed_df.empty or 'abstract_cleaned' not in processed_df.columns:
    #     print("Processed data is empty or missing required columns. Exiting pipeline.")
    #     return

    # Step 3 & 4: Topic Modeling and Assignment using BERTopic
    result = step3_4_topic_modeling.perform_topic_modeling(processed_df)
    if result[0] is None:
        print("Topic modeling failed. Exiting pipeline.")
        return
    papers_with_topics_df, topic_model, topic_info = result
    
    # Step 5: Citation Analysis using Semantic Scholar references
    G = step5_citation_analysis.build_citation_network(papers_with_topics_df)
    network_analysis = step5_citation_analysis.analyze_citation_network(G, papers_with_topics_df)
    
    # Step 6 & 7: Research Gap Identification with combined criteria and GPT-4 validation.
    potential_gaps_df = step6_7_gap_identification.identify_potential_gaps(
        papers_with_topics_df,
        topic_info,
        network_analysis,
        {}  # Trend analysis placeholder; can be extended later.
    )
    
    print("\n=== Pipeline Finished ===\n")
    
if __name__ == "__main__":
    run_pipeline()
