# # step3_4_topic_modeling.py
# import pandas as pd
# import os
# import config
# from sentence_transformers import SentenceTransformer
# from bertopic import BERTopic
# from sklearn.feature_extraction.text import CountVectorizer

# def perform_topic_modeling(processed_df):
#     if 'abstract_cleaned' not in processed_df.columns:
#         print("Error: 'abstract_cleaned' column not found in processed data.")
#         return None, None, None

#     docs = processed_df['abstract_cleaned'].fillna('').tolist()
#     valid_indices = [i for i, doc in enumerate(docs) if len(doc.split()) > 10]
#     valid_docs = [docs[i] for i in valid_indices]
#     original_indices_map = {new_idx: old_idx for new_idx, old_idx in enumerate(valid_indices)}

#     if not valid_docs:
#         print("Error: No valid documents found for topic modeling after filtering.")
#         return None, None, None

#     print(f"Performing topic modeling on {len(valid_docs)} documents.")

#     try:
#         embedding_model = SentenceTransformer('allenai/scibert_scivocab_uncased')
#         print("Calculating embeddings...")
#         embeddings = embedding_model.encode(valid_docs, show_progress_bar=True)
#     except Exception as e:
#         print(f"Error loading/using embedding model: {e}")
#         return None, None, None

#     vectorizer_model = CountVectorizer(stop_words="english")
#     topic_model = BERTopic(
#         embedding_model=embedding_model,
#         vectorizer_model=vectorizer_model,
#         language='english',
#         calculate_probabilities=False,
#         min_topic_size=config.BERTOPIC_MIN_TOPIC_SIZE,
#         nr_topics='auto',
#         verbose=True
#     )

#     print("Fitting BERTopic model...")
#     try:
#         topics, _ = topic_model.fit_transform(valid_docs, embeddings=embeddings)
#     except Exception as e:
#         error_message = str(e)
#         print(f"Error fitting BERTopic model: {error_message}")
#         if "Min cluster size must be greater than one" in error_message or "list index out of range" in error_message:
#             print("Falling back: Assigning all documents to a single topic (default cluster 0).")
#             topics = [0] * len(valid_docs)
#             dummy_topic_info = pd.DataFrame({
#                 "Topic": [0],
#                 "Name": ["All Documents"],
#                 "Count": [len(valid_docs)],
#                 "Representation": ["N/A"]
#             })
#             topic_model = None
#         else:
#             return None, None, None

#     processed_df['topic_id'] = -1
#     for new_idx, topic_id in enumerate(topics):
#         original_idx = original_indices_map.get(new_idx)
#         if original_idx is not None:
#             processed_df.loc[original_idx, 'topic_id'] = topic_id

#     if topic_model is not None:
#         model_path = os.path.join(config.RESULTS_DIR, "bertopic_model")
#         topic_model.save(model_path, serialization="safetensors", save_embedding_model=False)
#         print(f"BERTopic model saved to {model_path}")
#         topic_info = topic_model.get_topic_info()
#     else:
#         topic_info = dummy_topic_info

#     topic_info_path = os.path.join(config.RESULTS_DIR, "topics_info.csv")
#     topic_info.to_csv(topic_info_path, index=False)
#     print(f"Topic information saved to {topic_info_path}")

#     output_path = os.path.join(config.PROCESSED_DIR, "papers_with_topics.csv")
#     processed_df.to_csv(output_path, index=False)
#     print(f"Data with topic assignments saved to {output_path}")

#     if topic_model is not None:
#         try:
#             fig_topics = topic_model.visualize_topics()
#             fig_topics_path = os.path.join(config.RESULTS_DIR, "topics_visualization.html")
#             fig_topics.write_html(fig_topics_path)
#             print(f"Topic visualization saved to {fig_topics_path}")
#         except Exception as e:
#             print(f"Visualization error: {e}")

#     return processed_df, topic_model, topic_info

# if __name__ == "__main__":
#     print("\nStep 3 & 4: Starting Topic Modeling...")
#     processed_path = os.path.join(config.PROCESSED_DIR, "processed_papers.csv")
#     if os.path.exists(processed_path):
#         processed_df = pd.read_csv(processed_path)
#         processed_df['abstract_cleaned'] = processed_df['abstract_cleaned'].astype(str)
#         result = perform_topic_modeling(processed_df)
#         if result[0] is not None:
#             papers_with_topics_df, model, info = result
#             print("\nStep 3 & 4: Topic Modeling Finished.")
#         else:
#             print("\nStep 3 & 4: Topic Modeling Failed.")
#     else:
#         print(f"Error: Processed file not found at {processed_path}. Run Step 2 first.")

# step3_4_topic_modeling.py
import os
import pandas as pd
import config
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

# -- Debug/Workaround: Disable parallelism warnings from huggingface/tokenizers --
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def perform_topic_modeling(processed_df):
    # Combine multiple text fields if available
    text_fields = ['abstract_cleaned', 'conclusion_cleaned', 'future_work_cleaned', 'limitations_cleaned', 'fallback_text','gap_sentences']
    available_fields = [field for field in text_fields if field in processed_df.columns]
    if not available_fields:
        print("Error: None of the expected text fields were found in processed data.")
        return None, None, None

    # Create a combined text field from available columns
    processed_df['text_combined'] = processed_df[available_fields].fillna("").agg(" ".join, axis=1)
    docs = processed_df['text_combined'].tolist()

    # Filter out documents that are too short
    valid_indices = [i for i, doc in enumerate(docs) if len(doc.split()) > 10]
    valid_docs = [docs[i] for i in valid_indices]
    original_indices_map = {new_idx: old_idx for new_idx, old_idx in enumerate(valid_indices)}

    if not valid_docs:
        print("Error: No valid documents found for topic modeling after filtering.")
        return None, None, None

    print(f"Performing topic modeling on {len(valid_docs)} documents.")

    try:
        # Load the embedding model; if not found, it will create one with mean pooling
        embedding_model = SentenceTransformer('allenai/scibert_scivocab_uncased')
        print("Calculating embeddings...")
        embeddings = embedding_model.encode(valid_docs, show_progress_bar=True)
        print(f"Embeddings calculated: {len(embeddings)} vectors.")
    except Exception as e:
        print(f"Error loading/using embedding model: {e}")
        return None, None, None

    # Set up vectorizer and create BERTopic model
    vectorizer_model = CountVectorizer(stop_words="english")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        language='english',
        calculate_probabilities=False,
        min_topic_size=config.BERTOPIC_MIN_TOPIC_SIZE,
        nr_topics='auto',
        verbose=True
    )

    print("Fitting BERTopic model...")
    try:
        topics, _ = topic_model.fit_transform(valid_docs, embeddings=embeddings)
    except Exception as e:
        error_message = str(e)
        print(f"Error fitting BERTopic model: {error_message}")
        if "Min cluster size must be greater than one" in error_message or "list index out of range" in error_message:
            print("Falling back: Assigning all documents to a single topic (default cluster 0).")
            topics = [0] * len(valid_docs)
            dummy_topic_info = pd.DataFrame({
                "Topic": [0],
                "Name": ["All Documents"],
                "Count": [len(valid_docs)],
                "Representation": ["N/A"]
            })
            topic_model = None
        else:
            return None, None, None

    # Map topic labels back to the original DataFrame
    processed_df['topic_id'] = -1
    for new_idx, topic_id in enumerate(topics):
        original_idx = original_indices_map.get(new_idx)
        if original_idx is not None:
            processed_df.loc[original_idx, 'topic_id'] = topic_id

    # Save the model and topic information if possible
    if topic_model is not None:
        model_path = os.path.join(config.RESULTS_DIR, "bertopic_model")
        # Warning: The embedding model is not explicitly saved. See documentation.
        topic_model.save(model_path, serialization="safetensors", save_embedding_model=False)
        print(f"BERTopic model saved to {model_path}")
        topic_info = topic_model.get_topic_info()
    else:
        topic_info = dummy_topic_info

    topic_info_path = os.path.join(config.RESULTS_DIR, "topics_info.csv")
    topic_info.to_csv(topic_info_path, index=False)
    print(f"Topic information saved to {topic_info_path}")

    output_path = os.path.join(config.PROCESSED_DIR, "papers_with_topics.csv")
    processed_df.to_csv(output_path, index=False)
    print(f"Data with topic assignments saved to {output_path}")

    # ---- Optional Visualization of Topics ----
    if topic_model is not None:
        try:
            # Debug print: Check the topic info content
            topic_info_count = topic_info.shape[0]
            print(f"Number of topics for visualization: {topic_info_count}")
            # If there is only one topic or no topics, visualization may not work.
            if topic_info_count < 2:
                print("Not enough topics to generate a meaningful visualization, skipping visualization.")
            else:
                fig_topics = topic_model.visualize_topics()
                fig_topics_path = os.path.join(config.RESULTS_DIR, "topics_visualization.html")
                fig_topics.write_html(fig_topics_path)
                print(f"Topic visualization saved to {fig_topics_path}")
        except Exception as e:
            print(f"Visualization error: {e}")

    return processed_df, topic_model, topic_info

if __name__ == "__main__":
    print("\nStep 3 & 4: Starting Topic Modeling...")
    processed_path = os.path.join(config.PROCESSED_DIR, "processed_papers.csv")
    if os.path.exists(processed_path):
        processed_df = pd.read_csv(processed_path)
        # Ensure text fields are strings
        for col in ['abstract_cleaned', 'conclusion_cleaned', 'future_work_cleaned', 'limitations_cleaned', 'fallback_text','gap_sentences']:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].astype(str)
        result = perform_topic_modeling(processed_df)
        if result[0] is not None:
            papers_with_topics_df, model, info = result
            print("\nStep 3 & 4: Topic Modeling Finished.")
        else:
            print("\nStep 3 & 4: Topic Modeling Failed.")
    else:
        print(f"Error: Processed file not found at {processed_path}. Run Step 2 first.")
