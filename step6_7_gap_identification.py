# step6_7_gap_identification.py
import pandas as pd
import os
import re
from collections import Counter
import config
import openai
from transformers import pipeline as hf_pipeline

# Set OpenAI API key
openai.api_key = config.OPENAI_API_KEY

# Initialize a summarization pipeline for gap sentences
summarizer = hf_pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_gap_text(text):
    """Summarizes a block of text using a transformer summarizer.
       If the text is too short, returns it unchanged."""
    if not text or len(text.split()) < 30:
        return text
    try:
        summary = summarizer(text, max_length=60, min_length=20, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return text

def extract_pico_elements(text):
    pico = {'P': [], 'I': [], 'C': [], 'O': []}
    if not text:
        return pico
    sentences = re.split(r'[.!?]+', text)
    for sentence in sentences:
        if re.search(r'\b(patient|population|subject|group)\b', sentence, re.IGNORECASE):
            pico['P'].append(sentence.strip())
        if re.search(r'\b(intervention|treatment|therapy|method|approach)\b', sentence, re.IGNORECASE):
            pico['I'].append(sentence.strip())
        if re.search(r'\b(compar(e|ison)|control|placebo|standard)\b', sentence, re.IGNORECASE):
            pico['C'].append(sentence.strip())
        if re.search(r'\b(outcome|result|effect|measure|endpoint)\b', sentence, re.IGNORECASE):
            pico['O'].append(sentence.strip())
    for key in pico:
        pico[key] = pico[key][:2]
    return pico

def extract_limitations_future_work(text):
    limitations = []
    future_work = []
    if not text:
        return limitations, future_work
    lim_match = re.search(r'\n(limitations?|drawbacks)\n(.*?)\n', text, re.IGNORECASE | re.DOTALL)
    fut_match = re.search(r'\n(future work|future directions)\n(.*?)\n', text, re.IGNORECASE | re.DOTALL)
    if lim_match:
        limitations.append(lim_match.group(2).strip()[:500] + "...")
    if fut_match:
        future_work.append(fut_match.group(2).strip()[:500] + "...")
    if not limitations:
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            if re.search(r'\b(limit|drawback|weakness|however|restricted to)\b', sentence, re.IGNORECASE):
                limitations.append(sentence.strip())
                if len(limitations) >= 3:
                    break
    if not future_work:
         sentences = re.split(r'[.!?]+', text)
         for sentence in sentences:
            if re.search(r'\b(future work|next step|should investigate|further research)\b', sentence, re.IGNORECASE):
                future_work.append(sentence.strip())
                if len(future_work) >= 3:
                    break
    return limitations, future_work

def analyze_trends_and_patterns(papers_df, topic_info):
    print("Analyzing trends and patterns...")
    analysis = {'trends': {}, 'limitations': {}}
    topic_year_counts = papers_df[papers_df['topic_id'] != -1].groupby(['topic_id', 'year']).size().unstack(fill_value=0)
    current_year = pd.Timestamp.now().year
    recent_years = range(current_year - 3, current_year + 1)
    recent_activity = topic_year_counts.loc[:, topic_year_counts.columns.isin(recent_years)].sum(axis=1)
    total_topic_counts = topic_info.set_index('Topic')['Count']
    low_count_recent = recent_activity[(recent_activity > 0) & (total_topic_counts.loc[recent_activity.index] < 20)]
    analysis['trends']['topic_year_counts'] = topic_year_counts.to_dict()
    analysis['trends']['recent_activity_counts'] = recent_activity.to_dict()
    analysis['trends']['low_count_recent_topics'] = low_count_recent.to_dict()
    print(f"Identified {len(low_count_recent)} topics with low total count but recent activity.")
    all_limitations = []
    all_future_works = []
    if 'full_text' in papers_df.columns:
        papers_df['full_text'] = papers_df['full_text'].fillna('')
        for index, row in papers_df.iterrows():
            text_to_analyze = row['full_text']
            if text_to_analyze:
                lims, futs = extract_limitations_future_work(text_to_analyze)
                all_limitations.extend(lims)
                all_future_works.extend(futs)
        limitation_word_counts = Counter(re.findall(r'\b[a-z]{4,}\b', " ".join(all_limitations)))
        future_word_counts = Counter(re.findall(r'\b[a-z]{4,}\b', " ".join(all_future_works)))
        analysis['limitations']['common_limit_words'] = limitation_word_counts.most_common(15)
        analysis['limitations']['common_future_words'] = future_word_counts.most_common(15)
        print(f"Top common words in Limitations: {analysis['limitations']['common_limit_words']}")
        print(f"Top common words in Future Work: {analysis['limitations']['common_future_words']}")
    else:
        print("No full text found; skipping limitations analysis.")
        analysis['limitations']['common_limit_words'] = []
        analysis['limitations']['common_future_words'] = []
    return analysis

# def validate_gaps_with_gpt4(gap_candidates):
#     if not gap_candidates:
#          return None
#     try:
#          prompt = (
#              "Based on the following potential research gaps identified from scientific papers, "
#              "generate a concise summary that describes the gap in natural language. "
#              "Use the following details: \n"
#          )
#          for i, gap in enumerate(gap_candidates[:5]):
#              gap_text = gap.get('sample_gap_text', 'No detailed text provided.')
#              prompt += f"Gap {i+1}: {gap_text}\n"
#          prompt += "\nSummarize the gap as: 'There is a potential for research in this area because ...'"
         
#          response = openai.chat.completions.create(
#              model="gpt-4",
#              messages=[
#                  {"role": "system", "content": "You are an expert research analyst."},
#                  {"role": "user", "content": prompt}
#              ],
#              max_tokens=500,
#              temperature=0.5,
#          )
#          summary = response.choices[0].message.content
#          print("GPT-4 Summary/Validation:\n", summary)
#          return summary
#     except Exception as e:
#          print(f"Error interacting with OpenAI API: {e}")
#          return None

def validate_gaps_with_gpt4(gap_candidates):
    if not gap_candidates:
         return None
    try:
         # Revised prompt with more context per gap:
         prompt = (
             "You are an expert research analyst.\n\n"
             "The following potential research gaps have been identified from the analysis of scientific papers. "
             "For each gap, please provide a detailed summary that includes the context behind the gap, a clear "
             "description of the gap, and suggestions for future research directions. "
             "Be as specific as possible in your explanation.\n\n"
         )
         for i, gap in enumerate(gap_candidates[:5]):
             # Include topic name, keywords, and the sample gap text.
             topic_name = gap.get('topic_name', 'Unknown Topic')
             topic_keywords = gap.get('topic_keywords', 'No keywords provided')
             gap_text = gap.get('sample_gap_text', 'No detailed text provided.')
             prompt += (
                 f"Gap {i+1} (Topic: {topic_name}):\n"
                 f"Topic Keywords: {topic_keywords}\n"
                 f"Evidence: {gap_text}\n\n"
             )
         prompt += "Please provide a detailed summary for each gap."
         
         response = openai.chat.completions.create(
             model="gpt-4",
             messages=[
                 {"role": "system", "content": "You are an expert research analyst."},
                 {"role": "user", "content": prompt}
             ],
             max_tokens=1000,
             temperature=0.5,
         )
         summary = response.choices[0].message.content
         print("GPT-4 Summary/Validation:\n", summary)
         return summary
    except Exception as e:
         print(f"Error interacting with OpenAI API: {e}")
         return None


def identify_potential_gaps(papers_df, topic_info, network_analysis, trend_analysis):
    print("Identifying potential research gaps...")
    potential_gaps = []
    from collections import Counter
    topic_scores = Counter()
    gap_reasons = {}

    # --- Network Analysis Criteria (downweighted) ---
    # if network_analysis:
    #     isolated_topics = network_analysis.get('potentially_isolated_topics', [])
    #     for topic_id in isolated_topics:
    #         topic_scores[topic_id] += 0.5  # Lower weight for network connectivity
    #         gap_reasons.setdefault(topic_id, []).append("Low connectivity in citation network")
    #     avg_years = network_analysis.get('average_topic_year', {})
    #     current_year = pd.Timestamp.now().year
    #     for topic_id, avg_year in avg_years.items():
    #         if avg_year > current_year - 3:
    #             topic_scores[topic_id] += 0.5  # Lower weight for recent activity
    #             gap_reasons.setdefault(topic_id, []).append(f"Recent average publication year ({avg_year:.1f})")

    # --- Textual Indicator Criteria ---
    for idx, row in papers_df.iterrows():
        text_gap_score = len(row.get('gap_sentences', []))
        topic_id = row.get('topic_id', -1)
        if topic_id != -1:
            topic_scores[topic_id] += text_gap_score  # Full weight for text signals
            if text_gap_score > 0:
                gap_reasons.setdefault(topic_id, []).append(f"{text_gap_score} gap-indicative sentence(s) detected")

    # --- Trend Analysis Criteria ---
    if trend_analysis:
        low_count_recent = trend_analysis['trends'].get('low_count_recent_topics', {})
        for topic_id, count in low_count_recent.items():
            topic_scores[topic_id] += 2
            gap_reasons.setdefault(topic_id, []).append(f"Low total publications but recent activity ({count} recent)")
        common_future_words = [word for word, count in trend_analysis['limitations'].get('common_future_words', [])]
        topic_keywords_map = topic_info.set_index('Topic')['Name'].to_dict()
        for topic_id, keywords_str in topic_keywords_map.items():
            if topic_id == -1:
                continue
            topic_keywords = keywords_str.split('_')[1:]
            if any(future_word in topic_keywords for future_word in common_future_words):
                topic_scores[topic_id] += 1
                gap_reasons.setdefault(topic_id, []).append("Topic keywords align with common 'future work' terms")

    print("PICO-based gap analysis not implemented in detail.")

    # --- Aggregate Sample Textual Evidence per Topic ---
    topic_gap_samples = {}
    for idx, row in papers_df.iterrows():
        topic_id = row.get('topic_id', -1)
        if topic_id != -1:
            sample_texts = row.get('gap_sentences', [])
            if sample_texts:
                topic_gap_samples.setdefault(topic_id, []).extend(sample_texts)
    # For each topic, remove duplicates and summarize the collected gap-indicative sentences.
    for topic_id in topic_gap_samples:
        unique_samples = list(dict.fromkeys(topic_gap_samples[topic_id]))
        sample_text = " ".join(unique_samples[:5])  # Combine up to 5 sentences
        summarized_text = summarize_gap_text(sample_text)
        topic_gap_samples[topic_id] = summarized_text

    ranked_topics = topic_scores.most_common()
    for topic_id, score in ranked_topics:
        if score > 0:
            topic_details = topic_info[topic_info['Topic'] == topic_id].iloc[0]
            potential_gaps.append({
                'topic_id': topic_id,
                'topic_name': topic_details['Name'],
                'topic_keywords': topic_details['Representation'],
                'score': score,
                'reason': "; ".join(gap_reasons.get(topic_id, [])),
                'total_papers_in_topic': topic_details['Count'],
                'evidence': f"Score based on {len(gap_reasons.get(topic_id, []))} criteria.",
                'sample_gap_text': topic_gap_samples.get(topic_id, "No textual evidence provided.")
            })

    print(f"Identified {len(potential_gaps)} potential research gaps.")
    gaps_df = pd.DataFrame(potential_gaps)
    gaps_path = os.path.join(config.RESULTS_DIR, "potential_research_gaps.csv")
    gaps_df.to_csv(gaps_path, index=False)
    print(f"Potential gaps saved to {gaps_path}")

    gpt4_summary = validate_gaps_with_gpt4(potential_gaps)
    if gpt4_summary:
        summary_path = os.path.join(config.RESULTS_DIR, "gpt4_gap_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(gpt4_summary)
        print(f"GPT-4 summary saved to {summary_path}")

    return gaps_df

if __name__ == "__main__":
    print("\nStep 6 & 7: Starting Gap Identification...")
    papers_topics_path = os.path.join(config.PROCESSED_DIR, "papers_with_topics.csv")
    topic_info_path = os.path.join(config.RESULTS_DIR, "topics_info.csv")

    if os.path.exists(papers_topics_path) and os.path.exists(topic_info_path):
        papers_df = pd.read_csv(papers_topics_path)
        topic_info = pd.read_csv(topic_info_path)

        trend_analysis_results = analyze_trends_and_patterns(papers_df, topic_info)

        from step5_citation_analysis import analyze_citation_network, build_citation_network
        G = build_citation_network(papers_df)
        network_analysis_results = analyze_citation_network(G, papers_df)
        print("Using network analysis results from Step 5.")

        potential_gaps_df = identify_potential_gaps(
            papers_df,
            topic_info,
            network_analysis_results,
            trend_analysis_results
        )

        if not potential_gaps_df.empty:
            print("\nTop Potential Gaps Found:")
            print(potential_gaps_df.head())
        else:
            print("No significant gaps identified based on current criteria.")

        print("\nStep 6 & 7: Gap Identification Finished.")
    else:
        print("Error: Required input files not found. Ensure previous steps ran successfully.")
