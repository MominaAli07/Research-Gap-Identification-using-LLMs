# step2_text_processing.py
import pandas as pd
import os
import re
import spacy
import config
from transformers import pipeline
import concurrent.futures
import fitz  # PyMuPDF


try:
    nlp = spacy.load(config.SPACY_MODEL)
    print(f"Loaded spaCy model: {config.SPACY_MODEL}")
except OSError:
    print(f"spaCy model '{config.SPACY_MODEL}' not found. Run: python -m spacy download {config.SPACY_MODEL}")
    nlp = None

# Initialize zero-shot classification for gap extraction
gap_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def extract_texts_concurrently(pdf_paths):
    texts = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_pdf = {executor.submit(extract_text_from_pdf, path): path for path in pdf_paths}
        for future in concurrent.futures.as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                texts[pdf_path] = future.result()
            except Exception as exc:
                print(f"Error processing {pdf_path}: {exc}")
    return texts

def clean_text(text):
    if not text or not nlp:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\.\?!]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])
    return lemmatized_text

def fallback_section_extraction(text):
    fallback = ""
    if text:
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            if re.search(r"\b(challenge|open question|further research|need to investigate)\b", sentence, re.IGNORECASE):
                fallback += sentence.strip() + ". "
    return fallback.strip()

def extract_gap_sentences(text):
    if not text:
        return []
    sentences = re.split(r'[.!?]+', text)
    candidate_sentences = [s.strip() for s in sentences if s.strip()]
    labels = ["research gap", "limitation", "future work", "open question", "challenge", "unexplored"]
    gap_sentences = []
    for sentence in candidate_sentences:
        result = gap_classifier(sentence, labels)
        if max(result["scores"]) > 0.5:
            gap_sentences.append(sentence)
    return gap_sentences

def process_papers(metadata_df):
    processed_data = []
    pdf_paths = metadata_df['local_pdf_path'].dropna().tolist()
    extracted_texts = extract_texts_concurrently(pdf_paths)
    
    for index, row in metadata_df.iterrows():
        print(f"Processing paper {index + 1}/{len(metadata_df)}: {row['id']}")
        paper_id = row['id']
        abstract = row.get('abstract', '')
        conclusion_raw = row.get('conclusion', '')
        future_work_raw = row.get('future_work', '')
        limitations_raw = row.get('limitations', '')
        pdf_path = row.get('local_pdf_path')
        
        # Use concurrently extracted text if available
        full_text = extracted_texts.get(pdf_path, row.get('full_text', ''))
        
        cleaned_abstract = clean_text(abstract) if abstract else ''
        cleaned_conclusion = clean_text(conclusion_raw) if conclusion_raw else ''
        cleaned_future_work = clean_text(future_work_raw) if future_work_raw else ''
        cleaned_limitations = clean_text(limitations_raw) if limitations_raw else ''
        fallback_text = fallback_section_extraction(full_text) if not (cleaned_conclusion or cleaned_future_work or cleaned_limitations) else ""
        gap_sentences = extract_gap_sentences(full_text) if full_text else []
        
        processed_data.append({
            'id': paper_id,
            'title': row.get('title'),
            'year': row.get('year'),
            'abstract_cleaned': cleaned_abstract,
            'conclusion_cleaned': cleaned_conclusion,
            'future_work_cleaned': cleaned_future_work,
            'limitations_cleaned': cleaned_limitations,
            'fallback_text': fallback_text,
            'gap_sentences': gap_sentences,
            'local_pdf_path': pdf_path,
            'full_text': full_text
        })

    processed_df = pd.DataFrame(processed_data)
    processed_path = os.path.join(config.PROCESSED_DIR, "processed_papers.csv")
    processed_df.to_csv(processed_path, index=False)
    print(f"\nProcessed data saved to {processed_path}")
    return processed_df

if __name__ == "__main__":
    print("\nStep 2: Starting Text Processing...")
    metadata_path = os.path.join(config.PROCESSED_DIR, "collected_metadata.csv")
    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        processed_df = process_papers(metadata_df)
        print("\nStep 2: Text Processing Finished.")
    else:
        print(f"Error: Metadata file not found at {metadata_path}. Run Step 1 first.")


