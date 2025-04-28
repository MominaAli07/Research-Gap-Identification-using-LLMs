# step1_data_collection.py
import requests
import time
import os
import pandas as pd
import re
import config

def setup_directories():
    os.makedirs(config.RAW_PAPER_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    print("Directories checked/created.")

def fetch_semantic_scholar(query, max_results=5):
    print(f"Fetching from Semantic Scholar for query: '{query}'")
    papers = []
    params = {
        "query": query,
        "limit": max_results,
        "fields": "paperId,title,abstract,year,openAccessPdf,references"
    }
    headers = {}
    if config.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = config.SEMANTIC_SCHOLAR_API_KEY

    try:
        response = requests.get(config.SEMANTIC_SCHOLAR_SEARCH_URL, params=params, headers=headers)
        response.raise_for_status()
        data = response.json().get("data", [])
        for paper in data:
            paper_id = paper.get("paperId")
            if not paper_id:
                continue
            paper_entry = {
                "id": f"s2:{paper_id}",
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "year": paper.get("year", None),
                "pdf_url": paper.get("openAccessPdf", {}).get("url", ""),
                "references": [ref.get("paperId") for ref in paper.get("references", []) if ref.get("paperId")]
            }
            papers.append(paper_entry)
        print(f"Fetched {len(papers)} results from Semantic Scholar.")
    except Exception as e:
        print(f"Error fetching from Semantic Scholar: {e}")
    time.sleep(1)
    return papers

def download_full_text(paper_info, download_dir):
    pdf_url = paper_info.get('pdf_url')
    paper_id_safe = paper_info['id'].replace(':', '_')
    filepath = os.path.join(download_dir, f"{paper_id_safe}.pdf")

    if pdf_url and not os.path.exists(filepath):
        print(f"Attempting to download PDF: {pdf_url} for paper {paper_info['id']}")
        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            if 'application/pdf' in response.headers.get('Content-Type', '').lower():
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"Successfully downloaded {filepath}")
                return filepath
            else:
                print(f"Warning: Link {pdf_url} did not return PDF content type.")
                return None
        except Exception as e:
            print(f"Error downloading PDF {pdf_url}: {e}")
            return None
    elif os.path.exists(filepath):
        print(f"PDF already exists: {filepath}")
        return filepath
    else:
        return None

def fetch_and_prepare_papers(keywords, max_per_source):
    setup_directories()
    all_papers_metadata = []
    unique_ids = set()

    for keyword in keywords:
        print(f"\n--- Processing Keyword: {keyword} ---")
        papers = fetch_semantic_scholar(keyword, max_results=max_per_source)
        for paper in papers:
            if paper['id'] not in unique_ids:
                all_papers_metadata.append(paper)
                unique_ids.add(paper['id'])
    
    print(f"\n--- Total Unique Papers Found: {len(all_papers_metadata)} ---")
    # Download PDFs and extract PDF paths
    for paper in all_papers_metadata:
        pdf_path = download_full_text(paper, config.RAW_PAPER_DIR)
        paper['local_pdf_path'] = pdf_path
        # We'll extract full text later in Step 2; if no PDF, store empty string.
        if not pdf_path:
            paper['full_text'] = ""
    metadata_df = pd.DataFrame(all_papers_metadata)
    metadata_path = os.path.join(config.PROCESSED_DIR, "collected_metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)
    print(f"Metadata saved to {metadata_path}")
    return metadata_df

if __name__ == "__main__":
    print("Step 1: Starting Data Collection using Semantic Scholar...")
    collected_data = fetch_and_prepare_papers(config.SEARCH_KEYWORDS, config.MAX_PAPERS_PER_SOURCE)
    print("\nStep 1: Data Collection Finished.")
