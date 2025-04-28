# config.py
import os

# --- General Directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PAPER_DIR = os.path.join(BASE_DIR, "raw_papers")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# --- Semantic Scholar API ---
SEMANTIC_SCHOLAR_SEARCH_URL = "Paste your API key here"
# You can add your API key if needed. (The public endpoint is rate-limited.)
SEMANTIC_SCHOLAR_API_KEY = "Paste your API key here"  # Optional: If you have one

# --- Search Settings ---
SEARCH_KEYWORDS = ["Machine learning healthcare"]

def set_search_keyword(keyword):
    global SEARCH_KEYWORDS
    SEARCH_KEYWORDS = [keyword]


MAX_PAPERS_PER_SOURCE = 5  # Now fetching 50 papers

# --- spaCy ---
SPACY_MODEL = "en_core_web_sm"

# --- BERTopic ---
BERTOPIC_MIN_TOPIC_SIZE = 3  # Adjust as appropriate for 50 papers

# --- GPT-4 (OpenAI API) ---
# Replace with your actual OpenAI API key.
OPENAI_API_KEY = "Paste your API key here"
