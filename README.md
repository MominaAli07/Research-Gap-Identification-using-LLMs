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
