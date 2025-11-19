import os
import argparse
import json
import pandas as pd
from tqdm import tqdm
import torch
from langdetect import detect, LangDetectException
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared import get_vector_store

# --- Global Initialization of Translation Model ---
# Use a multi-language to English translation model
MODEL_NAME = "Helsinki-NLP/opus-mt-mul-en"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

try:
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    TRANSLATION_MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    print("Translation model loaded successfully.")
except Exception as e:
    print(f"Error loading translation model: {e}")
    TOKENIZER = None
    TRANSLATION_MODEL = None
# -------------------------------------------------

def translate_to_english(text: str) -> str:
    """
    Detects the language of the text and translates it to English if it's not already English.
    """
    if not text or not isinstance(text, str) or not TRANSLATION_MODEL:
        return text

    try:
        # Detect language
        lang = detect(text)
        if lang == "en":
            return text
    except LangDetectException:
        # If language detection fails, assume it might need translation or just proceed
        # This can happen with very short or ambiguous text
        pass

    try:
        # Translate the text to English
        inputs = TOKENIZER(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        translated_tokens = TRANSLATION_MODEL.generate(**inputs, max_length=512)
        translated_text = TOKENIZER.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        print(f"Translation failed for text snippet: '{text[:50]}...'. Error: {e}")
        return text # Return original text if translation fails


def ingest_csv(csv_path: str, index_name: str = "cord_idx", limit: int = 0, recreate: bool = False) -> dict:
    if not os.path.exists(csv_path):
        return {"status": "error", "message": f"CSV not found: {csv_path}"}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"status": "error", "message": f"Failed to read CSV: {e}"}

    required_cols = {"title", "abstract", "doi", "url"}
    if not required_cols.issubset(set(df.columns)):
        return {
            "status": "error",
            "message": f"CSV must contain columns: {sorted(list(required_cols))}"
        }

    df = df.dropna(subset=["abstract", "title"]).reset_index(drop=True)
    if limit and limit > 0:
        df = df.head(limit)

    texts = []
    metadatas = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing and Translating Rows"):
        
        # Translate title and abstract to English if necessary
        title = translate_to_english(row.get("title", ""))
        abstract = translate_to_english(row.get("abstract", ""))

        text_to_embed = f"{title}\n{abstract}"
        texts.append(text_to_embed)
        
        metadatas.append({
            "title": title,
            "doi": row.get("doi", ""),
            "url": row.get("url", "")
        })

    try:
        store = get_vector_store(index_name)
        if recreate:
            print(f"Clearing index: {index_name}")
            store.delete(delete_documents=True)
        
        print("Starting data ingestion...")
        ids = store.add_texts(texts=texts, metadatas=metadatas)
        print("Data ingestion complete.")
        
        return {"status": "ok", "index": index_name, "ingested": len(ids)}
    except Exception as e:
        return {"status": "error", "message": f"Vector ingest failed: {e}"}


def main():
    parser = argparse.ArgumentParser(description="Pre-ingest literature CSV into Redis vector index for RAG.")
    parser.add_argument("--csv_path", required=True, help="Path to CSV with title, abstract, doi, url columns")
    parser.add_argument("--index_name", default="cord_idx", help="Redis vector index name")
    parser.add_argument("--limit", type=int, default=30000, help="Optional row limit (0 means all)")
    parser.add_argument("--recreate", action="store_true", help="Recreate the index before ingesting data")

    args = parser.parse_args()

    result = ingest_csv(
        csv_path=args.csv_path,
        index_name=args.index_name,
        limit=args.limit,
        recreate=args.recreate
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
