import os
import argparse
import json
import pandas as pd

# 复用项目中的向量库与Embedding配置
from tools import _get_vector_store  # noqa: E402


def ingest_csv(csv_path: str, index_name: str = "cord_idx", limit: int = 0) -> dict:
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

    df = df.dropna(subset=["abstract"]).reset_index(drop=True)
    if limit and limit > 0:
        df = df.head(limit)

    texts = []
    metadatas = []
    for _, row in df.iterrows():
        text = f"{row['title']}\n{row['abstract']}"
        texts.append(text)
        metadatas.append({
            "title": row.get("title", ""),
            "doi": row.get("doi", ""),
            "url": row.get("url", "")
        })

    try:
        store = _get_vector_store(index_name)
        ids = store.add_texts(texts=texts, metadatas=metadatas)
        return {"status": "ok", "index": index_name, "ingested": len(ids)}
    except Exception as e:
        return {"status": "error", "message": f"Vector ingest failed: {e}"}


def main():
    parser = argparse.ArgumentParser(description="Pre-ingest literature CSV into Redis vector index for RAG.")
    parser.add_argument("--csv_path", required=True, help="Path to CSV with title, abstract, doi, url columns")
    parser.add_argument("--index_name", default="cord_idx", help="Redis vector index name")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit (0 means all)")

    args = parser.parse_args()

    result = ingest_csv(csv_path=args.csv_path, index_name=args.index_name, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
