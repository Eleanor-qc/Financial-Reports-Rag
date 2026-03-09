import os
import pickle
import time
import faiss
import numpy as np
import vertexai

from google.cloud import storage
from vertexai.preview.language_models import TextEmbeddingModel, TextEmbeddingInput

PROJECT_ID = "qc2360-ieor4526-fall2025"
BUCKET_NAME = "qc2360-fall2025-bucket"

CHUNKS_TEXT_BLOB = "rag/chunks_text.pkl"
CHUNKS_METADATA_BLOB = "rag/chunks_metadata.pkl"
FAISS_INDEX_BLOB = "rag/financial_reports_faiss.pkl"

LOCAL_DATA_DIR = "data"
LOCAL_CHUNKS_TEXT = os.path.join(LOCAL_DATA_DIR, "chunks_text.pkl")
LOCAL_CHUNKS_METADATA = os.path.join(LOCAL_DATA_DIR, "chunks_metadata.pkl")
LOCAL_FAISS_INDEX = os.path.join(LOCAL_DATA_DIR, "financial_reports_faiss.pkl")

def download_chunks_from_gcs():
    """Download prepared chunks from GCS to local container storage."""
    print("Downloading chunks from GCS...")

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)

    bucket.blob(CHUNKS_TEXT_BLOB).download_to_filename(LOCAL_CHUNKS_TEXT)
    bucket.blob(CHUNKS_METADATA_BLOB).download_to_filename(LOCAL_CHUNKS_METADATA)

    print("Chunks downloaded.")


def load_chunks():
    """Load chunk text and metadata from local files and validate alignment."""
    with open(LOCAL_CHUNKS_TEXT, "rb") as f:
        chunks_text = pickle.load(f)

    with open(LOCAL_CHUNKS_METADATA, "rb") as f:
        chunks_metadata = pickle.load(f)

    if not chunks_text:
        raise ValueError("chunks_text.pkl is empty.")

    if not chunks_metadata:
        raise ValueError("chunks_metadata.pkl is empty.")

    if len(chunks_text) != len(chunks_metadata):
        raise ValueError(
            f"Length mismatch: chunks_text={len(chunks_text)}, "
            f"chunks_metadata={len(chunks_metadata)}"
        )

    texts = []
    titles = []
    chunk_ids = []
    metadata = []

    for chunk_record, meta_record in zip(chunks_text, chunks_metadata):
        chunk_id_text = chunk_record.get("chunk_id")
        chunk_id_meta = meta_record.get("chunk_id")

        if chunk_id_text != chunk_id_meta:
            raise ValueError(
                f"chunk_id mismatch: text={chunk_id_text}, metadata={chunk_id_meta}"
            )

        text = chunk_record.get("content", "").strip()
        title = chunk_record.get("title", "").strip()

        if not text:
            continue

        texts.append(text)
        titles.append(title)
        chunk_ids.append(chunk_id_text)
        metadata.append(meta_record)

    if not texts:
        raise ValueError("No valid non-empty chunk texts found.")

    print(f"Loaded {len(texts)} valid chunks.")
    return texts, metadata, titles, chunk_ids

def prepare_chunk_records(texts, metadata, titles, chunk_ids):
    records = []
    for text, meta, title, chunk_id in zip(texts, metadata, titles, chunk_ids):
        records.append({
            "chunk_id": chunk_id,
            "metadata": meta,
            "title": title,
            "content": text,
        })
    return records

def embed_chunks(chunk_records, model, batch_size: int = 32, max_retries: int = 5):
    """
    Generate embeddings for chunk records using the notebook-style logic.

    Each chunk record should be a dict like:
    {
        "chunk_id": ...,
        "title": ...,
        "content": ...,
        "metadata": ...
    }

    Returns:
        embedded_chunks: list[dict]
            [
                {
                    "vector": np.ndarray(dtype=float32),
                    "chunk_id": ...,
                    "metadata": ...,
                    "title": ...,
                    "content": ...
                },
                ...
            ]
    """
    embedded_chunks = []
    valid_items = []
    for item in chunk_records:
        content = item["content"].strip()
        if content:
            valid_items.append(item)
    
    for start in range(0, len(valid_items), batch_size):
        batch = valid_items[start:start + batch_size]
    
        emb_inputs = [
            TextEmbeddingInput(
                task_type="RETRIEVAL_DOCUMENT",
                title=item["title"],
                text=item["content"].strip()
            )
            for item in batch
        ]
    
        embeddings = None
        for retry in range(max_retries):
            try:
                embeddings = model.get_embeddings(emb_inputs)
                break
            except Exception as e:
                wait_time = 2 ** retry
                print(f"[WARN] Batch {start // batch_size + 1} failed (retry {retry + 1}/{max_retries}): {e}")
                time.sleep(wait_time)
    
        if embeddings is None:
            print(f"[SKIP] Batch {start // batch_size + 1} failed after {max_retries} retries.")
            continue
    
        for item, embedding in zip(batch, embeddings):
            embedded_chunks.append({
                "vector": np.array(embedding.values, dtype="float32"),
                "chunk_id": item["chunk_id"],
                "metadata": item["metadata"],
                "title": item["title"],
                "content": item["content"].strip()
            })
    
        print(f"Processed batch {start // batch_size + 1} / {(len(valid_items) + batch_size - 1) // batch_size}")

    # Report total number of successfully embedded chunks
    print(f"Generated embeddings: {len(embedded_chunks)}")
    
    return embedded_chunks

def build_faiss_index(chunk_records, model):
    print("Embedding chunks...")
    embedded_chunks = embed_chunks(chunk_records, model=model)

    if not embedded_chunks:
        raise ValueError("No embedded chunks were generated.")

    X = np.vstack([c["vector"] for c in embedded_chunks]).astype("float32")
    dim = X.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(X)

    print(f"FAISS index built with {index.ntotal} vectors.")
    print(f"Embedding dimension: {dim}")

    return index, embedded_chunks

def save_index(index, embedded_chunks):
    if index.ntotal != len(embedded_chunks):
        raise ValueError(
            f"Index/vector count mismatch: index.ntotal={index.ntotal}, "
            f"embedded_chunks={len(embedded_chunks)}"
        )

    with open(LOCAL_FAISS_INDEX, "wb") as f:
        pickle.dump(
            {
                "index": index,
                "chunk_ids": [c["chunk_id"] for c in embedded_chunks],
                "metadata": [c["metadata"] for c in embedded_chunks],
                "titles": [c["title"] for c in embedded_chunks],
                "contents": [c["content"] for c in embedded_chunks],
            },
            f,
        )

    print(f"Saved FAISS package to {LOCAL_FAISS_INDEX}")

def upload_faiss_to_gcs():
    """Upload the built FAISS index to GCS."""
    print("Uploading FAISS index to GCS...")

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(FAISS_INDEX_BLOB)
    blob.upload_from_filename(LOCAL_FAISS_INDEX)

    print(f"FAISS index uploaded to gs://{BUCKET_NAME}/{FAISS_INDEX_BLOB}")

def main():
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

    download_chunks_from_gcs()

    texts, metadata, titles, chunk_ids = load_chunks()

    chunk_records = prepare_chunk_records(texts, metadata, titles, chunk_ids)

    vertexai.init(project="qc2360-ieor4526-fall2025", location="us-central1")
    model = TextEmbeddingModel.from_pretrained("text-embedding-005")

    index, embedded_chunks = build_faiss_index(chunk_records, model=model)

    save_index(index, embedded_chunks)

    upload_faiss_to_gcs()

    print("Index build completed successfully.")

if __name__ == "__main__":
    main()