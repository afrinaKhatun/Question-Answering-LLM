import os
import openai
import tiktoken
import numpy as np
import faiss
from typing import List, Dict
import re

# --- Config ---
openai.api_key = "your-openai-api-key"
root_folder = "/Users/afrinakhatun/Documents/About UCI/PhD Course/SWE 215 Dynamic Testing/project/Question_Answer_Dataset_v1.2/"
max_tokens_per_chunk = 512
model_name = "text-embedding-3-large"
batch_size = 10  # OpenAI allows batching for embeddings

# --- Chunking Helper ---
def chunk_text(text: str, max_tokens: int = 512) -> List[str]:
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [enc.decode(chunk) for chunk in chunks]

# --- Walk Folders & Collect Chunks ---
def collect_chunks_from_documents(root_folder: str) -> List[Dict]:
    chunks_info = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if re.fullmatch(r'a\d+\.txt', file):
                full_path = os.path.join(subdir, file)
                with open(full_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                chunks = chunk_text(text)
                for i, chunk in enumerate(chunks):
                    chunks_info.append({
                        "doc_path": full_path,
                        "chunk_id": i,
                        "chunk_text": chunk
                    })
    return chunks_info

# --- Batch Embedding ---
def embed_chunks_batch(chunks: List[str]) -> List[List[float]]:
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            response = openai.embeddings.create(
                model=model_name,
                input=batch
            )
            for e in response.data:
                embeddings.append(e.embedding)
        except Exception as e:
            print(f"Batch embedding failed: {e}")
    return embeddings

# --- Save to FAISS Index ---
def save_faiss_index(embeddings: List[List[float]], faiss_path: str = "index.faiss"):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, faiss_path)
    print(f"FAISS index saved to {faiss_path}")

# --- Save Metadata for Later Retrieval ---
def save_metadata(chunks_info: List[Dict], filepath: str = "chunk_metadata.json"):
    import json
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(chunks_info, f, indent=2)
    print(f"Metadata saved to {filepath}")


if __name__ == "__main__":
    print("📥 Collecting text chunks...")
    chunk_data = collect_chunks_from_documents(root_folder)
    
    print(f"🔢 Total chunks: {len(chunk_data)}")
    all_text_chunks = [c["chunk_text"] for c in chunk_data]

    print("📡 Sending to OpenAI (batched)...")
    all_embeddings = embed_chunks_batch(all_text_chunks)

    # Attach embeddings back to metadata
    for i, emb in enumerate(all_embeddings):
        chunk_data[i]["embedding"] = emb

    print("💾 Saving FAISS index and metadata...")
    save_faiss_index(all_embeddings, "index.faiss")
    save_metadata(chunk_data, "chunk_metadata.json")
