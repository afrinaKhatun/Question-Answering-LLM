from sklearn.cluster import KMeans
import numpy as np
import json
from typing import List

# -- Helper to check if an embedding is valid

def is_valid_vector(vec):
    if not vec or not isinstance(vec, list):
        return False
    
    arr = np.array(vec, dtype=np.float32)
    return (
        np.all(np.isfinite(arr)) and  # no NaN, inf
        not np.all(arr == 0.0) and    # not a zero vector
        np.max(np.abs(arr)) < 1000    # guard against huge values
    )

# -- K-means clustering
def cluster_embeddings(embeddings: List[List[float]], num_clusters):
    X = np.array(embeddings).astype("float32")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels, kmeans

# -- Load chunk metadata
with open("chunk_metadata.json", "r") as f:
    chunk_data = json.load(f)

# -- Filter invalid embeddings
valid_data=[]
for entry in chunk_data:
    embedding = entry.get("embedding")
    if is_valid_vector(embedding):
        valid_data.append(entry)
    else:
        print(embedding)

valid_data = [entry for entry in chunk_data if is_valid_vector(entry.get("embedding"))]
valid_embeddings = [entry["embedding"] for entry in valid_data]

print(f"✅ Loaded {len(chunk_data)} entries, {len(valid_embeddings)} valid for clustering.")

# -- Cluster
num_clusters = 7
labels, kmeans_model = cluster_embeddings(valid_embeddings, num_clusters)

# -- Attach cluster IDs to valid entries
for i, label in enumerate(labels):
    valid_data[i]["cluster_id"] = int(label)

# -- Save filtered & clustered metadata
with open("chunk_metadata_with_clusters.json", "w") as f:
    json.dump(valid_data, f, indent=2)

print(f"✅ Clustering complete: {num_clusters} clusters assigned.")
