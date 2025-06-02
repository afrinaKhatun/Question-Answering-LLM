import json
import re
from collections import defaultdict
from typing import List
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
#nltk.download("punkt")
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
nltk.download("stopwords")

# --- Load clustered metadata ---
with open("chunk_metadata_with_clusters.json", "r") as f:
    chunk_data = json.load(f)

# --- Group chunks by cluster ---
cluster_to_chunks = defaultdict(list)
for entry in chunk_data:
    text = entry.get("chunk_text", "")
    cluster_id = entry.get("cluster_id")
    if text and cluster_id is not None:
        cluster_to_chunks[cluster_id].append(text)

# --- LDA Topic Modeling ---
def extract_topics_from_chunks(chunks, num_topics=3, num_words=10):
    # Load stopwords
    stop_words = set(stopwords.words("english"))

    tokenized_chunks = []
    for chunk in chunks:
        # Tokenize using regex, keep only alphanumeric words
        #tokens = re.findall(r'\b\w+\b', chunk.lower())
        #filtered = [w for w in tokens if w not in stop_words and len(w)>2]
        filtered = [w for w in tokenizer.tokenize(chunk) if w.isalnum() and w not in stop_words and len(w)>2]
        tokenized_chunks.append(filtered)

    # Create dictionary and bag-of-words corpus
    dictionary = corpora.Dictionary(tokenized_chunks)
    corpus = [dictionary.doc2bow(text) for text in tokenized_chunks]

    # Train LDA model
    lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20)
    topics = lda.print_topics(num_topics=num_topics, num_words=num_words)
    return topics

# --- Extract topics per cluster ---
all_topics = {}
for cluster_id, chunks in cluster_to_chunks.items():
    print(f"\n🧠 Cluster {cluster_id} — {len(chunks)} chunks")
    topics = extract_topics_from_chunks(chunks)
    all_topics[cluster_id] = topics
    for topic_id, topic in topics:
        print(f"  Topic {topic_id}: {topic}")

# --- Optional: Save topics to file ---
with open("cluster_topics.json", "w") as f:
    json.dump(all_topics, f, indent=2)

print("\n✅ LDA topic modeling complete.")
