from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import json
import numpy as np
import nltk

# --- Setup ---
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# --- POS tag to WordNet tag mapping ---
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

# --- Lemmatized Tokenization ---
def tokenize_and_lemmatize(text):
    tokens = [w for w in tokenizer.tokenize(text) if w.isalnum() and w.lower() not in stop_words and len(w) > 2]
    pos_tags = pos_tag(tokens)
    lemmatized = [
        lemmatizer.lemmatize(token.lower(), get_wordnet_pos(pos))
        for token, pos in pos_tags
    ]
    return lemmatized

# Load clustered data
with open("chunk_metadata_with_clusters.json") as f:
    data = json.load(f)

# Group chunks by cluster
cluster_chunks = defaultdict(list)
for entry in data:
    cluster_id = entry["cluster_id"]
    chunk_text = entry["chunk_text"].lower()
    cluster_chunks[cluster_id].append(chunk_text)

# Store output for saving
topic_keywords = defaultdict(lambda: defaultdict(dict))

def extract_hybrid_topics(cluster_chunks, num_topics=3, num_words=10):
    for cluster_id, docs in cluster_chunks.items():
        print(f"\n🧠 Cluster {cluster_id} — {len(docs)} chunks")

        # Preprocessing
        tokenized = []
        clean_docs = []
        for doc in docs:
            tokens = tokenize_and_lemmatize(doc)
            if tokens:
                tokenized.append(tokens)
                clean_docs.append(" ".join(tokens))

        # Skip empty clusters
        if not tokenized or not clean_docs:
            continue

        # LDA Topic Modeling
        dictionary = corpora.Dictionary(tokenized)
        corpus = [dictionary.doc2bow(text) for text in tokenized]
        lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20)

        topic_doc_map = defaultdict(list)
        for doc_idx, bow in enumerate(corpus):
            topic_probs = lda_model.get_document_topics(bow)
            top_topic = max(topic_probs, key=lambda x: x[1])[0]
            topic_doc_map[top_topic].append(clean_docs[doc_idx])

        for topic_id in range(num_topics):
            print(f"\n  🔹 Topic {topic_id}")

            # --- LDA words ---
            lda_words = [w for w, _ in lda_model.show_topic(topic_id, topn=num_words)]
            print(f"    📚 LDA Words:     {', '.join(lda_words)}")

            # --- TF-IDF words ---
            topic_docs = topic_doc_map.get(topic_id, [])
            if topic_docs:
                tfidf = TfidfVectorizer(max_df=0.9, min_df=1)
                tfidf_matrix = tfidf.fit_transform(topic_docs)
                feature_array = np.array(tfidf.get_feature_names_out())
                tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
                top_indices = tfidf_scores.argsort()[::-1][:num_words]
                tfidf_words = feature_array[top_indices]
                print(f"    📈 TF-IDF Words:  {', '.join(tfidf_words)}")
            else:
                tfidf_words = []

            topic_keywords[cluster_id][topic_id]["lda"] = lda_words
            topic_keywords[cluster_id][topic_id]["tfidf"] = tfidf_words

            # --- Combined top words ---
            combined = list(dict.fromkeys(lda_words + list(tfidf_words)))[:num_words]
            #print(f"    🔗 Combined:      {', '.join(combined)}")
# Convert keys and values to pure Python types
def convert_to_serializable(d):
    if isinstance(d, dict):
        return {str(k): convert_to_serializable(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_to_serializable(i) for i in d]
    elif isinstance(d, np.ndarray):
        return d.tolist()
    elif isinstance(d, (np.int64, np.int32, np.float32, np.float64)):
        return d.item()
    else:
        return d

# Run it
extract_hybrid_topics(cluster_chunks)

# Save to JSON
with open("combined_cluster_topic_keywords.json", "w") as f:
    json.dump(convert_to_serializable(topic_keywords), f, indent=2)

print("\n✅ Saved topic keywords to 'cluster_topic_keywords.json'")