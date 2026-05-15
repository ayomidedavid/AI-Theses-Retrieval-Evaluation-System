import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

class EnsembleRanker:
    """
    Implements BMTS-Rank (BM25 + TF-IDF Stack)
    """
    def __init__(self, alpha=0.304, k1=2.05, b=0.432, norm_strategy="max"):
        self.alpha = alpha
        self.k1 = k1
        self.b = b
        self.norm_strategy = norm_strategy
        self.tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=2)
        
        # Stored indexes
        self.tfidf_matrix = None
        self.bm25 = None
        
        # Stored Original Data references
        self.documents = []  # Expected to be a list of cleaned/processed text strings

    def fit(self, documents):
        """Builds the TF-IDF and BM25 indices simultaneously"""
        self.documents = documents
        
        # Fit TF-IDF
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        
        # Fit BM25 (Requires tokenized lists instead of strings)
        tokenized_docs = [doc.split(" ") for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)

    def _normalize_scores(self, scores):
        """Applies normalization mapping scores"""
        if self.norm_strategy == "max":
            max_val = np.max(scores)
            if max_val == 0:
                return np.zeros_like(scores)
            return scores / max_val
        elif self.norm_strategy == "l1":
            sum_val = np.sum(np.abs(scores))
            if sum_val == 0:
                return np.zeros_like(scores)
            return scores / sum_val
        elif self.norm_strategy == "l2":
            l2_norm = np.linalg.norm(scores)
            if l2_norm == 0:
                return np.zeros_like(scores)
            return scores / l2_norm
        elif self.norm_strategy == "zscore":
            std_val = np.std(scores)
            if std_val == 0:
                return np.zeros_like(scores)
            return (scores - np.mean(scores)) / std_val
        else: # minmax
            min_val = np.min(scores)
            max_val = np.max(scores)
            if max_val - min_val == 0:
                return np.zeros_like(scores)
            return (scores - min_val) / (max_val - min_val)

    def transform(self, query):
        """
        Takes a processed query string, calculates both sub-models,
        normalizes them, and merges them using alpha weighting.
        Returns a list of tuples: (Doc Index, Ensemble Score, TF-IDF Score, BM25 Score)
        """
        # 1. TF-IDF Scoring
        query_vec = self.tfidf_vectorizer.transform([query])
        # Cosine similarity between query and all docs
        tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        norm_tfidf = self._normalize_scores(tfidf_scores)
        
        # 2. BM25 Scoring
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        norm_bm25 = self._normalize_scores(bm25_scores)
        
        # 3. Stacked Ensemble Fusion
        ensemble_scores = (self.alpha * norm_tfidf) + ((1.0 - self.alpha) * norm_bm25)
        
        # 4. Pack the results
        results = []
        for i in range(len(self.documents)):
            results.append({
                "doc_id": i,
                "score": float(ensemble_scores[i]),
                "tfidf": float(tfidf_scores[i]),     # raw
                "bm25": float(bm25_scores[i])        # raw
            })
            
        # Sort by ensemble score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
