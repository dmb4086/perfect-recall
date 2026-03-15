import re
import nltk
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

class BM25Retriever:
    def __init__(self, memory_md_path):
        self.memory_md_path = memory_md_path
        self.stemmer = PorterStemmer()
        self.entries = []
        self.corpus = []
        self.bm25 = None
        self._load_and_index()

    def _tokenize_and_stem(self, text):
        words = nltk.word_tokenize(text.lower())
        # keep only alphanumeric
        words = [w for w in words if w.isalnum()]
        stemmed = [self.stemmer.stem(w) for w in words]

        # Add bigrams
        bigrams = [f"{stemmed[i]}_{stemmed[i+1]}" for i in range(len(stemmed)-1)]
        return stemmed + bigrams

    def _load_and_index(self):
        with open(self.memory_md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by headers like "## [2024-03-20...]"
        raw_entries = re.split(r'\n## \[\d{4}-\d{2}-\d{2}[^\]]*\]', content)

        # the first split might be preamble/title, we filter out short ones or preamble
        for entry in raw_entries:
            entry = entry.strip()
            if len(entry) > 10 and not entry.startswith("# Perfect Recall Synthetic"):
                self.entries.append(entry)

        tokenized_corpus = [self._tokenize_and_stem(doc) for doc in self.entries]
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, limit: int = 5):
        if not self.bm25:
            return []

        tokenized_query = self._tokenize_and_stem(query)
        doc_scores = self.bm25.get_scores(tokenized_query)

        # get top K
        top_k_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:limit]

        results = []
        for idx in top_k_indices:
            if doc_scores[idx] > 0:
                results.append({
                    "content": self.entries[idx],
                    "score": doc_scores[idx]
                })

        return results

if __name__ == "__main__":
    import sys
    # For basic testing
    retriever = BM25Retriever("synthetic_memory.md")
    res = retriever.search("indentation style", limit=2)
    for r in res:
        print(f"Score: {r['score']:.2f}\n{r['content']}\n")
