# retriever.py
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from pathlib import Path

EMB_MODEL_NAME = "all-MiniLM-L6-v2"
EMB_DIM = 384
FAISS_DIR = Path("../faiss_index")

class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(EMB_MODEL_NAME)
        self.index = faiss.read_index(str(FAISS_DIR / "faiss.index"))
        with open(FAISS_DIR / "meta.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def embed(self, text: str) -> np.ndarray:
        """
        Encode une question/texte et renvoie un vecteur (1, d) float32 normalisé.
        """
        # encode renvoie (d,) pour un seul texte => on le force en (1, d)
        emb = self.model.encode([text])  # liste => shape (1, d)
        emb = np.array(emb).astype("float32")
        faiss.normalize_L2(emb)
        return emb  # shape (1, d)

    def retrieve(self, query: str, top_k: int = 5):
        """
        Cherche les top_k passages les plus pertinents pour la query.
        """
        q_emb = self.embed(query)  # (1, d)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            m = self.meta[idx]
            results.append({
                "score": float(score),
                "id": m["id"],
                "source": m["source"],
                "text": m["text"]
            })
        return results
