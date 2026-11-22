
import numpy as np


class SemanticCache:

    def __init__(self, embedding_model, similarity_threshold=0.85):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.cache = []

        self.hits = 0
        self.misses = 0

    def _get_embedding(self, text):
        embedding = self.embedding_model.embed_query(text)
        return np.array(embedding)

    def _cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get(self, question):

        query_embedding = self._get_embedding(question)

        best_similarity = -1
        best_entry = None

        for entry in self.cache:

            similarity = self._cosine_similarity(
                query_embedding,
                entry["embedding"]
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry

        if (
            best_entry is not None
            and best_similarity >= self.similarity_threshold
        ):
            self.hits += 1

            print(
                f"Semantic cache HIT | "
                f"similarity: {best_similarity:.3f}"
            )

            return best_entry["result"]

        self.misses += 1

        print(
            f"Semantic cache MISS | "
            f"best similarity: {best_similarity:.3f}"
        )

        return None

    def add(self, question, result):

        question_embedding = self._get_embedding(question)

        entry = {
            "question": question,
            "result": result,
            "embedding": question_embedding
        }

        self.cache.append(entry)

        print(f"Added to semantic cache | entries: {len(self.cache)}")

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self):

        total = self.hits + self.misses

        hit_rate = (
            self.hits / total
            if total > 0
            else 0
        )

        return {
            "hits": self.hits,
            "misses": self.misses,
            "entries": len(self.cache),
            "hit_rate": round(hit_rate * 100, 2)
        }
