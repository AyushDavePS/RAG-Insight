from .models import Candidate


class Reranker:
    def __init__(self, model_name):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rank(self, query, candidates):
        if not candidates:
            return []
        scores = self.model.predict([(query, candidate.chunk.text) for candidate in candidates])
        return sorted([Candidate(c.chunk, float(score))
                       for c, score in zip(candidates, scores, strict=True)],
                      key=lambda c: c.score, reverse=True)
