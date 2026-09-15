import math
import re
from collections import Counter

from .models import Candidate


def terms(text):
    return re.findall(r"[\w]+(?:[-.][\w]+)*", text.lower())


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def dense_search(query_vector, rows, k):
    return sorted([Candidate(chunk, dot(query_vector, vector)) for chunk, vector in rows],
                  key=lambda item: item.score, reverse=True)[:k]


def bm25_search(query, rows, k):
    if not rows:
        return []
    counters = [Counter(terms(chunk.text)) for chunk, _ in rows]
    lengths = [sum(counter.values()) for counter in counters]
    average = sum(lengths) / len(lengths) or 1
    frequency = Counter(term for counter in counters for term in counter)
    results = []
    for (chunk, _), counter, length in zip(rows, counters, lengths, strict=True):
        score = 0.0
        for term in set(terms(query)):
            tf = counter[term]
            idf = math.log(1 + (len(rows) - frequency[term] + 0.5) / (frequency[term] + 0.5))
            score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / average))
        if score > 0:
            results.append(Candidate(chunk, score))
    return sorted(results, key=lambda item: item.score, reverse=True)[:k]


def fuse(rankings, k=60):
    scores, chunks = {}, {}
    for ranking in rankings:
        for rank, candidate in enumerate(ranking, 1):
            key = candidate.chunk.chunk_id
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
            chunks[key] = candidate.chunk
    return sorted([Candidate(chunks[key], score) for key, score in scores.items()],
                  key=lambda item: item.score, reverse=True)


def select_context(candidates, rows, settings, tokenizer):
    vectors = {chunk.chunk_id: vector for chunk, vector in rows}
    remaining, selected, used = list(candidates), [], 0
    # Rank-normalized relevance avoids mixing uncalibrated reranker logits with cosine.
    relevance = {c.chunk.chunk_id: 1 - i / max(len(candidates), 1)
                 for i, c in enumerate(candidates)}
    while remaining and len(selected) < settings.context_k:
        if settings.mmr and selected:
            def utility(candidate):
                vector = vectors[candidate.chunk.chunk_id]
                redundancy = max(dot(vector, vectors[c.chunk.chunk_id]) for c in selected)
                return settings.mmr_lambda * relevance[candidate.chunk.chunk_id] - (1 - settings.mmr_lambda) * redundancy
            best = max(remaining, key=utility)
        else:
            best = remaining[0]
        remaining.remove(best)
        size = len(tokenizer.encode(best.chunk.text, add_special_tokens=False))
        if used + size <= settings.context_tokens:
            selected.append(best)
            used += size
    return selected
