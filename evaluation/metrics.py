def evidence_recall(context, expected):
    """Recall over gold source passages, invariant to chunk ID changes.

    Each label identifies a filename and a verbatim evidence substring.
    Returns None for questions without gold evidence.
    """
    if not expected:
        return None
    hits = sum(any(c.filename == label["filename"] and label["quote"] in c.text
                   for c in context) for label in expected)
    return hits / len(expected)
