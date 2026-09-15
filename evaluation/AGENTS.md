# Evaluation Guidance

These instructions apply to `evaluation/`, its datasets, metrics, and experiment runner.

## Dataset integrity

- Keep examples as valid JSON Lines: one complete JSON object per line.
- Give every example a unique, stable ID.
- Preserve explicit `dev` and `test` splits.
- Tune configurations only with the development split.
- Do not inspect or repeatedly run the held-out split to guide tuning.
- Represent answerable, multi-passage, exact-identifier, paraphrase, near-duplicate, conflicting-evidence, and unanswerable cases as the corpus grows.
- For answerable questions, label the smallest sufficient evidence passages and expected sources.
- For unanswerable questions, use an empty evidence list and a null expected answer.
- Ensure every quoted evidence substring exists verbatim in its named source document.
- Never alter expected labels merely to make a configuration score higher; document genuine labelling corrections.

## Experiment discipline

- Use isolated indexes or collections for incompatible configurations.
- Change one primary variable at a time when attributing improvements.
- Record configuration, model identifiers and revisions, corpus version, split, timestamp, hardware context, latency, and per-question output.
- Preserve initial and post-rewrite retrieval results separately.
- Distinguish retrieval quality, answerability decisions, answer correctness, faithfulness, citation support, and abstention behavior.
- Do not average non-applicable metrics into zero.
- Include denominators, failure cases, and uncertainty when reporting aggregate results.
- Do not publish placeholder or fabricated scores.

## Required review

- Automated metrics do not replace manual review of answer correctness, faithfulness, and citation support.
- Leave manual fields unset until a reviewer evaluates them.
- Validate any LLM judge against a human-scored sample before relying on it.
- Keep generated run artifacts out of Git by default because they can contain document text.

Run retrieval experiments from the repository root:

```powershell
python -m evaluation.run --split dev
python -m evaluation.run --generate --split dev
```

Run the test split only after settings are frozen.
