import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from rag_insight.bootstrap import build
from rag_insight.config import Settings

from .metrics import evidence_recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/datasets/demo.jsonl"))
    parser.add_argument("--documents", type=Path, default=Path("data/sample_documents"))
    parser.add_argument("--experiments", type=Path, default=Path("configs/experiments.json"))
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--generate", action="store_true", help="Also grade, correct, and generate using the configured LLM")
    args = parser.parse_args()
    dataset = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    dataset = [item for item in dataset if args.split == "all" or item["split"] == args.split]
    if not dataset:
        parser.error("Selected split is empty")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    root = Path("artifacts/runs") / stamp
    root.mkdir(parents=True)
    summary = {}
    for name, config in json.loads(args.experiments.read_text()).items():
        settings = Settings(**config)
        # Unique indexes avoid contamination from earlier runs or other strategies.
        pipeline = build(settings, Path("data/indexes") / stamp / f"{name}.sqlite")
        for path in sorted(args.documents.iterdir()):
            if path.suffix.lower() in {".md", ".txt", ".pdf"}:
                pipeline.ingest(path)
        records = []
        for item in dataset:
            context, trace = pipeline.retrieve(item["question"])
            record = {
                "id": item["id"], "question": item["question"],
                "answerable": item["answerable"],
                "expected_answer": item["expected_answer"],
                "initial_evidence_recall_at_context_k": evidence_recall([c.chunk for c in context], item["expected_evidence"]),
                "initial_trace": trace,
                "answer_correctness_manual": None,
                "citation_support_manual": None,
                "faithfulness_manual": None,
            }
            if args.generate:
                answer = pipeline.ask(item["question"])
                chunks = {c.chunk_id: c for c, _ in pipeline.store.all()}
                final_context = [chunks[key] for key in answer.trace[-1]["context_ids"]]
                record.update(
                    answer=asdict(answer),
                    final_evidence_recall_at_context_k=evidence_recall(final_context, item["expected_evidence"]),
                    abstention_correct=(not answer.sufficient) == (not item["answerable"]),
                )
            records.append(record)
        (root / f"{name}.json").write_text(json.dumps({"settings": asdict(settings), "records": records}, indent=2), encoding="utf-8")
        recalls = [r["initial_evidence_recall_at_context_k"] for r in records if r["initial_evidence_recall_at_context_k"] is not None]
        summary[name] = {"questions": len(records), "initial_evidence_recall_at_context_k": mean(recalls) if recalls else None}
        if args.generate:
            final_recalls = [r["final_evidence_recall_at_context_k"] for r in records if r["final_evidence_recall_at_context_k"] is not None]
            summary[name].update(
                final_evidence_recall_at_context_k=mean(final_recalls) if final_recalls else None,
                abstention_accuracy=mean(r["abstention_correct"] for r in records),
            )
    report = Path("artifacts/reports") / f"{stamp}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"split": args.split, "generation_enabled": args.generate, "results": summary}, indent=2), encoding="utf-8")
    print(f"Runs: {root}\nReport: {report}")


if __name__ == "__main__":
    main()
