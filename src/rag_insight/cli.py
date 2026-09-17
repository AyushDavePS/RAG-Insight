import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .baseline import run_baseline, write_baseline
from .benchmarking import benchmark_embedding_batches
from .bootstrap import build
from .config import Settings
from .inspection import CharacterBudgetTokenizer, inspect_documents, write_inspection
from .sources import LocalHtmlAdapter


def main():
    parser = argparse.ArgumentParser(description="Ingest documents or query RAG Insight")
    parser.add_argument("command", choices=["ingest", "ingest-html", "ask", "inspect", "baseline", "benchmark-embeddings"])
    parser.add_argument("value", help="Document directory or quoted question")
    parser.add_argument("--index", type=Path, default=Path("data/indexes/default.sqlite"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--strategy", choices=["recursive", "structure", "hybrid"])
    parser.add_argument("--format", dest="output_format", choices=["json", "markdown"], default="json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict-tokenizer", action="store_true",
                        help="Fail rather than use the offline character-budget fallback.")
    parser.add_argument("--ocr", action="store_true", help="OCR scanned PDFs locally with CPU PP-OCRv5.")
    args = parser.parse_args()
    settings_data = json.loads(args.config.read_text())
    if args.strategy:
        settings_data["chunk_strategy"] = args.strategy
    if args.ocr:
        settings_data["ocr_enabled"] = True
    settings = Settings(**settings_data)
    if args.command == "inspect":
        if args.output is None:
            parser.error("inspect requires --output")
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
        except (ImportError, OSError, RuntimeError) as error:
            if args.strict_tokenizer:
                parser.error(f"Could not load tokenizer '{settings.embedding_model}': {error}")
            print(
                "Warning: could not load the embedding tokenizer; using a conservative "
                "character-budget fallback. Token counts are not model-token counts. "
                f"Cause: {error}"
            )
            tokenizer = CharacterBudgetTokenizer()
        paths = sorted(p for p in Path(args.value).iterdir() if p.suffix.lower() in {".md", ".txt", ".pdf"})
        if not paths:
            parser.error("No supported documents found")
        write_inspection(inspect_documents(paths, settings, tokenizer), args.output_format, args.output)
        return
    if args.command == "baseline":
        if args.output is None:
            parser.error("baseline requires --output")
        pipeline = build(settings, args.index)
        paths = sorted(p for p in Path(args.value).iterdir()
                       if p.suffix.lower() in {".md", ".txt", ".pdf"})
        if not paths:
            parser.error("No supported documents found")
        for path in paths:
            pipeline.ingest(path)
        write_baseline(run_baseline(pipeline), args.output)
        pipeline.store.close()
        return
    if args.command == "benchmark-embeddings":
        if args.output is None:
            parser.error("benchmark-embeddings requires --output")
        paths = sorted(p for p in Path(args.value).iterdir() if p.suffix.lower() in {".md", ".txt", ".pdf"})
        if not paths:
            parser.error("No supported documents found")
        result = benchmark_embedding_batches(settings, paths)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    pipeline = build(settings, args.index)
    if args.command == "ingest-html":
        for source in LocalHtmlAdapter().fetch(args.value):
            print(f"{source.title}: {pipeline.ingest_source(source)} chunks")
    elif args.command == "ingest":
        paths = sorted(p for p in Path(args.value).iterdir() if p.suffix.lower() in {".md", ".txt", ".pdf"})
        if not paths:
            parser.error("No supported documents found")
        for path in paths:
            print(f"{path.name}: {pipeline.ingest(path)} chunks")
    else:
        print(json.dumps(asdict(pipeline.ask(args.value)), indent=2))


if __name__ == "__main__":
    main()
