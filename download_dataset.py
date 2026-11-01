"""Stream 500 Natural Questions QA examples from Hugging Face Datasets."""

import json
import sys
from pathlib import Path


DATASET_ID = "sentence-transformers/natural-questions"
CONFIG_NAME = "pair"
EXAMPLE_COUNT = 500
OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "evaluation"
    / "natural_questions.json"
)


def main() -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "ERROR: Hugging Face Datasets is not installed in this Python environment.\n"
            "Install it in the project's .venv with:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install datasets",
            file=sys.stderr,
        )
        return 1

    print(f"Loading {DATASET_ID!r}, config {CONFIG_NAME!r} (streaming)...")
    try:
        dataset = load_dataset(
            DATASET_ID,
            CONFIG_NAME,
            split="train",
            streaming=True,
        )

        # Inspect the dataset metadata before accessing any row values. The
        # published `pair` config exposes `query` and `answer` text features.
        features = dataset.features
        feature_names = set(features.keys()) if features is not None else set()
        print(f"Detected dataset features: {', '.join(sorted(feature_names)) or '(unavailable)'}")
        if not {"query", "answer"}.issubset(feature_names):
            raise ValueError(
                "Unexpected dataset schema for config 'pair': expected 'query' "
                f"and 'answer' features, found {sorted(feature_names)}"
            )

        records = []
        for row in dataset.take(EXAMPLE_COUNT):
            question = row["query"]
            answer = row["answer"]
            if not isinstance(question, str) or not isinstance(answer, str):
                raise ValueError("A dataset row has a non-text query or answer.")
            records.append({"question": question, "answer": answer})
            if len(records) % 100 == 0:
                print(f"Read {len(records)} of {EXAMPLE_COUNT} examples...")

        if not records:
            raise RuntimeError("The selected training split returned no examples.")

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
            json.dump(records, output_file, indent=2, ensure_ascii=False)
            output_file.write("\n")

        print(f"Saved {len(records)} question-answer pairs to: {OUTPUT_PATH}")
        if len(records) < EXAMPLE_COUNT:
            print(f"The dataset returned only {len(records)} examples.")
        return 0
    except Exception as exc:
        message = str(exc)
        auth_hint = ""
        if any(term in message.lower() for term in ("gated", "401", "403", "unauthorized", "forbidden")):
            auth_hint = (
                " The Hugging Face repository may require authorization. Check its access "
                "terms and authenticate with `hf auth login` if access is granted. "
                "This script will not switch to another dataset."
            )
        print(
            f"ERROR: could not download/export {DATASET_ID!r}: "
            f"{type(exc).__name__}: {message}.{auth_hint}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
