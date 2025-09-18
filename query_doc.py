"""CLI for querying a processed document run directory."""
from __future__ import annotations
import argparse
from mlops.query.query_engine import RunArtifacts, answer_query


def main():
    """CLI entry point for querying a processed run.

    Exits with code 0 always; prints ranked sections and optional entity list.
    """
    parser = argparse.ArgumentParser(description="Query processed document artifacts")
    parser.add_argument("run_dir", help="Path to run directory under artifacts/")
    parser.add_argument("query", help="Query string")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    artifacts = RunArtifacts(args.run_dir)
    result = answer_query(artifacts, args.query, top_k=args.top_k)
    for ans in result["answers"]:
        print(f"[Section {ans['section_id']}] {ans['title']} (score={ans['score']:.2f})")
        print(ans['snippet'])
        print("-" * 60)
    if result.get("entities"):
        print("Top entities:")
        for ent, count in result['entities']:
            print(f"  {ent}: {count}")


if __name__ == "__main__":
    main()
