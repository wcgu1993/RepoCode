#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path

def parse_summary_line(line):
    """
    Extract (score, total_tasks) from a summary line like:
      "Pass@3: 0.06 (1/16 tasks passed)"
    """
    line = line.strip()
    m = re.match(
        r"Pass@\d+:\s*([\d.]+)\s*\(\d+/(\d+)\s*tasks passed\)",
        line
    )
    if not m:
        raise ValueError(f"Could not parse summary line: {line!r}")
    score = float(m.group(1))
    total = int(m.group(2))
    return score, total

def main():
    parser = argparse.ArgumentParser(
        description="Compute weighted average from pass_at_k.json files."
    )
    parser.add_argument(
        "--base_path",
        type=Path,
        help="Directory containing one subfolder per repository"
    )
    parser.add_argument(
        "--experiment",
        help="Name of the experiment subfolder inside each repo"
    )
    parser.add_argument(
        "--k",
        type=int,
        help="The 'k' value for pass_at_{k}.json"
    )
    args = parser.parse_args()

    total_samples = 0
    weighted_sum = 0.0

    for repo in sorted(args.base_path.iterdir()):
        if not repo.is_dir():
            continue

        json_path = repo / args.experiment / "pass_at_k" / f"pass_at_{args.k}.json"
        if not json_path.exists():
            print(f"Warning: {json_path} not found, skipping.", file=sys.stderr)
            continue

        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error reading JSON in {json_path}: {e}", file=sys.stderr)
            continue

        if not data:
            print(f"Warning: {json_path} is empty, skipping.", file=sys.stderr)
            continue

        # The last element holds the summary line
        try:
            score, num = parse_summary_line(data[-1])
        except ValueError as e:
            print(f"Error parsing summary in {json_path}: {e}", file=sys.stderr)
            continue

        print(f"{repo.name}: score={score:.4f}, samples={num}")
        total_samples += num
        weighted_sum += score * num

    if total_samples == 0:
        print("No valid samples found; cannot compute weighted average.", file=sys.stderr)
        sys.exit(1)

    weighted_avg = weighted_sum / total_samples
    print(f"\nOverall weighted average: {weighted_avg:.6f} "
          f"({total_samples} total samples)")

if __name__ == "__main__":
    main()