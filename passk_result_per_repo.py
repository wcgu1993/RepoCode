import json
import argparse
from pathlib import Path
import os

def pass_at_k(results_dir: Path, k: int = None):
    # List to collect output messages in order.
    output_lines = []
    
    # Check if the results directory exists.
    if not results_dir.exists():
        msg = f"Directory {results_dir} does not exist!"
        print(msg)
        output_lines.append(msg)
        return output_lines

    # Get all JSONL files in the directory
    json_files = sorted(
        [f for f in results_dir.glob("*.jsonl")],
        key=lambda p: p.name
    )
    
    if not json_files:
        msg = "No JSONL files found!"
        print(msg)
        output_lines.append(msg)
        return output_lines

    # To preserve the order of first occurrence for each unique sample (task_id).
    samples_order = []     # List of task_ids in the order first encountered.
    sample_results = {}    # Dictionary mapping task_id -> {"passed": bool, "run": str or None}.

    # Process each JSONL file
    for jf in json_files:
        try:
            with jf.open("r") as f:
                # Each line in the file represents a different run
                runs = [json.loads(line) for line in f]
        except Exception as e:
            msg = f"Error reading {jf}: {e}"
            print(msg)
            output_lines.append(msg)
            continue

        # Get task_id from the first run (all runs in a file are for the same task)
        if not runs:
            continue
            
        task_id = runs[0].get("task_id")
        if task_id is None:
            continue

        # Record task_id order if this is the first time seen
        if task_id not in sample_results:
            samples_order.append(task_id)
            sample_results[task_id] = {"passed": False, "run": None}

        # If k is specified, only consider the first k runs
        runs_to_check = runs[:k] if k is not None else runs

        # Check each run for this task
        for run_idx, run_data in enumerate(runs_to_check):
            # If this sample hasn't been marked as passed and it passes in this run, record the run
            if not sample_results[task_id]["passed"] and run_data.get("passed") is True:
                run_info = f"{jf.name}/run_{run_idx + 1}"
                sample_results[task_id] = {"passed": True, "run": run_info}
                msg = f"Task {task_id} passed on run {run_info}"
                print(msg)
                output_lines.append(msg)
                break  # No need to check more runs for this task
    
    # For each sample in the order they were first encountered, if not passed, output a failure message
    for task_id in samples_order:
        if not sample_results[task_id]["passed"]:
            msg = f"Task {task_id} cannot pass"
            print(msg)
            output_lines.append(msg)
    
    # Calculate the pass rate
    total_tasks = len(samples_order)
    passed_count = sum(1 for res in sample_results.values() if res["passed"])
    pass_rate = passed_count / total_tasks if total_tasks > 0 else 0.0
    summary = f"\nPass@{k}: {pass_rate:.2f} ({passed_count}/{total_tasks} tasks passed)"
    
    print(summary)
    output_lines.append(summary)
    
    return output_lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default="results/pypara/final_exp",
                        help='Directory containing the result JSONL files')
    parser.add_argument('--k', type=int, default=None,
                        help='Number of runs to consider for pass@k. If not given, all runs are used.')
    args = parser.parse_args()
    
    # Generate output lines
    output_lines = pass_at_k(Path(args.results_dir), args.k)
    
    # Create pass_at_k directory if it doesn't exist
    output_dir = Path(args.results_dir) / "pass_at_k"
    output_dir.mkdir(exist_ok=True)
    
    # Write the output lines to a JSON file (overwriting if it exists)
    output_file = output_dir / f"pass_at_{args.k}.json"
    with open(output_file, "w") as outfile:
        json.dump(output_lines, outfile, indent=2)
    
    final_msg = f"\nThe logs have been written to {output_file}"
    print(final_msg)
    output_lines.append(final_msg)