import json
import argparse
from pathlib import Path

def analyze_results(results_dir: Path):
    if not results_dir.exists():
        print(f"Directory {results_dir} does not exist!")
        return
    
    passed_count = 0
    failed_count = 0
    
    for file_path in sorted(results_dir.glob("results_*.jsonl")):
        print(f"\nAnalyzing {file_path.name}:")
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        result = json.loads(line.strip())
                        task_id = result.get('task_id')
                        passed = result.get('passed')
                        print(f"Task {task_id}: passed = {passed}")
                        if passed:
                            passed_count += 1
                        else:
                            failed_count += 1
                    except json.JSONDecodeError:
                        print(f"Error parsing JSON line in {file_path.name}")
                        continue
        except Exception as e:
            print(f"Error reading file {file_path.name}: {str(e)}")
    
    total = passed_count + failed_count
    accuracy = (passed_count / total * 100) if total > 0 else 0
    
    print("\nSummary:")
    print(f"Number of passed tests: {passed_count}")
    print(f"Number of failed tests: {failed_count}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default="results/pypara",
                      help='Directory containing the results files')
    args = parser.parse_args()
    
    analyze_results(Path(args.results_dir))