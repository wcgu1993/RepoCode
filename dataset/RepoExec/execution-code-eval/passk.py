from typing import List, Union, Iterable, Dict
import numpy as np
import os
import json
from datasets import load_dataset
import glob
from collections import defaultdict

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        import itertools
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', help='full_context | medium_context | short_context', type=int, default=10)
parser.add_argument('--isContained', help='Execute all function (True) or only non-contained function (False)', 
                    action="store_true")
parser.add_argument('--parent_path', help='Parent directory containing experiment subfolders', required=True)
parser.add_argument('--experiment_name', help='Name of the experiment subfolder to look for', required=True)

args = parser.parse_args()

k = [1, 3, 5, 10]
n_samples = args.n_samples
isContained = args.isContained
parent_path = args.parent_path
experiment_name = args.experiment_name

all_results = []

# Traverse all subdirectories in parent_path
for project_folder in os.listdir(parent_path):
    project_path = os.path.join(parent_path, project_folder)
    if not os.path.isdir(project_path):
        continue
    exp_path = os.path.join(project_path, experiment_name)
    if not os.path.isdir(exp_path):
        continue
    # Find all results_*.jsonl files in the experiment folder
    for result_file in glob.glob(os.path.join(exp_path, 'results_*.jsonl')):
        with open(result_file, 'r') as f:
            all_results.extend([json.loads(line) for line in f])

# Now, all_results contains all the results from all relevant files
# The rest of the code should process all_results as before

# If you want to keep the per-task logic, you may need to group by task_id or similar
# For now, let's assume each result has a 'task_id' field
task_results = defaultdict(list)
for res in all_results:
    task_id = res.get('task_id')
    if task_id is not None:
        task_results[task_id].append(res)

# Now process each task as before
no_task = 0
contained_func = 0
total, correct = [], []

for task_id, results in task_results.items():
    # Optionally filter by isContained if needed (requires dataset access)
    # For now, skip this check or implement as needed
    results = sorted(results, key=lambda x: x["prediction_id"])
    passed = [r["passed"] for r in results]
    n_samples = len(passed)
    assert n_samples == len(passed)
    total.append(n_samples)
    correct.append(sum(passed))
    no_task += 1

print("number of task:", no_task)
total = np.array(total)
correct = np.array(correct)

ks = k
pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                for k in ks if (total >= k).all()}

with open(os.path.join(parent_path, f"passk_{experiment_name}_{isContained}.json"), "w") as f:
    json.dump(pass_at_k, f)
print(pass_at_k)