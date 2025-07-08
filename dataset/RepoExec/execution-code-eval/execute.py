import os
import datasets
import subprocess
import argparse

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--subset', help='full_context | medium_context | short_context')
parser.add_argument('--prediction_dir', help='Directory containing model outputs with generation.json file',
                    default="./results/predictions/repo-codegen-full-context-v3/gpt-3.5")
parser.add_argument('--execution_dir', help='Directory to save execution results',
                    default="./results/execution_rs/repo-codegen-full-context-v3/gpt-3.5")
parser.add_argument('--start_task_id', type=int, help='Starting task ID (inclusive)', required=True)
parser.add_argument('--end_task_id', type=int, help='Ending task ID (inclusive)', required=True)

# Parse command-line arguments
args = parser.parse_args()

# Convert relative paths to absolute paths
pred_dir = os.path.abspath(args.prediction_dir)
save_dir = os.path.abspath(args.execution_dir)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load the dataset
data = datasets.load_dataset("Fsoft-AIC/RepoExec")
data = data[args.subset]

# Ensure the repo_dir is an absolute path
repo_dir = os.path.abspath(os.path.dirname(os.getcwd()))
print("Repository directory:", repo_dir)
print("Total tasks in subset:", len(data))

# Ensure the task ID range is valid
if args.start_task_id < 0 or args.end_task_id >= len(data):
    raise ValueError(f"Task ID range is out of bounds. Valid range is 0 to {len(data) - 1}")

if args.start_task_id > args.end_task_id:
    raise ValueError(f"Start task ID {args.start_task_id} cannot be greater than end task ID {args.end_task_id}")

# Execute tasks within the specified range
for task_id in range(args.start_task_id, args.end_task_id + 1):
    # Skip the task if the result file already exists
    if os.path.exists(os.path.join(save_dir, f"results_{task_id}.jsonl")):
        continue

    project = data[task_id]["project"]

    # Build Docker execution command
    cmd = [
        "sudo", "docker", "run", "--rm",
        "-v", f"{pred_dir}:/pred_dir:ro",
        "-v", f"{save_dir}:/rs_dir",
        "-v", f"{repo_dir}:/input:ro",
        "-v", f"{repo_dir}/data_with_test_case:/output:ro",
        "-v", f"{repo_dir}/{project}/:/package:ro",
        "codeeval-runner",
        "--task_id", str(task_id),
        "--problem_file", "/pred_dir/processed_generations.json",
        "--rs_dir", "/rs_dir",
        "--timeout", "120"
    ]

    # Print the command for debugging purposes
    print("Executing command:")
    print(' '.join(cmd))

    # Run the command
    subprocess.run(cmd)