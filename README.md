# Towards Repository-Level Code Generation Based on Tree-Structured Code Repository Information Retrieval with Large Language Models

This project implements a hierarchical retrieval system that enhances function-level code generation by identifying and utilizing possible API invocations from the codebase. The system employs a tree-structured approach to organize and retrieve potential API calls that could be used in implementing target functions. The process involves:

1. Constructing a hierarchical tree representation of the codebase
2. Retrieving relevant API invocations for target function implementation
3. Generating code that incorporates the retrieved APIs

The system leverages Large Language Models (LLMs) at multiple stages:
- Tree construction: Building the hierarchical structure of the codebase
- API retrieval: Identifying relevant API calls for function implementation
- Code generation: Creating function implementations using retrieved APIs

## Setup

Please ensure Python 3.8+ is installed. Clone repository and install dependencies.


1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
   - Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Directory Structure

- `dataset/`: Contains the benchmark datasets and execution environment. The execution scripts are modified for adaptation to our task and needs
- `code_generation_output/`: Stores generated function implementations
- `results/`: Contains execution results and metrics
- `tree_views/`: Stores visualizations of hierarchical tree representations of codebases
- `retrieval_results/`: Contains retrieved API invocations
- `repo_func_descr/`: Contains function descriptions of extracted functions

## Usage

### 1. Tree Construction, API Retrieval, and Code Generation

Build the hierarchical tree, retrieve APIs for a repository, and generate the implementation of the target function:
```bash
python main.py --no-precomputed --repo_name python-string-utils --code_generator_model GPT4.1Mini --experiment exp --code_generation_output --clustering_method HDBSCAN --n_samples 5
```

### 2. Execution and Evaluation

Set up and run the benchmark environment using [RepoExec](https://github.com/FSoft-AI4Code/RepoExec):
```bash
cd dataset/RepoExec/execution-code-eval
python3 execute.py --subset full_context \
    --prediction_dir /path/to/code_generation_output/python-string-utils/exp \
    --execution_dir /path/to/results/python-string-utils/exp \
    --repo_name python-string-utils \
    --start_task_id 0 \
    --end_task_id 38
```

**Task ID Lookup Table**

Use the following table to determine the correct `start_task_id` and `end_task_id` for your target repository:

| Repository | Start Task ID | End Task ID | Task Count |
|------------|---------------|-------------|------------|
| python-string-utils | 0 | 38 | 39 |
| scrapy | 39 | 112 | 74 |
| cookiecutter | 113 | 119 | 7 |
| py-backwards | 120 | 124 | 5 |
| flutils | 125 | 145 | 21 |
| youtube-dl | 146 | 152 | 7 |
| python-semantic-release | 153 | 153 | 1 |
| pytutils | 154 | 164 | 11 |
| docstring_parser | 165 | 167 | 3 |
| fastapi | 168 | 170 | 3 |
| sanic | 171 | 191 | 21 |
| luigi | 192 | 222 | 31 |
| thonny | 223 | 224 | 2 |
| apimd | 225 | 240 | 16 |
| tornado | 241 | 291 | 51 |
| pyMonet | 292 | 299 | 8 |
| pypara | 300 | 325 | 26 |
| httpie | 326 | 329 | 4 |
| typesystem | 330 | 333 | 4 |
| flutes | 334 | 344 | 11 |
| dataclasses-json | 345 | 350 | 6 |
| black/src | 351 | 353 | 3 |
| PySnooper | 354 | 354 | 1 |

For detailed setup instructions and additional benchmark configurations, please refer to the [RepoExec GitHub repository](https://github.com/FSoft-AI4Code/RepoExec).

### 3. Metrics Calculation

Calculate pass@k metrics for all repositories:
```bash
python3 passk.py --parent_path /path/to/results --experiment_name exp --n_samples 5 --isContained
```

Replace `/path/to/results` with the actual path to your results directory and `exp` with your experiment name.

### 4. Visualization

Visualize the hierarchical tree structure of the codebase:
```bash
python visualising_tree.py --repo_name python-string-utils --experiment exp
```

## Notes

- Use `--precomputed` flag if you want to use precomputed tree structures instead of reconstructing them
- Results are organized by experiment name for easy comparison