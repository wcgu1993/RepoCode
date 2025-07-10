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

1. Clone the repository
```bash
git clone https://github.com/YusufEmreGenc/thesis_project.git
cd thesis_project
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
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
python main.py --no-precomputed --repo_name <repo_name> --experiment <experiment_name> --clustering_method HDBSCAN --n_samples 5
```
- **What happens:**  
  - Builds the hierarchical tree for the repository (unless `--precomputed` is used).
  - Retrieves relevant API invocations for each target function.
  - Generates code for each target function.
- **Outputs:**
  - `code_generation_output/<repo_name>/<experiment_name>/processed_generations.json`: Generated code for each function (updated during run).
  - `retrieval_results/<repo_name>/<experiment_name>/retrieval_results.json`: API retrieval results for each function (updated during run).
  - `retrieval_stats/<repo_name>/<experiment_name>/retrieval_stats.jsonl`: Aggregate retrieval statistics (written at the end).

**Arguments:**
- `--repo_name` (required): Name of the repository.
- `--experiment` (optional): Name for this experiment/run (used in output folder names).
- `--clustering_method` (optional): Clustering method for tree construction (`HDBSCAN` or `GMM`).
- `--n_samples` (optional): Number of code generations per function.
- `--precomputed`/`--no-precomputed`: Use an existing tree or build a new one.
- `--true_invocations`: Use oracle retrieval (ground-truth APIs) instead of the retrieval system.
- `--dense_rag`: Use dense RAG baseline instead of hierarchical retrieval.
- `--code_generator_model`: Choose code generation model (`GPT4.1Mini`, `PlainGPT4.1Mini`, etc.).
- `--code_retrieval_model`, `--code_describer_model`, `--summarization_model`, `--embedding_model`: Advanced model options (see main.py for defaults).

**Example: Gemini model, oracle case**
```bash
python main.py --true_invocations --repo_name python-string-utils --experiment gemini_oracle_exp --clustering_method HDBSCAN --code_generator_model Gemini2.5Flash --code_retrieval_model Gemini2.5Flash --code_describer_model Gemini2.5Flash --summarization_model Gemini2.5Flash --embedding_model Gemini --n_samples 5
```

#### Other Scenarios

- **Use precomputed tree only for code generation:**
  ```bash
  python main.py --precomputed --repo_name <repo_name> --experiment <experiment_name> --clustering_method HDBSCAN --n_samples 5
  ```
- **Oracle retrieval (ground-truth APIs):**
  ```bash
  python main.py --true_invocations --repo_name <repo_name> --experiment <experiment_name> --clustering_method HDBSCAN --n_samples 5
  ```
- **Plain model (no context from repository):**
  ```bash
  python main.py --repo_name <repo_name> --experiment <experiment_name> --code_generator_model PlainGPT4.1Mini --clustering_method HDBSCAN --n_samples 5
  ```
- **Dense RAG baseline:**
  ```bash
  python main.py --dense_rag --repo_name <repo_name> --experiment <experiment_name> --clustering_method HDBSCAN --n_samples 5
  ```

### 2. Execution and Evaluation

For the execution part, a new environment for RepoExec benchmark should be created. Set up the benchmark environment using [RepoExec](https://github.com/FSoft-AI4Code/RepoExec):
```bash
cd dataset/RepoExec
pip install -r requirements
cd execution-code-eval
python3 execute.py --subset full_context \
    --prediction_dir /path/to/code_generation_output/<repo_name>/<experiment_name> \
    --execution_dir /path/to/results/<repo_name>/<experiment_name> \
    --repo_name <repo_name> \
    --start_task_id <start_id> \
    --end_task_id <end_id>
```
- **Output:** Execution results are saved in the `results/` folder.

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

### 3. Metrics Calculation

Calculate pass@k metrics for all repositories:
```bash
python3 passk.py --parent_path /path/to/results --experiment_name <experiment_name> --n_samples 5 --isContained
```

### 4. Visualization

Visualize the hierarchical tree structure of the codebase:
```bash
python visualising_tree.py --repo_name <repo_name> --experiment <experiment_name>
```
- **Output:** Visualization is saved in `tree_views/<repo_name>/<experiment_name>/` (file format and naming depend on the implementation).
