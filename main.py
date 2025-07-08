from dotenv import load_dotenv
load_dotenv()

import argparse
import sys
import json
from datasets import load_dataset
from RepoLevelCodeGen import (
    RetrievalAugmentationConfig, 
    RetrievalAugmentation, 
    OpenAISummarizationModel,
    GeminiFlashSummarizationModel,
    OpenAIEmbeddingModel,
    GeminiEmbeddingModel,
    SBertEmbeddingModel, 
    ThreadSafeSBertEmbedding, 
    OpenAICodeDescriberModel,
    GeminiFlashCodeDescriberModel,
    PlainOpenAICodeGeneratorModel,
    OpenAICodeGeneratorModel,
    GeminiFlashCodeGeneratorModel,
    OpenAICodeRetrievalModel,
)
from RepoLevelCodeGen.utils import remove_function_from_tree, put_function_back_tree, get_nodes_by_func_names
import jsonlines
import os
from utils import get_actual_solution, extract_file_context, extract_dependencies
from retrieval_eval import evaluate_api_retrieval
from gt_api_extractor import extract_api_invocations, get_repo_functions

# Set up argument parser
parser = argparse.ArgumentParser(description='Repository-Level Code Generation Based on Tree-Structured Code Repository Information Retrieval with Large Language Models')
parser.add_argument('--precomputed', action='store_true', help='Whether to use precomputed tree')
parser.add_argument('--no-precomputed', action='store_false', dest='precomputed', help='Disable precomputed tree')
parser.add_argument('--true_invocations', action='store_true', help='Disables Retriever System and uses true invocations for baseline exp')
parser.add_argument('--dense_rag', action='store_true', help='Dense RAG mode evaluates only leaf nodes 5 by 5. For baseline evaluation.')
parser.add_argument('--repo_name', type=str, required=True, help='Repository name')
parser.add_argument('--experiment', type=str, default="", help='Experiment name, this is actually the tree name')
parser.add_argument('--clustering_method', type=str, default='HDBSCAN', help='Clustering method. GMM or HDBSCAN')
parser.add_argument('--code_generator_model', type=str, default='GPT4.1Mini', help='Code generator model type')
parser.add_argument('--code_retrieval_model', type=str, default='GPT4.1Mini', help='Code retrieval model type')
parser.add_argument('--code_describer_model', type=str, default='GPT4.1Mini', help='Code describer model type')
parser.add_argument('--summarization_model', type=str, default='GPT4.1Mini', help='Summarization model type')
parser.add_argument('--embedding_model', type=str, default='OpenAI', help='Embedding model type')
parser.add_argument('--tb_reduction_dimension', type=int, default=16, help='Tree builder reduction dimension')
parser.add_argument('--tb_max_node_in_cluster', type=int, default=8, help='Maximum nodes in cluster')
parser.add_argument('--n_samples', type=int, default=1, help='How many code generation samples to generate')

print("Received arguments:", sys.argv)
args = parser.parse_args()
print("Parsed arguments:", args)

if args.true_invocations and args.code_generator_model == "PlainGPT4.1Mini":
    raise ValueError("You can pick either oracle case or without retrieval case, not both.")

# Load and parse the dataset index lookup file
with open('dataset_index_lookup.json', 'r') as f:
    dataset_index = json.load(f)
# Extract information for the specified repo
repo_key = f"test-apps/{args.repo_name}"
if repo_key in dataset_index:
    repo_info = dataset_index[repo_key]
    start_line = repo_info['start_line']
    end_line = repo_info['end_line']
    line_count = repo_info['line_count']


# Model mapping dictionaries
code_generator_models = {
    'PlainGPT4.1Mini': PlainOpenAICodeGeneratorModel(),
    'GPT4.1Mini': OpenAICodeGeneratorModel(),
    'Gemini2.5Flash': GeminiFlashCodeGeneratorModel()
}

code_describer_models = {
    'GPT4.1Mini': OpenAICodeDescriberModel(),
    'Gemini2.5Flash': GeminiFlashCodeDescriberModel()
}

summarization_models = {
    'GPT4.1Mini': OpenAISummarizationModel(),
    'Gemini2.5Flash': GeminiFlashSummarizationModel()
}

# code_retrieval_models = {
#     'GPT4.1Mini': OpenAICodeRetrievalModel()
# }

embedding_models = {
    'OpenAI': OpenAIEmbeddingModel(),
    'Gemini': GeminiEmbeddingModel(),
    'SBert': SBertEmbeddingModel(),
    'ThreadSafeSBert': ThreadSafeSBertEmbedding()
}

dataset = load_dataset("Fsoft-AIC/RepoExec", split="full_context")

RAC = RetrievalAugmentationConfig(
    code_generator_model=code_generator_models[args.code_generator_model],
    code_describer_model=code_describer_models[args.code_describer_model],
    summarization_model=summarization_models[args.summarization_model],
    code_retrieval_model=args.code_retrieval_model,
    embedding_model=embedding_models[args.embedding_model],
    tb_reduction_dimension=args.tb_reduction_dimension,
    tb_max_node_in_cluster=args.tb_max_node_in_cluster
)

path_to_repo=f"./dataset/RepoExec/test-apps/{args.repo_name}"
path_to_functions = f"repo_func_descr/{args.repo_name}/{args.experiment}"
repo_functions = get_repo_functions(os.path.abspath(path_to_repo))
if args.true_invocations or args.code_generator_model == "PlainGPT4.1Mini" or args.dense_rag: # for baseline experiments. Oracle case or without retrieval
    print("default tree is loading")
    experiment = "_gpt_4_1_mini"   # If you do not have any tree first create one and change this name
    RA = RetrievalAugmentation(config=RAC, tree=f"demo/{args.repo_name}{experiment}")
elif args.precomputed:
    print("precomputed is true")
    experiment = ("_" + args.experiment) if args.experiment else args.experiment
    RA = RetrievalAugmentation(config=RAC, tree=f"demo/{args.repo_name}{experiment}")
else:
    print("precomputed is false")
    RA = RetrievalAugmentation(config=RAC)
    RA.add_projects(path_to_repo=path_to_repo, repo_name=args.repo_name, experiment=args.experiment, clustering_method=args.clustering_method)      # If you want to use precomputed functions, you need to add the path to the functions
    # RA.add_projects(path_to_functions=path_to_functions, clustering_method=args.clustering_method)
    experiment = ("_" + args.experiment) if args.experiment else args.experiment
    SAVE_PATH = f"demo/{args.repo_name}{experiment}"
    RA.save(SAVE_PATH)

repo_data = []
for sample in dataset:
    if sample["project"] == "test-apps/" + args.repo_name:
        repo_data.append(sample)

assert len(repo_data) == line_count

# Extract dependencies from the environment
dependencies = extract_dependencies(path_to_repo)

output_path = f"code_generation_output/{args.repo_name}/{args.experiment}"
os.makedirs(output_path, exist_ok=True)
retr_output_path = f"retrieval_results/{args.repo_name}/{args.experiment}"
os.makedirs(retr_output_path, exist_ok=True)
retrieval_results = []
precision_values = []  # List to store precision values
recall_values = []     # List to store recall values
f1_values = []        # List to store F1 scores
tp_values = []
fp_values = []
fn_values = []
context_size_values = []  # List to store context size values
all_nums_of_llm_invoke_retriever = []
counter = 0
for n_sample, sample in enumerate(repo_data):
    target_function_prompt = sample["target_function_prompt"]
    path_to_sample_function_module = os.path.abspath(f"{'/'.join(path_to_repo.split('/')[:])}/{'/'.join(sample['module'].split('.'))+'.py'}")
    file_content_with_target_function, file_content_without_target_function = extract_file_context(path_to_sample_function_module, sample["target_function_prompt"], sample["function_signature"])
    
    removed_nodes = remove_function_from_tree(RA.tree, sample["solution"])
    
    target_api_invocations = set(extract_api_invocations(sample["solution"], path_to_sample_function_module, repo_functions))
    
    # switch this to run only the ones which calls at least one API
    # if not target_api_invocations:
    #     put_function_back_tree(RA.tree, removed_nodes)
    #     continue
    if args.code_generator_model == "PlainGPT4.1Mini":
        context = []
    elif args.true_invocations:
        context = get_nodes_by_func_names(target_api_invocations, RA.tree, repo_functions)
    elif args.dense_rag:
        context, layer_information, num_of_llm_invoke_retriever = RA.retrieve(file_content_without_target_function, target_function_prompt, top_k=3, return_layer_information=True, rcg_retr=False, dense_rag=True)
        all_nums_of_llm_invoke_retriever.append(num_of_llm_invoke_retriever)
    else:
        context, layer_information, num_of_llm_invoke_retriever = RA.retrieve(file_content_without_target_function, target_function_prompt, top_k=3, return_layer_information=True)
        all_nums_of_llm_invoke_retriever.append(num_of_llm_invoke_retriever)
    
    # on the retrieval system evaluation, skipping the samples which does not invoke any API in the ground truth invocation. It skews the recall. But code generation task includes all the samples
    if target_api_invocations:
    
        # evaluate the retrieval method
        api_retrieval_result = evaluate_api_retrieval(context, target_api_invocations, repo_functions)
        print(f"API retrieval result: {api_retrieval_result}")
        
        retrieval_results.append({
            "function": sample["entry_point"],
            "target_function_prompt": sample["target_function_prompt"],
            "f1_score": api_retrieval_result["f1_score"],
            "recall": api_retrieval_result["recall"],
            "precision": api_retrieval_result["precision"],
            "context_size": api_retrieval_result["context_size"],
            "target_api_invocations": list(target_api_invocations),
            "context": [{"fname": c['function'].split('def ')[1].split('(')[0].strip() if c['function'] is not None else "None", 
                        "evidence": c.get('evidence', "No evidence provided")} for c in context]
        })
        
        # Save each retrieval result as it is created (append mode)
        with jsonlines.open(f"{retr_output_path}/retrieval_results.json", mode='a') as writer:
            writer.write(retrieval_results[-1])
        
        # Store precision, recall and F1 values
        precision_values.append(api_retrieval_result["precision"])
        recall_values.append(api_retrieval_result["recall"])
        f1_values.append(api_retrieval_result["f1_score"])
        tp_values.append(api_retrieval_result["tp"])
        fp_values.append(api_retrieval_result["fp"])
        fn_values.append(api_retrieval_result["fn"])
        context_size_values.append(api_retrieval_result["context_size"])
    
    # creating test instance for processed generations
    if sample["solution"] not in sample["check"]:
        actual_solution = get_actual_solution(sample)
    else:
        actual_solution = sample["solution"]
    test_case = sample["check"]
    assert actual_solution in test_case
    
    all_predictions = []
    all_test = []
    for i in range(args.n_samples):
        answer = RA.generate_code(file_content_without_target_function, target_function_prompt, repo_functions, context, dependencies)
        all_predictions.append(answer)
        all_test.append(test_case.replace(actual_solution, answer))
    
    with jsonlines.open(f"{output_path}/processed_generations.json", mode='a') as writer:
        writer.write({
        "task_id": n_sample + start_line, 
        "project": sample["project"], 
        "module": sample["module"], 
        "predictions": all_predictions, 
        "test": all_test,
        # "context": context, 
        # "precision": api_retrieval_result["precision"] if target_api_invocations else 'No Ground Truth API', 
        # "recall": api_retrieval_result["recall"] if target_api_invocations else 'No Ground Truth API', 
        # "f1_score": api_retrieval_result["f1_score"] if target_api_invocations else 'No Ground Truth API',
        # "context_size": api_retrieval_result["context_size"] if target_api_invocations else 'No Ground Truth API'
        })
    
    put_function_back_tree(RA.tree, removed_nodes)

if not args.experiment:
    args.experiment = "no_experiment"

# After the loop, print average precision, recall and F1 score
if precision_values and recall_values and f1_values and context_size_values:
    # Prepare statistics dictionary
    statistics = {
        "macro_metrics": {
            "precision": float(f"{sum(precision_values) / len(precision_values):.3f}"),
            "recall": float(f"{sum(recall_values) / len(recall_values):.3f}"),
            "f1_score": float(f"{sum(f1_values) / len(f1_values):.3f}"),
            "context_size": float(f"{sum(context_size_values) / len(context_size_values):.3f}")
        },
        "micro_metrics": {
            "precision": 0.0 if sum(tp_values) + sum(fp_values) == 0 else float(f"{sum(tp_values) / (sum(tp_values) + sum(fp_values)):.3f}"),
            "recall": 0.0 if sum(tp_values) + sum(fn_values) == 0 else float(f"{sum(tp_values) / (sum(tp_values) + sum(fn_values)):.3f}"),
            "f1_score": 0.0 if (p_den := sum(tp_values) + sum(fp_values)) == 0 or (r_den := sum(tp_values) + sum(fn_values)) == 0 or (precision := sum(tp_values)/p_den) + (recall := sum(tp_values)/r_den) == 0 else float(f"{2 * precision * recall / (precision + recall):.3f}")
        },
        "sample_info": {
            "total_samples": len(precision_values),
            "nodes_in_topmost_layer": len(RA.tree.layer_to_nodes[len(RA.tree.layer_to_nodes) - 1])
        },
        "retriever_stats": {
            "invocations_per_sample": all_nums_of_llm_invoke_retriever,
            "total_invocations": sum(all_nums_of_llm_invoke_retriever)
        }
    }
    
    # Print statistics
    print(f"\nAggregate Metrics:")
    print(f"Average Precision: {statistics['macro_metrics']['precision']}")
    print(f"Average Recall: {statistics['macro_metrics']['recall']}")
    print(f"Average F1 Score: {statistics['macro_metrics']['f1_score']}")
    print(f"Average Context Size: {statistics['macro_metrics']['context_size']}")
    print(f"Total Samples: {statistics['sample_info']['total_samples']}")
    print(f"Number of nodes in topmost layer of tree: {statistics['sample_info']['nodes_in_topmost_layer']}")
    print(f"Number of retriever model invocations per sample: {statistics['retriever_stats']['invocations_per_sample']}")
    print(f"Sum of the retriever model invocations of all samples: {statistics['retriever_stats']['total_invocations']}")
    print("----Micro Averaging----")
    print(f"Micro Precision: {statistics['micro_metrics']['precision']}")
    print(f"Micro Recall: {statistics['micro_metrics']['recall']}")
    print(f"Micro F1 Score: {statistics['micro_metrics']['f1_score']}")
    
    # Create directory and write statistics to jsonl file
    retr_stats_path = f"retrieval_stats/{args.repo_name}/{args.experiment}"
    os.makedirs(retr_stats_path, exist_ok=True)
    with jsonlines.open(f"{retr_stats_path}/retrieval_stats.jsonl", mode='w') as writer:
        writer.write(statistics)
else:
    # Prepare statistics dictionary
    statistics = {
        "sample_info": {
            "nodes_in_topmost_layer": len(RA.tree.layer_to_nodes[len(RA.tree.layer_to_nodes) - 1])
        },
        "retriever_stats": {
            "invocations_per_sample": all_nums_of_llm_invoke_retriever,
            "total_invocations": sum(all_nums_of_llm_invoke_retriever)
        }
    }
    
    # Print statistics
    print(f"Number of nodes in topmost layer of tree: {statistics['sample_info']['nodes_in_topmost_layer']}")
    print(f"Number of retriever model invocations per sample: {statistics['retriever_stats']['invocations_per_sample']}")
    print(f"Sum of the retriever model invocations of all samples: {statistics['retriever_stats']['total_invocations']}")
    
    # Create directory and write statistics to jsonl file
    retr_stats_path = f"retrieval_stats/{args.repo_name}/{args.experiment}"
    os.makedirs(retr_stats_path, exist_ok=True)
    with jsonlines.open(f"{retr_stats_path}/retrieval_stats.jsonl", mode='w') as writer:
        writer.write(statistics)
