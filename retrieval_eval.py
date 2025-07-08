import ast
import logging
from typing import Dict, List, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def normalize_code(code: str) -> str:
    """Normalize code by parsing and unparsing to remove formatting differences."""
    try:
        return ast.unparse(ast.parse(code)).strip()
    except Exception:
        return code.strip().replace(' ', '').replace('\n', '')

def evaluate_api_retrieval(
    context: List[Dict],
    target_api_invocations: Set,
    repo_functions: Dict
) -> Dict[str, float]:
    
    context_functions = set()
    for item in context:
        if 'function' in item and isinstance(item['function'], str):
            try:
                func_tree = ast.parse(item['function'])
                for node in ast.walk(func_tree):
                    if isinstance(node, ast.FunctionDef):
                        context_body = ast.get_source_segment(item['function'], node) or ""
                        normalized_context = normalize_code(context_body)
                        for repo_func in repo_functions.values():
                            if normalize_code(repo_func.body) == normalized_context:
                                context_functions.add(repo_func.name)
                                break
            except SyntaxError:
                lines = item['function'].strip().split('\n')
                if lines and lines[0].startswith('def '):
                    context_functions.add(lines[0][4:].split('(')[0].strip())
    if not context_functions:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "context_size": len(context), "tp": 0, "fp": 0, "fn": len(target_api_invocations - context_functions)}
    
    true_positives = len(target_api_invocations & context_functions)
    false_positives = len(context_functions - target_api_invocations)
    false_negatives = len(target_api_invocations - context_functions)
    precision = true_positives / len(context_functions) if context_functions else 0.0
    recall = true_positives / len(target_api_invocations) if target_api_invocations else 0.0
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision, 
        "recall": recall, 
        "f1_score": f1_score,
        "context_size": len(context),
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives
    }