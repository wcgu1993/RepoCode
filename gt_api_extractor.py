import ast
import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Set, Optional
import logging
import builtins

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

BUILTIN_NAMES = set(dir(builtins))

@dataclass
class RepoFunction:
    """Class to store information about a function or class in the repository."""
    name: str
    module_path: str
    lineno: int
    body: str
    is_class: bool = False
    
    def __hash__(self):
        return hash((self.name, self.module_path))
    

def get_repo_functions(repo_path: str) -> Dict[str, RepoFunction]:
    """
    Get all functions and classes defined in the repository, including class methods.
    """
    repo_functions = {}
    
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    tree = ast.parse(content, filename=file_path)
                except (SyntaxError, UnicodeDecodeError) as e:
                    logger.warning(f"Error parsing {file_path}: {e}")
                    continue
                
                class StackVisitor(ast.NodeVisitor):
                    def __init__(self):
                        self.current_class = None
                        
                    def visit_ClassDef(self, node):
                        rel_path = os.path.relpath(file_path, repo_path)
                        source = ast.get_source_segment(content, node) or ""
                        repo_functions[node.name] = RepoFunction(
                            name=node.name,
                            module_path=rel_path,
                            lineno=node.lineno,
                            body=source,
                            is_class=True
                        )
                        previous_class = self.current_class
                        self.current_class = node.name
                        self.generic_visit(node)
                        self.current_class = previous_class
                        
                    def visit_FunctionDef(self, node):
                        qualified_name = f"{self.current_class}.{node.name}" if self.current_class else node.name
                        rel_path = os.path.relpath(file_path, repo_path)
                        source = ast.get_source_segment(content, node) or ""
                        repo_functions[qualified_name] = RepoFunction(
                            name=qualified_name,
                            module_path=rel_path,
                            lineno=node.lineno,
                            body=source,
                            is_class=False
                        )
                        self.generic_visit(node)
                
                visitor = StackVisitor()
                visitor.visit(tree)
    
    logger.debug(f"Total functions/classes found in repository: {len(repo_functions)}")
    return repo_functions

    
def extract_api_invocations(
    function_str: str, 
    file_path: str, 
    repo_functions: Dict[str, RepoFunction]
) -> List[str]:
    """
    Extract repository-internal function calls, excluding recursive self-calls and constructor functions.
    """
    try:
        tree = ast.parse(function_str)
    except SyntaxError as e:
        logger.warning(f"Syntax error in function string: {e}")
        return []
    
    target_function_name = None
    target_function_node = None
    nested_function_names = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            target_function_name = node.name
            target_function_node = node
            break

    if target_function_node:
        for child in ast.walk(target_function_node):
            if isinstance(child, ast.FunctionDef) and child is not target_function_node:
                nested_function_names.add(child.name)

    function_calls: Set[str] = set()
    
    container_types = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue

        for target in targets:
            if isinstance(target, ast.Name):
                if isinstance(value, ast.DictComp) and isinstance(value.value, ast.Call):
                    call_node = value.value
                    if isinstance(call_node.func, ast.Name):
                        class_name = call_node.func.id
                        if class_name in repo_functions and repo_functions[class_name].is_class:
                            container_types[target.id] = class_name
                elif isinstance(value, ast.Call):
                    if isinstance(value.func, ast.Name):
                        class_name = value.func.id
                        if class_name in repo_functions and repo_functions[class_name].is_class:
                            container_types[target.id] = class_name
                    elif isinstance(value.func, ast.Attribute) and isinstance(value.func.value, ast.Name):
                        module_part = value.func.value.id
                        class_name = f"{module_part}.{value.func.attr}"
                        if class_name in repo_functions and repo_functions[class_name].is_class:
                            container_types[target.id] = class_name

    global_vars = {}
    imported_externals = set()
    standard_libraries = set(sys.builtin_module_names)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            module_tree = ast.parse(f.read(), filename=file_path)
    except Exception as e:
        logger.warning(f"Error parsing module {file_path}: {e}")
        module_tree = None

    if module_tree:
        for node in module_tree.body:
            if isinstance(node, ast.ImportFrom):
                module_name = node.module
                if node.level > 0: continue
                if module_name and module_name.split('.')[0] in standard_libraries:
                    imported_externals.update(a.asname or a.name for a in node.names)
                    continue
                is_external = not any(
                    repo_func.module_path.startswith(os.path.join(*module_name.split('.')))
                    for repo_func in repo_functions.values()
                ) if module_name else True
                if is_external:
                    imported_externals.update(a.asname or a.name for a in node.names)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in standard_libraries:
                        imported_externals.add(alias.asname or alias.name)
                        continue
                    expected_path = os.path.join(*alias.name.split('.'))
                    is_external = not any(
                        repo_func.module_path.startswith(expected_path)
                        for repo_func in repo_functions.values()
                    )
                    if is_external:
                        imported_externals.add(alias.asname or alias.name)

        for stmt in module_tree.body:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                value = getattr(stmt, 'value', None)
                if isinstance(value, ast.Call):
                    for target in targets:
                        if isinstance(target, ast.Name):
                            if isinstance(value.func, ast.Name):
                                class_name = value.func.id
                                if repo_functions.get(class_name, RepoFunction('', '', 0, '')).is_class:
                                    global_vars[target.id] = class_name
                            elif isinstance(value.func, ast.Attribute) and isinstance(value.func.value, ast.Name):
                                module_part = value.func.value.id
                                class_name = f"{module_part}.{value.func.attr}"
                                if repo_functions.get(class_name, RepoFunction('', '', 0, '')).is_class:
                                    global_vars[target.id] = class_name

    class DeepCallVisitor(ast.NodeVisitor):
        def __init__(self, exclude_name: Optional[str], imported_externals: Set[str], nested_functions: Set[str]):
            self.exclude_name = exclude_name
            self.imported_externals = imported_externals
            self.nested_functions = nested_functions
            self.standard_modules = set(sys.builtin_module_names)

        def visit_Call(self, node):
            logger.debug(f"\n{'='*50}\nProcessing Call node at line {node.lineno}")
            logger.debug(f"Call node code: {ast.unparse(node)}")

            self._process_call_func(node)
            self._process_call_arguments(node)

            self.generic_visit(node)

        def _process_call_func(self, node):
            if isinstance(node.func, ast.Name):
                called_name = node.func.id
                if (called_name in self.nested_functions or
                    called_name in self.imported_externals or
                    called_name == self.exclude_name or
                    called_name in BUILTIN_NAMES):
                    return

                if repo_func := repo_functions.get(called_name):
                    qual_name = f"{called_name}.__init__" if repo_func.is_class else called_name
                    if qual_name not in function_calls:
                        function_calls.add(qual_name)

            elif isinstance(node.func, ast.Attribute):
                attr = node.func
                method_name = attr.attr
                added = False
                if isinstance(attr.value, ast.Name):
                    var_name = attr.value.id
                    if var_name in global_vars:
                        qualified = f"{global_vars[var_name]}.{method_name}"
                        if qualified in repo_functions:
                            function_calls.add(qualified)
                            added = True
                    if not added and var_name in container_types:
                        qualified = f"{container_types[var_name]}.{method_name}"
                        if qualified in repo_functions:
                            function_calls.add(qualified)
                            added = True

                if not added and isinstance(attr.value, ast.Subscript):
                    subscript_value = attr.value.value
                    if isinstance(subscript_value, ast.Name) and subscript_value.id in container_types:
                        qualified = f"{container_types[subscript_value.id]}.{method_name}"
                        if qualified in repo_functions:
                            function_calls.add(qualified)
                            added = True

                if not added and isinstance(attr.value, ast.Name):
                    class_name = attr.value.id
                    if repo_functions.get(class_name, RepoFunction('', '', 0, '')).is_class:
                        qualified = f"{class_name}.{method_name}"
                        if qualified in repo_functions:
                            function_calls.add(qualified)
                            added = True

                if not added and isinstance(attr.value, ast.Call):
                    call_node = attr.value
                    if isinstance(call_node.func, (ast.Name, ast.Attribute)):
                        class_name = None
                        if isinstance(call_node.func, ast.Name):
                            class_name = call_node.func.id
                        elif isinstance(call_node.func, ast.Attribute):
                            if isinstance(call_node.func.value, ast.Name):
                                class_name = f"{call_node.func.value.id}.{call_node.func.attr}"
                        if class_name and repo_functions.get(class_name, RepoFunction('', '', 0, '')).is_class:
                            qualified = f"{class_name}.{method_name}"
                            if qualified in repo_functions:
                                function_calls.add(qualified)
                                added = True

                if not added and isinstance(attr.value, ast.Name):
                    module_name = attr.value.id
                    if module_name in self.imported_externals or module_name in self.standard_modules:
                        return

        def _process_call_arguments(self, node):
            for arg in node.args:
                self._process_function_reference(arg)
            for keyword in node.keywords:
                self._process_function_reference(keyword.value)

        def _process_function_reference(self, node):
            if isinstance(node, ast.Name):
                called_name = node.id
                if (called_name in self.nested_functions or
                    called_name in self.imported_externals or
                    called_name == self.exclude_name or
                    called_name in BUILTIN_NAMES):
                    return
                if called_name in repo_functions:
                    repo_func = repo_functions[called_name]
                    qual_name = f"{called_name}.__init__" if repo_func.is_class else called_name
                    if qual_name not in function_calls:
                        function_calls.add(qual_name)
            elif isinstance(node, ast.Attribute):
                parts = []
                current = node
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                    qualified_name = '.'.join(reversed(parts))
                    if (qualified_name in self.nested_functions or
                        qualified_name in self.imported_externals or
                        qualified_name == self.exclude_name or
                        qualified_name in BUILTIN_NAMES):
                        return
                    if qualified_name in repo_functions:
                        repo_func = repo_functions[qualified_name]
                        qual_name = f"{qualified_name}.__init__" if repo_func.is_class else qualified_name
                        if qual_name not in function_calls:
                            function_calls.add(qual_name)

    visitor = DeepCallVisitor(
        exclude_name=target_function_name,
        imported_externals=imported_externals,
        nested_functions=nested_function_names
    )
    visitor.visit(tree)
    
    # Filter out constructor functions from the results
    filtered_calls = [call for call in function_calls if not call.endswith('.__init__')]
    
    return filtered_calls