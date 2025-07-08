from codetext.parser import PythonParser
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
import logging
import os

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

def get_actual_solution(dp):
    root = parser.parse(bytes(dp["check"], "utf8"))
    root_node = root.root_node

    function_nodes = PythonParser.get_function_list(root_node)
    for function_node in function_nodes:
        entry_point = PythonParser.get_function_metadata(function_node, dp["check"])["identifier"]

        if entry_point == dp["entry_point"]:
            return function_node.text.decode()
    return None


def extract_file_context(file_path, function_signature_and_docstring, function_signature):
    """
    Extracts content from a file from the beginning until the end of a specified function signature and docstring.
    
    Args:
        file_path (str): Path to the file from which to extract context.
        function_signature_and_docstring (str): The function signature and docstring to find in the file.
            This should NOT include the function implementation.
        function_signature (str): The function signature to find in the file.
    
    Returns:
        str: The content from the beginning of the file up to the end of the function signature and docstring.
             Returns None if the function signature is not found in the file.
    """
    try:
        # Read the entire file
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
        
        # Find the position of the function signature in the file
        signature_position = file_content.find(function_signature)
        
        if signature_position == -1:
            logging.warning(f"Function signature not found in {file_path}")
            return None
        
        context = file_content[:signature_position]
        
        return context + function_signature_and_docstring, context
        
    except Exception as e:
        logging.error(f"Error extracting context from {file_path}: {e}")
        return None

def extract_dependencies(path_to_repo) -> str:
    """
    Reads and returns the contents of package.txt from the given repository path.
    
    Args:
        path_to_repo (str): Path to the repository directory
        
    Returns:
        str: Contents of package.txt file
        
    Raises:
        FileNotFoundError: If package.txt doesn't exist
        IOError: If there are issues reading the file
    """
    try:
        with open(os.path.join(path_to_repo, "package.txt"), 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"package.txt not found in {path_to_repo}")
    except IOError as e:
        raise IOError(f"Error reading package.txt: {str(e)}")