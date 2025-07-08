from typing import List, Tuple, Dict, Optional
import json
import ast
import re
import logging
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_random_exponential

import os
import google.generativeai as genai

from .model_utils import gemini_wait

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenAICodeRetrievalModel:
    """
    This model uses an LLM (GPT-4 or GPT-4o mini) to evaluate candidate code snippets or descriptions 
    for relevance to a target function. It constructs a structured prompt including the target function code, 
    file context, and candidates, and parses the LLM's JSON output into a list of relevant candidates.
    """
    def __init__(self, target_function_code: str, file_context: str):
        """
        Initialize the code retrieval model.
        
        :param target_function_code: Source code of the target function.
        :param file_context: Additional file context around the target function (for understanding usage/environment).
        """
        self.target_function_code = target_function_code
        self.file_context = file_context
        self.llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=1.0)
        # self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=1.0)

    def format_candidates(self, candidates: List[Dict[str, Optional[str]]]) -> str:
        """
        Formats the batch of candidates into a string that is ready for the prompt.
        Each candidate is numbered starting from 1.
        """
        formatted_candidates = ""
        for idx, cand in enumerate(candidates, start=1):
            desc = cand.get("description", "") or ""
            impl = cand.get("implementation", None)
            if impl:
                formatted_candidates += (
                    f"{idx}. (Function Implementation)\n"
                    f"Description: {desc}\n"
                    f"Implementation:\n```python\n{impl}\n```\n"
                )
            else:
                formatted_candidates += (
                    f"{idx}. (Summary Description)\n"
                    f"Description: {desc}\n\n"
                )
        return formatted_candidates

    def _build_prompt(self, candidates: List[Dict[str, Optional[str]]]) -> Tuple[str, str]:
        """
        Construct the system and user messages for the prompt to the LLM.
        """

        system_instructions = (
            "You are an expert at analyzing and identifying potential API invocations of a target function "
            "based on function definitions or function group summaries from the same repository. "
            "Follow the chain-of-thought techniques provided in the examples and detail your reasoning step-by-step "
            "before arriving at your final decision for each candidate."
        )

        user_content = f"""Primary Task:
- Identify which functions from the codebase could be INVOKED/CALLED during the implementation of target function.

Input Format:
1. File Content: The file content where the target function is defined. The content is from the beginning of the file until the target function.
2. Target Function: Function signature with docstring.
3. Nodes: A numbered list of nodes, which can either be:
    - **Description Node**: Contains a natural language description and full Python implementation.
    - **Summary Node**: Contains only a natural language summary of multiple functions.

Important Notes:
- When evaluating the Target Function, **EMPHASIZE** the docstring explanation if it given.
- The nodes are hierarchical; summary nodes may refer to multiple lower-level functions.
- Each node should be evaluated individually.
- The File Content is only for context (e.g., naming patterns, implementation style). Do not use it to justify inclusion or exclusion.
- Focus SOLELY on invocation potential, not implementation similarity
- A function may be invoked for various reasons: subtasks, validation, formatting, logging, etc.
- When provided, Analyze the implementation of Description nodes.

Evaluation of Nodes:
- For Summary nodes, consider all functions such a summary could include.
- For Description nodes, examine the actual implementation to evaluate invocation potential.
- Give clear evidence to support your decision.

Output Format:
- Construct your final output **strictly as a JSON list**.
- You **MUST** delimit the JSON list with ```json```.
- Each element must correspond to a node (same order as given).
- The output must have exactly {len(candidates)} elements
- Each element must include:
- "decision": true or false (whether the node might be invoked),
- "reason": a clear and specific justification.

### Examples:

Example:
File Content: ```python
import logging
from typing import List, Dict, Optional, Union, Any

# Utility functions for mathematical operations
def add(a: int, b: int) -> int:
    \"""Add two integers and return the result.\"""
    return a + b

def subtract(a: int, b: int) -> int:
    \"""Subtract b from a and return the result.\"""
    return a - b

# String manipulation utilities
def format_message(template: str, **kwargs) -> str:
    \"""Format a message using the provided template and keyword arguments.\"""
    return template.format(**kwargs)

# Logging configuration
logger = logging.getLogger(__name__)

class MathOperations:
    \"""Class containing various mathematical operations.\"""
    
    @staticmethod
    def multiply(a: int, b: int) -> int:
        \"""Multiply two integers and return the result.\"""
        return a * b
```

Target Function: ```python
def compare_values(a: int, b: int) -> str:
    \"""Compare two integers and return a string describing their relationship (e.g., 'equal', 'greater', 'less').\"""
```

Nodes:
Node_1. Summary: "Functions for basic arithmetic operations like addition and subtraction."
Node_2. Description: "Utility functions for string formatting and manipulation."
Implementation: 
```python
def format_string(template: str, *args, **kwargs) -> str:
    \"""Format a string using the provided template and arguments.\"""
    return template.format(*args, **kwargs)
```
Node_3. Description: "String concatenation utility that joins strings with a specified separator."
Implementation: 
```python
def join_strings(strings: List[str], separator: str = " ") -> str:
    \"""Join a list of strings using the provided separator.\"""
    return separator.join(strings)
```
Node_4. Summary: "Comparison functions for various data types, including integers."
Node_5. Summary: "General-purpose logging and debugging utilities."

### Step 1: Analyze the File Content
The file content provides context about the codebase:
- It imports logging and various typing modules
- It contains basic arithmetic functions (add, subtract)
- It has string formatting utilities (format_message)
- It sets up a logger
- It has a MathOperations class with a multiply method
- The codebase appears to have both standalone functions and class methods
- Some functions use type hints and docstrings

### Step 2: Analyze the Target Function
The target function compare_values(a: int, b: int) takes two integers as input and returns a string describing their relationship (e.g., "equal", "greater", "less"). Possible implementations might involve:
- Comparing a and b using operators (==, >, <).
- Constructing a string based on the comparison result.
- Potentially validating inputs or formatting the output string.

API Invocation Needs:
- Comparison utilities (if not using built-in operators).
- String manipulation or formatting functions to construct the return value.
- No arithmetic operations seem necessary beyond comparison.
- Debugging or logging might be used if part of the implementation.

IMPORTANT: While the file content shows the repository context and patterns, our API invocation decisions must be based ONLY on the target function's needs, not what's available in the file content.

### Step 3: Analyze Each Node and Assess Invocation Potential

Node_1:
Service/Capability: Provides functions for arithmetic operations (e.g., add, subtract).
Invocation Potential: The target function focuses on comparison, not arithmetic. Adding or subtracting a and b is not required to determine their relationship.
Decision: Excluded. Arithmetic operations are not needed for comparing and describing relationships between values.
EVIDENCE: I don't see a scenario where arithmetic operations would be needed for a function that simply compares values and returns a descriptive string. Adding or subtracting the values isn't necessary to determine their relationship or to format the output string.

Node_2:
Service/Capability: Provides a function for string formatting similar to Python's built-in format method.
Implementation Analysis: The function implementation reveals several important details:
1. It accepts a template string with placeholders
2. It can handle both positional and keyword arguments 
3. It leverages Python's built-in format method for the actual formatting
4. It returns the formatted result as a string

Looking at this API design, it's specifically built for formatting strings with placeholders, which is exactly what would be needed for constructing comparison result messages.

Invocation Potential: The target function must return a string describing the relationship between two integers. It could use format_string to create this output message rather than using string concatenation or f-strings.
Decision: Included.
EVIDENCE: The target function compare_values must return a descriptive string about the relationship between integers. Analyzing the implementation, format_string provides exactly what's needed because:
1. It handles template strings with placeholders, ideal for the comparison output format
2. It accepts keyword arguments, allowing clear parameter naming (a=5, relation="greater", b=3)
3. The actual formatting is delegated to a mature method (string.format)
4. It returns exactly what compare_values needs to return - a formatted string

For example, compare_values could invoke it as format_string("{{a}} is {{relation}} than {{b}}", a=5, relation="greater", b=3) to produce "5 is greater than 3". This API is perfectly aligned with the target function's need to generate human-readable comparison results with variable content based on the actual comparison outcome.

Node_3:
Service/Capability: Provides a function to join multiple strings with a separator.
Implementation Analysis: The function implementation reveals several important details:
1. It requires a list of strings as its primary input parameter
2. It takes an optional separator parameter (defaulting to a space)
3. It simply delegates to Python's built-in string join method
4. It returns a single concatenated string

Looking at this API design, it's specifically built for combining multiple independent strings with a consistent separator character.

Invocation Potential: The target function only needs to return a single formatted string describing the relationship. It doesn't need to join multiple strings together to form its output.
Decision: Excluded.
EVIDENCE: When analyzing the implementation against the needs of compare_values:
1. The join_strings function requires an iterable of strings as input
2. A comparison result is naturally a single formatted message, not multiple discrete strings
3. The compare_values function has no clear need to join multiple strings with a separator

While technically it could be forced to work (e.g., join_strings([str(a), "is greater than", str(b)])), this would be an awkward and inefficient use of the API. The implementation is designed for joining collections of strings, but compare_values generates a single message with embedded values. There's no scenario where decomposing the comparison message into multiple strings for joining would be the natural or efficient approach.

Node_4:
Service/Capability: Encompasses various comparison functions for different data types.
Invocation Potential: The core task of compare_values is comparing integers. While Python built-in operators could suffice, the target function might leverage specialized comparison functions from this group for consistency, additional logic, or handling edge cases.
Decision: Included.
EVIDENCE: While the target function could implement comparison using built-in operators (>, <, ==), it's reasonable that it might leverage specialized comparison functions from this group for consistency with codebase patterns, handling edge cases, or advanced comparison logic. Since the summary indicates functions specifically designed for comparisons, there's a reasonable possibility the target function might invoke one of these functions.

Node_5:
Service/Capability: Provides logging and debugging utilities.
Invocation Potential: While not essential for the core comparison logic, the target function might use logging functions to track execution, debug issues, or record comparison results, especially in a production environment.
Decision: Included.
EVIDENCE: While not strictly necessary for the core comparison functionality, proper software engineering practice often incorporates logging. Being lenient, I can reasonably imagine the target function using logging to track inputs, outputs, or comparison results, especially in a production system with comprehensive logging requirements.

### Step 4: Compile the Final Output
```json
[
  {{"decision": false, "reason": "Arithmetic functions are unrelated to the goal of comparing and describing relationships between values."}},
  {{"decision": true, "reason": "The function formats output strings, which aligns with the target function's need to return descriptive messages."}},
  {{"decision": false, "reason": "Joining multiple strings is unnecessary for returning a single descriptive comparison message."}},
  {{"decision": true, "reason": "This node includes comparison utilities, which might be used instead of built-in operators for consistency or special logic."}},
  {{"decision": true, "reason": "Logging may be used for debugging or tracing behavior in the comparison function, especially in production environments."}}
]
```

### Input:
File Context:
```python
{self.file_context}
```

Target Function Code:
```python
{self.target_function_code}
```

Nodes:
{self.format_candidates(candidates)}

Analysis:"""
       
        return system_instructions, user_content

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _get_relevant_response(self, candidates: List[Dict[str, Optional[str]]]) -> Tuple[List[int], List[str]]:
        """
        Calls the LLM and parses the response. This method is decorated with Tenacity for automatic retries.
        """
        system_msg, user_msg = self._build_prompt(candidates)
        messages = [SystemMessage(content=system_msg), HumanMessage(content=user_msg)]
        result = self.llm.invoke(messages)
        response_text = ""
        if hasattr(result, "content"):
            response_text = result.content
        elif isinstance(result, str):
            response_text = result
        else:
            try:
                response_text = result.generations[0][0].text
            except Exception:
                response_text = str(result)
        return self._parse_response(response_text, num_candidates=len(candidates))

    def _parse_response(self, response_text: str, num_candidates: int) -> Tuple[List[int], List[str]]:
        """
        Parse the LLM response, extracting JSON and returning selected indices and reasons.
        """
        # First try to find JSON in markdown code blocks
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # If no markdown block found, try to find a JSON array directly
            match = re.search(r"\[\s*\{.*?\}\s*\]", response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                # fallback: try to find the first list-like structure
                match = re.search(r"($begin:math:display$\\s*{.*?}\\s*$end:math:display$)", response_text, re.DOTALL)
                if match:
                    json_str = match.group(1)
                else:
                    print(response_text)
                    raise ValueError("Failed to find a JSON list in the response.")

        try:
            data = json.loads(json_str)
        except Exception as e1:
            try:
                data = ast.literal_eval(json_str)
            except Exception as e2:
                raise ValueError(f"Failed to parse model response as JSON.\nExtracted: {json_str}") from e2

        if not isinstance(data, list):
            raise ValueError(f"Model response is not a JSON list: {data}")
        if len(data) != num_candidates:
            raise ValueError(f"Expected {num_candidates} items, but got {len(data)}")

        selected_indices: List[int] = []
        selected_reasons: List[str] = []

        for i, item in enumerate(data):
            if not isinstance(item, dict) or "decision" not in item or "reason" not in item:
                raise ValueError(f"Item {i} is malformed: {item}")
            if isinstance(item["decision"], str):
                decision = item["decision"].strip().lower() == "true"
            else:
                decision = bool(item["decision"])
            if decision:
                selected_indices.append(i)
                selected_reasons.append(str(item["reason"]))

        return selected_indices, selected_reasons

    def is_relevant(self, candidates: List[Dict[str, Optional[str]]]) -> Tuple[List[int], List[str]]:
        """
        Determine which candidates are relevant to the target function by leveraging the Tenacity retry mechanism.
        """
        return self._get_relevant_response(candidates)


class GeminiFlashCodeRetrievalModel:
    """
    This model uses an LLM (GPT-4 or GPT-4o mini) to evaluate candidate code snippets or descriptions 
    for relevance to a target function. It constructs a structured prompt including the target function code, 
    file context, and candidates, and parses the LLM's JSON output into a list of relevant candidates.
    """
    def __init__(self, target_function_code: str, file_context: str):
        """
        Initialize the code retrieval model.
        
        :param target_function_code: Source code of the target function.
        :param file_context: Additional file context around the target function (for understanding usage/environment).
        """
        self.target_function_code = target_function_code
        self.file_context = file_context
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        model_name = "models/gemini-2.5-flash-preview-05-20"
        genai.configure(api_key=api_key)
        
        # Add safety settings to prevent blocking
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]
        
        self.gemini = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=(
                "You are an expert at analyzing and identifying potential API invocations of a target function based on function definitions or function group summaries from the same repository. Follow the chain-of-thought techniques provided in the examples and detail your reasoning step-by-step before arriving at your final decision for each candidate."
            ),
            safety_settings=safety_settings
        )

    def format_candidates(self, candidates: List[Dict[str, Optional[str]]]) -> str:
        """
        Formats the batch of candidates into a string that is ready for the prompt.
        Each candidate is numbered starting from 1.
        """
        formatted_candidates = ""
        for idx, cand in enumerate(candidates, start=1):
            desc = cand.get("description", "") or ""
            impl = cand.get("implementation", None)
            if impl:
                formatted_candidates += (
                    f"{idx}. (Function Implementation)\n"
                    f"Description: {desc}\n"
                    f"Implementation:\n```python\n{impl}\n```\n"
                )
            else:
                formatted_candidates += (
                    f"{idx}. (Summary Description)\n"
                    f"Description: {desc}\n\n"
                )
        return formatted_candidates

    def _build_prompt(self, candidates: List[Dict[str, Optional[str]]]) -> Tuple[str, str]:
        """
        Construct the system and user messages for the prompt to the LLM.
        """

        user_content = f"""Primary Task:
- Identify which functions from the codebase could be INVOKED/CALLED during the implementation of target function.

Input Format:
1. File Content: The file content where the target function is defined. The content is from the beginning of the file until the target function.
2. Target Function: Function signature with docstring.
3. Nodes: A numbered list of nodes, which can either be:
    - **Description Node**: Contains a natural language description and full Python implementation.
    - **Summary Node**: Contains only a natural language summary of multiple functions.

Important Notes:
- When evaluating the Target Function, **EMPHASIZE** the docstring explanation if it given.
- The nodes are hierarchical; summary nodes may refer to multiple lower-level functions.
- Each node should be evaluated individually.
- The File Content is only for context (e.g., naming patterns, implementation style). Do not use it to justify inclusion or exclusion.
- Focus SOLELY on invocation potential, not implementation similarity
- A function may be invoked for various reasons: subtasks, validation, formatting, logging, etc.
- When provided, Analyze the implementation of Description nodes.

Evaluation of Nodes:
- For Summary nodes, consider all functions such a summary could include.
- For Description nodes, examine the actual implementation to evaluate invocation potential.
- Give clear evidence to support your decision.

Output Format:
- Your response MUST be a valid JSON list **containing EXACTLY {len(candidates)} elements**
- Each element MUST be a JSON object with **EXACTLY** these two keys:
  - "decision": boolean (true or false)
  - "reason": string (a clear and specific justification)
- DO NOT add any additional fields or keys
- The JSON must be properly formatted with:
  - Single curly braces {{}} for objects
  - Double quotes " for strings
  - No trailing commas
  - No comments
  - No extra whitespace
- The response MUST be delimited with ```json``` markers

Example of correct JSON format:
```json
[
  {{"decision": false, "reason": "Arithmetic functions are unrelated to the goal of comparing and describing relationships between values."}},
  {{"decision": true, "reason": "The function formats output strings, which aligns with the target function's need to return descriptive messages."}}
]
```

### Examples:

Example:
File Content: ```python
import logging
from typing import List, Dict, Optional, Union, Any

# Utility functions for mathematical operations
def add(a: int, b: int) -> int:
    \"""Add two integers and return the result.\"""
    return a + b

def subtract(a: int, b: int) -> int:
    \"""Subtract b from a and return the result.\"""
    return a - b

# String manipulation utilities
def format_message(template: str, **kwargs) -> str:
    \"""Format a message using the provided template and keyword arguments.\"""
    return template.format(**kwargs)

# Logging configuration
logger = logging.getLogger(__name__)

class MathOperations:
    \"""Class containing various mathematical operations.\"""
    
    @staticmethod
    def multiply(a: int, b: int) -> int:
        \"""Multiply two integers and return the result.\"""
        return a * b
```

Target Function: ```python
def compare_values(a: int, b: int) -> str:
    \"""Compare two integers and return a string describing their relationship (e.g., 'equal', 'greater', 'less').\"""
```

Nodes:
Node_1. Summary: "Functions for basic arithmetic operations like addition and subtraction."
Node_2. Description: "Utility functions for string formatting and manipulation."
Implementation: 
```python
def format_string(template: str, *args, **kwargs) -> str:
    \"""Format a string using the provided template and arguments.\"""
    return template.format(*args, **kwargs)
```
Node_3. Description: "String concatenation utility that joins strings with a specified separator."
Implementation: 
```python
def join_strings(strings: List[str], separator: str = " ") -> str:
    \"""Join a list of strings using the provided separator.\"""
    return separator.join(strings)
```
Node_4. Summary: "Comparison functions for various data types, including integers."
Node_5. Summary: "General-purpose logging and debugging utilities."

### Step 1: Analyze the File Content
The file content provides context about the codebase:
- It imports logging and various typing modules
- It contains basic arithmetic functions (add, subtract)
- It has string formatting utilities (format_message)
- It sets up a logger
- It has a MathOperations class with a multiply method
- The codebase appears to have both standalone functions and class methods
- Some functions use type hints and docstrings

### Step 2: Analyze the Target Function
The target function compare_values(a: int, b: int) takes two integers as input and returns a string describing their relationship (e.g., "equal", "greater", "less"). Possible implementations might involve:
- Comparing a and b using operators (==, >, <).
- Constructing a string based on the comparison result.
- Potentially validating inputs or formatting the output string.

API Invocation Needs:
- Comparison utilities (if not using built-in operators).
- String manipulation or formatting functions to construct the return value.
- No arithmetic operations seem necessary beyond comparison.
- Debugging or logging might be used if part of the implementation.

IMPORTANT: While the file content shows the repository context and patterns, our API invocation decisions must be based ONLY on the target function's needs, not what's available in the file content.

### Step 3: Analyze Each Node and Assess Invocation Potential

Node_1:
Service/Capability: Provides functions for arithmetic operations (e.g., add, subtract).
Invocation Potential: The target function focuses on comparison, not arithmetic. Adding or subtracting a and b is not required to determine their relationship.
Decision: Excluded. Arithmetic operations are not needed for comparing and describing relationships between values.
EVIDENCE: I don't see a scenario where arithmetic operations would be needed for a function that simply compares values and returns a descriptive string. Adding or subtracting the values isn't necessary to determine their relationship or to format the output string.

Node_2:
Service/Capability: Provides a function for string formatting similar to Python's built-in format method.
Implementation Analysis: The function implementation reveals several important details:
1. It accepts a template string with placeholders
2. It can handle both positional and keyword arguments 
3. It leverages Python's built-in format method for the actual formatting
4. It returns the formatted result as a string

Looking at this API design, it's specifically built for formatting strings with placeholders, which is exactly what would be needed for constructing comparison result messages.

Invocation Potential: The target function must return a string describing the relationship between two integers. It could use format_string to create this output message rather than using string concatenation or f-strings.
Decision: Included.
EVIDENCE: The target function compare_values must return a descriptive string about the relationship between integers. Analyzing the implementation, format_string provides exactly what's needed because:
1. It handles template strings with placeholders, ideal for the comparison output format
2. It accepts keyword arguments, allowing clear parameter naming (a=5, relation="greater", b=3)
3. The actual formatting is delegated to a mature method (string.format)
4. It returns exactly what compare_values needs to return - a formatted string

For example, compare_values could invoke it as format_string("{{a}} is {{relation}} than {{b}}", a=5, relation="greater", b=3) to produce "5 is greater than 3". This API is perfectly aligned with the target function's need to generate human-readable comparison results with variable content based on the actual comparison outcome.

Node_3:
Service/Capability: Provides a function to join multiple strings with a separator.
Implementation Analysis: The function implementation reveals several important details:
1. It requires a list of strings as its primary input parameter
2. It takes an optional separator parameter (defaulting to a space)
3. It simply delegates to Python's built-in string join method
4. It returns a single concatenated string

Looking at this API design, it's specifically built for combining multiple independent strings with a consistent separator character.

Invocation Potential: The target function only needs to return a single formatted string describing the relationship. It doesn't need to join multiple strings together to form its output.
Decision: Excluded.
EVIDENCE: When analyzing the implementation against the needs of compare_values:
1. The join_strings function requires an iterable of strings as input
2. A comparison result is naturally a single formatted message, not multiple discrete strings
3. The compare_values function has no clear need to join multiple strings with a separator

While technically it could be forced to work (e.g., join_strings([str(a), "is greater than", str(b)])), this would be an awkward and inefficient use of the API. The implementation is designed for joining collections of strings, but compare_values generates a single message with embedded values. There's no scenario where decomposing the comparison message into multiple strings for joining would be the natural or efficient approach.

Node_4:
Service/Capability: Encompasses various comparison functions for different data types.
Invocation Potential: The core task of compare_values is comparing integers. While Python built-in operators could suffice, the target function might leverage specialized comparison functions from this group for consistency, additional logic, or handling edge cases.
Decision: Included.
EVIDENCE: While the target function could implement comparison using built-in operators (>, <, ==), it's reasonable that it might leverage specialized comparison functions from this group for consistency with codebase patterns, handling edge cases, or advanced comparison logic. Since the summary indicates functions specifically designed for comparisons, there's a reasonable possibility the target function might invoke one of these functions.

Node_5:
Service/Capability: Provides logging and debugging utilities.
Invocation Potential: While not essential for the core comparison logic, the target function might use logging functions to track execution, debug issues, or record comparison results, especially in a production environment.
Decision: Included.
EVIDENCE: While not strictly necessary for the core comparison functionality, proper software engineering practice often incorporates logging. Being lenient, I can reasonably imagine the target function using logging to track inputs, outputs, or comparison results, especially in a production system with comprehensive logging requirements.

### Step 4: Compile the Final Output
```json
[
  {{"decision": false, "reason": "Arithmetic functions are unrelated to the goal of comparing and describing relationships between values."}},
  {{"decision": true, "reason": "The function formats output strings, which aligns with the target function's need to return descriptive messages."}},
  {{"decision": false, "reason": "Joining multiple strings is unnecessary for returning a single descriptive comparison message."}},
  {{"decision": true, "reason": "This node includes comparison utilities, which might be used instead of built-in operators for consistency or special logic."}},
  {{"decision": true, "reason": "Logging may be used for debugging or tracing behavior in the comparison function, especially in production environments."}}
]
```

### Input:
File Context:
```python
{self.file_context}
```

Target Function Code:
```python
{self.target_function_code}
```

Nodes:
{self.format_candidates(candidates)}

Analysis:"""
       
        return user_content

    # @retry(wait=gemini_wait, stop=stop_after_attempt(6))
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _get_relevant_response(self, candidates: List[Dict[str, Optional[str]]]) -> Tuple[List[int], List[str]]:
        """
        Calls the LLM and parses the response. This method is decorated with Tenacity for automatic retries.
        """
        user_msg = self._build_prompt(candidates)
        result = self.gemini.generate_content(
            user_msg,
            generation_config=genai.GenerationConfig(
                temperature=1.0,
                # response_mime_type="application/json",
                # response_schema=list[dict[str, str | bool]],
            )
        )
        logger.info(f"Gemini Code Retrieval")
        response_text = ""
        if hasattr(result, "text"):
            response_text = result.text
        elif isinstance(result, str):
            response_text = result
        else:
            response_text = result.generations[0][0].text
        return self._parse_response(response_text, num_candidates=len(candidates))

    def _parse_response(self, response_text: str, num_candidates: int) -> Tuple[List[int], List[str]]:
        """
        Parse the LLM response, extracting JSON and returning selected indices and reasons.
        """
        # First try to find JSON in markdown code blocks
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # If no markdown block found, try to find a JSON array directly
            match = re.search(r"\[\s*\{.*?\}\s*\]", response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                # fallback: try to find the first list-like structure
                match = re.search(r"($begin:math:display$\\s*{.*?}\\s*$end:math:display$)", response_text, re.DOTALL)
                if match:
                    json_str = match.group(1)
                else:
                    raise ValueError("Failed to find a JSON list in the response.")

        try:
            data = json.loads(json_str)
        except Exception as e1:
            try:
                data = ast.literal_eval(json_str)
            except Exception as e2:
                raise ValueError(f"Failed to parse model response as JSON.\nExtracted: {json_str}") from e2

        if not isinstance(data, list):
            raise ValueError(f"Model response is not a JSON list: {data}")
        if len(data) != num_candidates:
            raise ValueError(f"Expected {num_candidates} items, but got {len(data)}")

        selected_indices: List[int] = []
        selected_reasons: List[str] = []

        for i, item in enumerate(data):
            if not isinstance(item, dict) or "decision" not in item or "reason" not in item:
                raise ValueError(f"Item {i} is malformed: {item}")
            if isinstance(item["decision"], str):
                decision = item["decision"].strip().lower() == "true"
            else:
                decision = bool(item["decision"])
            if decision:
                selected_indices.append(i)
                selected_reasons.append(str(item["reason"]))

        return selected_indices, selected_reasons

    def is_relevant(self, candidates: List[Dict[str, Optional[str]]]) -> Tuple[List[int], List[str]]:
        """
        Determine which candidates are relevant to the target function by leveraging the Tenacity retry mechanism.
        """
        return self._get_relevant_response(candidates)
