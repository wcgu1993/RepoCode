from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_random_exponential
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from abc import ABC, abstractmethod
import re, os, logging

import openai
import google.generativeai as genai

from .model_utils import gemini_wait

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

openai.api_key = os.getenv('OPENAI_API_KEY')

class BaseCodeGeneratorModel(ABC):
    @abstractmethod
    def generate_code(self, context, query):
        pass

class OpenAICodeGeneratorModel(BaseCodeGeneratorModel):
    def __init__(self, model="gpt-4.1-mini"):
    # def __init__(self, model="gpt-4o-mini"):
        """Initialize the code generator with a model, prompt, and chain."""
        self.llm = ChatOpenAI(model=model, temperature=1.0)
        template = """
You are an advanced Python code generator. You have a **query function** (with its signature and docstring) that needs to be implemented, and you are given several **context functions** (from the same codebase) that might be useful for this task, **file content** which locates the query function, and **dependencies** which contains the installed dependencies in the environment where the code is being generated.

Follow these steps **before writing the code**:
1. **Understand the Query:** Read the query function's signature and docstring carefully. Determine what the function is supposed to do and outline a plan for how to implement it.
2. **Analyze Context Functions:** For **EACH** function or method provided in *Context Functions*:
   - **Purpose:** Understand what the function does based on its description.
   - **Behavior:** Review its implementation details to see how it works and handles edge cases.
   - **Usage:** Decide if this context function can help implement the query. If yes, figure out how to use it (note any required imports or class instances).
3. **Analyze File Content:** Review the provided File Content carefully, which contains code from the beginning of the file up to the target function's definition. Use a chain-of-thought approach to understand the implementation pattern and target function environment. **DO NOT** import any dependency from file context.
4. **Analyze Dependencies:** Review the provided dependencies and the version of the dependencies. Make sure that your implementation is compatible with the dependencies.
5. **Plan the Implementation:** Refine your plan for the query function using relevant context functions and the file content insights. Determine where and how to call them, and ensure you have any necessary imports or initializations.
6. **Implement the Function:** Now write the full code for the query function in Python, following your plan:
   - Include all necessary import statements at the top so the code runs independently.
   - Implement the function step by step, handling edge cases and using context functions where appropriate.
   - **Strict Output Format:** The code must be delimited with Python code delimiter (between "```python" and "```")

**File Content:**
{file_content}

**Dependencies:**
{dependencies}

**Query Function:**
{query}

**Context Functions:**
{context_text}
"""
        self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = self.prompt | self.llm

    def parse_code(self, text: str) -> str:
        """
        Extracts and returns the last Python code block delimited by '```python' and '```' from the input string.

        Args:
            text (str): The input string containing one or more Python code blocks.
            
        Returns:
            str: The code within the last Python code block, or an empty string if none is found.
        """
        pattern = r"```python(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[-1].strip() if matches else ""

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def generate_code(self, context: List[Dict], file_content: str, dependencies: str, query: str) -> str:
        """
        Generates Python code for the given query using the provided context functions and file content.
        Args:
            context (List[Dict]): A list of dictionaries with information about a context function.
            file_content (str): The code from the beginning of the file up to the target function's definition.
            dependencies (str): The dependencies in the environment where the code is being generated.
            query (str): The function definition (signature and docstring) to be implemented.
        Returns:
            str: The generated Python code for the query function, with only the output code.
        """
        context_text = ""
        for idx, function in enumerate(context):
            context_text += f"Context Function {idx+1}:\n"
            context_text += f"  Description: {function.get('description', 'No description provided')}\n"
            if 'import_statement' in function:
                context_text += f"  Import Statement: {function['import_statement']}\n"
            if function.get('is_method', False):
                context_text += f"  Class: This is a method of Class {function.get('class_name', 'Unknown')}\n"
                if 'constructor' in function:
                    context_text += f"  Constructor: {function['constructor']}\n"
            context_text += f"  Implementation: {function.get('function', 'No implementation provided')}\n\n"
        
        
        try:
            
            result = self.chain.invoke({
                "query": query,
                "context_text": context_text,
                "file_content": file_content,
                "dependencies": dependencies
            })
            answer = self.parse_code(result.content)
            return answer
        except Exception as e:
            return f"Error: {e}"


class GeminiFlashCodeGeneratorModel(BaseCodeGeneratorModel):
    def __init__(self):
        """Initialize the code generator with a model, prompt, and chain."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        model_name = "models/gemini-2.5-flash-preview-05-20"
        genai.configure(api_key=api_key)
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            safety_settings=safety_settings
        )
        
        self.template = """
You are an advanced Python code generator. You have a **query function** (with its signature and docstring) that needs to be implemented, and you are given several **context functions** (from the same codebase) that might be useful for this task, **file content** which locates the query function, and **dependencies** which contains the installed dependencies in the environment where the code is being generated.

Follow these steps **before writing the code**:
1. **Understand the Query:** Read the query function's signature and docstring carefully. Determine what the function is supposed to do and outline a plan for how to implement it.
2. **Analyze Context Functions:** For **EACH** function or method provided in *Context Functions*:
   - **Purpose:** Understand what the function does based on its description.
   - **Behavior:** Review its implementation details to see how it works and handles edge cases.
   - **Usage:** Decide if this context function can help implement the query. If yes, figure out how to use it (note any required imports or class instances).
3. **Analyze File Content:** Review the provided File Content carefully, which contains code from the beginning of the file up to the target function's definition. Use a chain-of-thought approach to understand the implementation pattern and target function environment. **DO NOT** import any dependency from file context.
4. **Analyze Dependencies:** Review the provided dependencies and the version of the dependencies. Make sure that your implementation is compatible with the dependencies.
5. **Plan the Implementation:** Refine your plan for the query function using relevant context functions and the file content insights. Determine where and how to call them, and ensure you have any necessary imports or initializations.
6. **Implement the Function:** Now write the full code for the query function in Python, following your plan:
   - Include all necessary import statements at the top so the code runs independently.
   - Implement the function step by step, handling edge cases and using context functions where appropriate.
   - **Strict Output Format:** The code must be delimited with Python code delimiter (between "```python" and "```")

**File Content:**
{file_content}

**Dependencies:**
{dependencies}

**Query Function:**
{query}

**Context Functions:**
{context_text}
"""

    def parse_code(self, text: str) -> str:
        """
        Extracts and returns the last Python code block delimited by '```python' and '```' from the input string.

        Args:
            text (str): The input string containing one or more Python code blocks.
            
        Returns:
            str: The code within the last Python code block, or an empty string if none is found.
        """
        pattern = r"```python(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[-1].strip() if matches else ""

    # @retry(wait=gemini_wait, stop=stop_after_attempt(6))
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def generate_code(self, context: List[Dict], file_content: str, dependencies: str, query: str) -> str:
        """
        Generates Python code for the given query using the provided context functions and file content.
        Args:
            context (List[Dict]): A list of dictionaries with information about a context function.
            file_content (str): The code from the beginning of the file up to the target function's definition.
            dependencies (str): The dependencies in the environment where the code is being generated.
            query (str): The function definition (signature and docstring) to be implemented.
        Returns:
            str: The generated Python code for the query function, with only the output code.
        """
        context_text = ""
        for idx, function in enumerate(context):
            context_text += f"Context Function {idx+1}:\n"
            context_text += f"  Description: {function.get('description', 'No description provided')}\n"
            if 'import_statement' in function:
                context_text += f"  Import Statement: {function['import_statement']}\n"
            if function.get('is_method', False):
                context_text += f"  Class: This is a method of Class {function.get('class_name', 'Unknown')}\n"
                if 'constructor' in function:
                    context_text += f"  Constructor: {function['constructor']}\n"
            context_text += f"  Implementation: {function.get('function', 'No implementation provided')}\n\n"
        
        # Truncate context_text if too long (minimal fix)
        if len(context_text) > 500000:
            context_text = context_text[:500000] + "\n... (context truncated due to length)"
        
        # Truncate file_content if too long (minimal fix)
        if len(file_content) > 80000:
            file_content = file_content[:80000] + "\n... (file content truncated due to length)"
        
        prompt = self.template.format(
            query=query,
            context_text=context_text,
            file_content=file_content,
            dependencies=dependencies
        )
        
        resp = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=1.0)
        )
        logger.info(f"Gemini Code Generator")
        
        code = self.parse_code(resp.text)
        if not code:
            raise ValueError("No ```python``` block found in the model response.")
        return code

class PlainOpenAICodeGeneratorModel(BaseCodeGeneratorModel):
    def __init__(self, model="gpt-4.1-mini"):
    # def __init__(self, model="gpt-4o-mini"):
        self.model = model
    
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def generate_code(self, function_definition: str):
        """
        Given the query (function signature, docstring or explanation), it generates the code

        Args:
            function_definition (str): Information about the function to generate (function signature, docstring or explanation)
        
        Returns:
            str: generated code
        """
        # Define the prompt to instruct the model to summarize the function
        prompt = f"""You are an expert Python programmer. Your task is to complete a Python function for a repository under development.
        
You will receive:
1. A function definition. It might be only function signature or function signature and docstring.

The output should be just the entire function. No further text or markdown should be included.
You should include the function definition as it is given to you. Do not make any changes on the function signature or docstring given to you.

Function definition: {function_definition}

Entire function:
"""
        try:
            response = openai.chat.completions.create( # openai.ChatCompletion.create(
                model="gpt-4.1-mini",
                # model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert Python programmer."},
                    {"role": "user", "content": prompt},
                ],
                # temperature=0,
            )
            return response.choices[0].message.content #.strip()    ### response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error: {e}"
