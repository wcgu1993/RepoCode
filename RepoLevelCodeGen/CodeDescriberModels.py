from abc import ABC, abstractmethod
import os, re, logging
from typing import Dict

import openai
import google.generativeai as genai

from tenacity import retry, stop_after_attempt, wait_random_exponential

from .model_utils import gemini_wait

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

openai.api_key = os.getenv('OPENAI_API_KEY')

class BaseCodeDescriberModel(ABC):
    @abstractmethod
    def describe_code(self, context):
        pass


class OpenAICodeDescriberModel(BaseCodeDescriberModel):
    def __init__(self, model="gpt-4.1-mini"):
    # def __init__(self, model="gpt-4o-mini"):
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def describe_code(self, function: Dict):
        """
        Sends the given function code to GPT-4.1-mini to get a natural language description.

        Args:
            function (Dict): Python function code structure
        
        Returns:
            str: Natural language description of the function.
        """
        
        body = function['body']
        is_method = function.get('is_method', False)
        
        # Define the prompt to instruct the model to summarize the function
        prompt = """Your task is to create natural language descriptions of Python functions.

Focus: Describe WHAT the function does, not HOW it does it.

Key Question: "What service/capability does this function provide that other functions might want to use?"

Function Analysis:
1. Make sure you understand the whole function
2. Focus on the function's primary purpose/service and core functionality
3. Emphasize the function's potential use cases

**Note**: If the function is a method, its class name will be provided under `Class:`. Please use it to better understand the context and responsibility of the method.

Description Requirements:
1. The description MUST be concise, strictly less than 40 words and maximum 2 sentences
2. Use clear, direct language
3. Return exactly "Not Implemented" for placeholder functions, not implemented functions, or functions with no retrieval value
4. Return exactly "Not sure" if the function's purpose is unclear

Example:
Class: DocumentEmbedder
```python
def retrieve_document_embeddings(doc_id: str, embedding_model: EmbeddingModel) -> Dict[str, np.ndarray]:
    \"""
    Retrieve or generate vector embeddings for a document.
    
    Args:
        doc_id: Unique identifier of the document
        embedding_model: Model to use for generating embeddings
        
    Returns:
        Dictionary mapping section names to embedding vectors
    \"""
    document = document_store.get(doc_id)
    if document is None:
        raise ValueError(f"Document with ID {{doc_id}} not found")
    
    cached_embeddings = cache.get_embeddings(doc_id)
    if cached_embeddings:
        return cached_embeddings
    
    sections = document.get_sections()
    embeddings = {{}}
    for section_name, section_text in sections.items():
        embeddings[section_name] = embedding_model.encode(section_text)
    
    cache.store_embeddings(doc_id, embeddings)
    return embeddings
```

Chain of Thought:
Looking at this method, it's clearly dealing with document embeddings in an NLP context. The name retrieve_document_embeddings reveals its purpose. Analyzing its role, this is a service provider function with retrieval and computation capabilities. It first tries to find cached embeddings before generating new ones, showing it's also optimized for performance.
The method operates in the natural language processing and vector embedding domain, working with document sections rather than whole documents, enabling granular text representation. It serves as a key utility method that bridges document storage and vector representation systems.
From a domain perspective, this method is critical for semantic search pipelines, employing a section-level embedding approach that supports fine-grained retrieval and similarity computations.

Output: Core utility method of the DocumentEmbedder class that retrieves or generates vector embeddings for document sections, enabling semantic search with caching for efficiency.
Now, please describe this function:
"""
        
        if is_method:
            prompt += f"\nClass: {function['class_name']}"
        
        prompt += f"\n\n```python\n{body}\n```\n\Chain of Thought:"
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert Python programmer."},
                    {"role": "user", "content": prompt},
                ],
                # temperature=0
            )
            response = response.choices[0].message.content
            if "Output: " in response:
                response = response.split("Output: ")[-1].strip()
            return response
        except Exception as e:
            return f"Error: {e}"
        

class GeminiFlashCodeDescriberModel(BaseCodeDescriberModel):
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        model_name="models/gemini-2.5-flash-preview-05-20"
        genai.configure(api_key=api_key)
        
        # Add safety settings to prevent blocking
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction="You are an expert Python programmer.",
            safety_settings=safety_settings
        )
        

    # @retry(wait=gemini_wait, stop=stop_after_attempt(6))
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def describe_code(self, function: Dict):
        """
        Sends the given function code to Gemini 2.5 Flash to get a natural language description.

        Args:
            function (Dict): Python function code structure
        
        Returns:
            str: Natural language description of the function.
        """
        body = function['body']
        is_method = function.get('is_method', False)
        
        # Define the prompt to instruct the model to summarize the function
        prompt = """Your task is to create natural language descriptions of Python functions.

Focus: Describe WHAT the function does, not HOW it does it.

Key Question: "What service/capability does this function provide that other functions might want to use?"

Function Analysis:
1. Make sure you understand the whole function
2. Focus on the function's primary purpose/service and core functionality
3. Emphasize the function's potential use cases

**Note**: If the function is a method, its class name will be provided under `Class:`. Please use it to better understand the context and responsibility of the method.

Description Requirements:
1. The description MUST be concise, strictly less than 40 words and maximum 2 sentences
2. Use clear, direct language
3. Return exactly "Not Implemented" for placeholder functions, not implemented functions, or functions with no retrieval value
4. Return exactly "Not sure" if the function's purpose is unclear

Example:
Class: DocumentEmbedder
```python
def retrieve_document_embeddings(doc_id: str, embedding_model: EmbeddingModel) -> Dict[str, np.ndarray]:
    \"""
    Retrieve or generate vector embeddings for a document.
    
    Args:
        doc_id: Unique identifier of the document
        embedding_model: Model to use for generating embeddings
        
    Returns:
        Dictionary mapping section names to embedding vectors
    \"""
    document = document_store.get(doc_id)
    if document is None:
        raise ValueError(f"Document with ID {{doc_id}} not found")
    
    cached_embeddings = cache.get_embeddings(doc_id)
    if cached_embeddings:
        return cached_embeddings
    
    sections = document.get_sections()
    embeddings = {{}}
    for section_name, section_text in sections.items():
        embeddings[section_name] = embedding_model.encode(section_text)
    
    cache.store_embeddings(doc_id, embeddings)
    return embeddings
```

Chain of Thought:
Looking at this method, it's clearly dealing with document embeddings in an NLP context. The name retrieve_document_embeddings reveals its purpose. Analyzing its role, this is a service provider function with retrieval and computation capabilities. It first tries to find cached embeddings before generating new ones, showing it's also optimized for performance.
The method operates in the natural language processing and vector embedding domain, working with document sections rather than whole documents, enabling granular text representation. It serves as a key utility method that bridges document storage and vector representation systems.
From a domain perspective, this method is critical for semantic search pipelines, employing a section-level embedding approach that supports fine-grained retrieval and similarity computations.

Output: Core utility method of the DocumentEmbedder class that retrieves or generates vector embeddings for document sections, enabling semantic search with caching for efficiency.
Now, please describe this function:
"""
        
        if is_method:
            prompt += f"\nClass: {function['class_name']}"

        prompt += f"\n\n```python\n{body}\n```\n\Chain of Thought:"

        try:
            resp = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(temperature=1.0)
            )
            logger.info(f"Gemini Describer")
            # Check if response contains valid content
            if (resp.candidates and 
                len(resp.candidates) > 0 and 
                resp.candidates[0].content and 
                resp.candidates[0].content.parts):
                
                text = resp.text.strip()
                return text.split("Output:")[-1].strip()
            
            else:
                # Handle blocked/filtered content
                finish_reason = resp.candidates[0].finish_reason if resp.candidates else "Unknown"
                logger.info(f"Response blocked or empty. Finish reason: {finish_reason}")
                
                # Return a default description or raise an exception
                return "Code description unavailable due to content filtering."
                
        except Exception as e:
            logger.info(f"Error generating content: {str(e)}")
            # You might want to re-raise or return a default value
            raise
