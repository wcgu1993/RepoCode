import os, logging
from abc import ABC, abstractmethod

import openai
import google.generativeai as genai

from tenacity import retry, stop_after_attempt, wait_random_exponential

from transformers import AutoTokenizer, pipeline

from .model_utils import gemini_wait

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# You can define your own Summarization model by extending the base Summarization Class. 
class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass

class OpenAISummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-4.1-mini"):
   #  def __init__(self, model="gpt-4o-mini"):
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500):

        prompt = f"""Your task is to craft a summary for a set of function or function group descriptions. The summary should clearly address: 'What shared capabilities do these functions provide that make them valuable to others?'

**Summary Requirements:**
- Keep it concise, **STRICTLY LESS** than 35 words and **MAXIMUM** 2 sentences.
- The summary **MUST** be plain text, **NO** enumerations, bullet points.
- Describe the capabilities from the perspective of the services provided.
- Highlight the purpose of these services and include potential use cases.

**Approach:**
1. Review each function description individually to pinpoint its purpose and capabilities.
2. Identify overlapping features, complementary services, or common goals among the functions.
3. Explore use cases where this group of functions could benefit other parts of a system.
4. Write a summary that showcases their collective value.

**Guidelines:**
- For strongly related functions, deliver a detailed summary of their shared capability.
- For subtly related functions, provide a brief summary emphasizing general utility.
- For unrelated functions, state exactly: 'No clear commonality can be found among the descriptions.'
- Focus solely on capabilities and services, avoiding implementation details.
- Use clear, direct language to ensure accessibility.
- Emphasize what makes this group of functions uniquely valuable together.

EXAMPLES:

Example 1:
Descriptions:
1. Checks if user input contains special characters, is within length limits, and matches expected format.
2. Removes potentially harmful characters, escapes HTML entities, and normalizes whitespace in user-provided data.
3. Applies AES-256 encryption to user passwords and sensitive information before storage.

My analysis:
1. Function-by-function examination:
   - First function: Ensures input meets quality standards by checking format, length, and character validity.
   - Second function: Takes raw user data and makes it safe by removing dangerous elements.
   - Third function: Secures sensitive data through strong encryption.

2. Identifying relationships and common themes:
   All three functions deal with user-provided data, but focus on different aspects of securing that data:
   - Input validation ensures quality
   - Sanitization ensures safety
   - Encryption ensures confidentiality
   They form a security chain for processing user data.

3. Potential use cases:
   - User registration systems needing to handle credentials securely
   - Form submission handlers requiring input validation
   - Any system that accepts and stores user-generated content

4. Common capability analysis:
   These functions collectively provide data safety mechanisms at different stages of user data processing.

Summary: These functions provide a comprehensive security pipeline for processing user-provided data, validating input quality, sanitizing potentially harmful content, and encrypting sensitive information. Their combined capabilities enable secure user data handling in applications that collect, process, and store user-generated content such as registration systems and form handlers.

Example 2:
Descriptions:
1. Retrieves current weather conditions from external API based on city or coordinates.
2. Handles financial transactions by validating payment details and communicating with payment gateways.
3. Creates unique alphanumeric identifiers for system entities.

My analysis:
1. Function-by-function examination:
   - First function: External API consumer for weather information based on location.
   - Second function: Financial transaction handler that interfaces with payment services.
   - Third function: Utility for creating unique identifiers within a system.

2. Identifying relationships and common themes:
   These functions serve quite different purposes:
   - Weather data retrieval is about external information gathering
   - Payment processing handles financial transactions
   - ID generation is a system utility
   There's no strong thematic connection beyond being utility functions.

3. Potential use cases:
   These would be used in different contexts:
   - Weather data for informational displays or planning applications
   - Payment processing for e-commerce
   - ID generation across many system components

4. Common capability analysis:
   The functions don't share a clear common purpose or capability domain. 
   They represent disparate utility services for different application needs.

Summary: No clear commonality can be found among the descriptions.

Descriptions:
{context}

Important Note: The summary must obey the requirements. It should be plain text at most 2 sentences and under 35 words.

My analysis:"""

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing software components and their capabilities."},
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
               #  temperature=0
            )

            response = response.choices[0].message.content
            response = response.replace("*", "").strip()
            
            # Extract just the summary part if the model includes its analysis
            if "Summary:" in response:
                response = response.split("Summary:")[-1].strip()
            
            return response

        except Exception as e:
            print(e)
            return e
        

class GeminiFlashSummarizationModel(BaseSummarizationModel):
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
            system_instruction="You are an expert at analyzing software components and their capabilities.",
            safety_settings=safety_settings
        )

   #  @retry(wait=gemini_wait, stop=stop_after_attempt(6))
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500):

        prompt = f"""Your task is to craft a summary for a set of function or function group descriptions. The summary should clearly address: 'What shared capabilities do these functions provide that make them valuable to others?'

**Summary Requirements:**
- Keep it concise, **STRICTLY LESS** than 35 words and **MAXIMUM** 2 sentences.
- The summary **MUST** be plain text, **NO** enumerations, bullet points.
- Describe the capabilities from the perspective of the services provided.
- Highlight the purpose of these services and include potential use cases.

**Approach:**
1. Review each function description individually to pinpoint its purpose and capabilities.
2. Identify overlapping features, complementary services, or common goals among the functions.
3. Explore use cases where this group of functions could benefit other parts of a system.
4. Write a summary that showcases their collective value.

**Guidelines:**
- For strongly related functions, deliver a detailed summary of their shared capability.
- For subtly related functions, provide a brief summary emphasizing general utility.
- For unrelated functions, state exactly: 'No clear commonality can be found among the descriptions.'
- Focus solely on capabilities and services, avoiding implementation details.
- Use clear, direct language to ensure accessibility.
- Emphasize what makes this group of functions uniquely valuable together.

EXAMPLES:

Example 1:
Descriptions:
1. Checks if user input contains special characters, is within length limits, and matches expected format.
2. Removes potentially harmful characters, escapes HTML entities, and normalizes whitespace in user-provided data.
3. Applies AES-256 encryption to user passwords and sensitive information before storage.

My analysis:
1. Function-by-function examination:
   - First function: Ensures input meets quality standards by checking format, length, and character validity.
   - Second function: Takes raw user data and makes it safe by removing dangerous elements.
   - Third function: Secures sensitive data through strong encryption.

2. Identifying relationships and common themes:
   All three functions deal with user-provided data, but focus on different aspects of securing that data:
   - Input validation ensures quality
   - Sanitization ensures safety
   - Encryption ensures confidentiality
   They form a security chain for processing user data.

3. Potential use cases:
   - User registration systems needing to handle credentials securely
   - Form submission handlers requiring input validation
   - Any system that accepts and stores user-generated content

4. Common capability analysis:
   These functions collectively provide data safety mechanisms at different stages of user data processing.

Summary: These functions provide a comprehensive security pipeline for processing user-provided data, validating input quality, sanitizing potentially harmful content, and encrypting sensitive information. Their combined capabilities enable secure user data handling in applications that collect, process, and store user-generated content such as registration systems and form handlers.

Example 2:
Descriptions:
1. Retrieves current weather conditions from external API based on city or coordinates.
2. Handles financial transactions by validating payment details and communicating with payment gateways.
3. Creates unique alphanumeric identifiers for system entities.

My analysis:
1. Function-by-function examination:
   - First function: External API consumer for weather information based on location.
   - Second function: Financial transaction handler that interfaces with payment services.
   - Third function: Utility for creating unique identifiers within a system.

2. Identifying relationships and common themes:
   These functions serve quite different purposes:
   - Weather data retrieval is about external information gathering
   - Payment processing handles financial transactions
   - ID generation is a system utility
   There's no strong thematic connection beyond being utility functions.

3. Potential use cases:
   These would be used in different contexts:
   - Weather data for informational displays or planning applications
   - Payment processing for e-commerce
   - ID generation across many system components

4. Common capability analysis:
   The functions don't share a clear common purpose or capability domain. 
   They represent disparate utility services for different application needs.

Summary: No clear commonality can be found among the descriptions.

Descriptions:
{context}

Important Note: The summary must obey the requirements. It should be plain text at most 2 sentences and under 35 words.

My analysis:"""

        resp = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=1.0)
        )
        logger.info(f"Gemini Summarization")
        text = resp.text.strip()
        # Remove analysis if present and keep only the summary
        if "Summary:" in text:
            text = text.split("Summary:")[-1]
        return text.replace("*", "").strip()

