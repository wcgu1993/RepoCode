import logging
import os
from abc import ABC, abstractmethod
import threading

import time
import random

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from google.api_core.exceptions import ResourceExhausted, DeadlineExceeded

import google.generativeai as genai

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        pass

        
class OpenAIEmbeddingModel(BaseEmbeddingModel):
    # def __init__(self, model="text-embedding-3-small"):
    def __init__(self, model="text-embedding-ada-002"):
        self.client = OpenAI()
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )


class GeminiEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        self.model_name = "models/embedding-001"
        genai.configure(api_key=api_key)
        
        # Rate limiting variables
        self.last_request_time = 0
        self.min_request_interval = 0.04  # 40ms between requests (1500/min = ~25/sec)
        self.request_count = 0
        self.minute_start = time.time()

    def _rate_limit(self):
        """Implement rate limiting to stay within API quotas"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.minute_start >= 60:
            self.request_count = 0
            self.minute_start = current_time
        
        # Check if we're approaching the limit (1500/min)
        if self.request_count >= 1400:  # Leave some buffer
            sleep_time = 60 - (current_time - self.minute_start)
            if sleep_time > 0:
                logger.info(f"Rate limit approaching, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                self.request_count = 0
                self.minute_start = time.time()
        
        # Ensure minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1

    @retry(
        wait=wait_random_exponential(min=1, max=60),  # Increased max wait time
        stop=stop_after_attempt(10),  # Increased retry attempts
        retry=retry_if_exception_type((ResourceExhausted, DeadlineExceeded, Exception))
    )
    def create_embedding(self, text):
        # Apply rate limiting before making the request
        self._rate_limit()
        
        try:
            response = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            logger.info("Gemini Embedding Model")
            return response["embedding"]
            
        except ResourceExhausted as e:
            print(f"Rate limit exceeded: {str(e)}")
            # Add extra delay for rate limit errors
            time.sleep(random.uniform(5, 15))
            raise
            
        except Exception as e:
            print(f"Error creating embedding: {str(e)}")
            raise


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)


class ThreadSafeSBertEmbedding(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.local = threading.local()
        self.model_name = model_name

    def create_embedding(self, text):
        if not hasattr(self.local, "model"):
            self.local.model = SentenceTransformer(self.model_name)
        return self.local.model.encode(text)