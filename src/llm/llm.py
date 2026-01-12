"""LLM wrapper for Ollama with enhanced functionality."""

import os
from typing import Optional, Any, Dict, List, Union, Type
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from src.utils import get_logger, retry

load_dotenv()
logger = get_logger(__name__)


class LLM:
    """
    LLM wrapper for Ollama.
    """
    
    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        temperature: float = 0.3,
        num_predict: int = 2048,
        top_p: float = 0.9,
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize LLM with Ollama.
        
        Args:
            model_name: Ollama model to use (e.g., 'llama2', 'mistral', 'qwen2.5:3b')
            temperature: Sampling temperature (0-1)
            num_predict: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            base_url: Ollama server URL
        """
        logger.info(f"Initializing {model_name}")
        
        # Store configuration
        self.model_name = model_name
        self.temperature = temperature
        self.num_predict = num_predict
        self.base_url = base_url
        
        # Initialize LLM
        try:
            self.llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                num_predict=num_predict,
                top_p=top_p,
                base_url=base_url
            )
            
            logger.info(f"LLM initialized successfully with Ollama")
        except Exception as e:
            logger.error(f"Failed to initialize {model_name}: {e}")
            raise RuntimeError(f"LLM initialization failed: {e}") from e

    def bind_tools(self, tools: List) -> BaseChatModel:
        """
        Bind tools to LLM for tool calling.
        
        Args:
            tools: List of @tool decorated functions
            
        Returns:
            LLM with tools bound
        """
        return self.llm.bind_tools(tools)
    
    @retry(retries=3, delay=1.0, backoff=2.0)
    def invoke(
        self,
        prompt: str,
        system_message: Optional[str] = None
    ) -> str:
        """
        Invoke LLM with a simple prompt.
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            
        Returns:
            Generated text response
        """
        try:
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=prompt))
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to invoke LLM: {e}")
            raise RuntimeError(f"LLM invocation failed: {e}") from e
    
    @retry(retries=3, delay=1.0, backoff=2.0)
    def invoke_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system_message: Optional[str] = None
    ) -> BaseModel:
        """
        Invoke LLM with structured output using Pydantic schema.
        
        Args:
            prompt: User prompt
            schema: Pydantic model class defining output structure
            system_message: Optional system message
            
        Returns:
            Validated Pydantic model instance
        """
        try:
            # Create structured LLM
            structured_llm = self.llm.with_structured_output(schema)
            
            # Build messages
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=prompt))
            
            # Invoke and return structured output
            return structured_llm.invoke(messages)
            
        except Exception as e:
            logger.error(f"Failed to get structured output: {e}")
            raise RuntimeError(f"Structured output failed: {e}") from e