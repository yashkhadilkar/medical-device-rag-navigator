import os
import logging
import json
import time
import requests
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from enum import Enum
import subprocess
import tempfile

# Set this before importing any Hugging Face libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"  # For local open-source models

class LLMInterface:
    """Interface for LLM APIs and local models.
    Supports both cloud APIs and local open-source models.
    """
    
    def __init__(self, 
                 provider: Union[str, LLMProvider] = LLMProvider.LOCAL, 
                 model_name: Optional[str] = None,
                 api_key: Optional[str] = None,
                 max_tokens: int = 1024,
                 temperature: float = 0.1,
                 cache_dir: str = ".cache/llm_responses",
                 local_model_path: Optional[str] = None):
        """
        Initialize the LLM interface.
        
        Args:
            provider: LLM provider (openai, anthropic, huggingface, or local)
            model_name: Model name (if None, uses provider default)
            api_key: API key (if None, reads from environment variable)
            max_tokens: Maximum tokens in the response
            temperature: Sampling temperature
            cache_dir: Directory for caching responses
            local_model_path: Path to local model (for local provider)
        """
        if isinstance(provider, str):
            try:
                self.provider = LLMProvider(provider.lower())
            except ValueError:
                logger.warning(f"Unknown provider: {provider}. Using LOCAL.")
                self.provider = LLMProvider.LOCAL
        else:
            self.provider = provider
            
        self.model_name = model_name or self._get_default_model()
        self.api_key = api_key or self._get_api_key()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.local_model_path = local_model_path
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_dir = cache_dir
        else:
            self.cache_dir = None
            
        self.cache = {}

        if self.provider == LLMProvider.LOCAL:
            try:
                subprocess.run(["llama-cli", "--version"], 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE,
                             check=False)
                logger.info("Found llama.cpp. Will use for local inference.")
                self._local_implementation = "llama.cpp"
            except (FileNotFoundError, subprocess.SubprocessError):
                try:
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    logger.info("Found Hugging Face transformers. Will use for local inference.")
                    self._local_implementation = "transformers"
                except ImportError:
                    logger.warning("No viable local model implementation found. "
                                  "Install llama.cpp or transformers for local models.")
                    self._local_implementation = None
        
        logger.info(f"Initialized LLM interface for {self.provider.value} with model {self.model_name}")
    
    def _get_default_model(self) -> str:
        """Get the default model for the provider."""
        if self.provider == LLMProvider.OPENAI:
            return "gpt-4.1"
        elif self.provider == LLMProvider.ANTHROPIC:
            return "claude-instant-v1"
        elif self.provider == LLMProvider.HUGGINGFACE:
            return "mistralai/Mistral-7B-Instruct-v0.1"
        elif self.provider == LLMProvider.LOCAL:
            return "mistral-7b-instruct"  # Default open-source model
        else:
            return "mistral-7b-instruct"
            
    def _get_api_key(self) -> Optional[str]:
        """Get the API key from environment variables."""
        if self.provider == LLMProvider.OPENAI:
            return os.environ.get("OPENAI_API_KEY")
        elif self.provider == LLMProvider.ANTHROPIC:
            return os.environ.get("ANTHROPIC_API_KEY")
        elif self.provider == LLMProvider.HUGGINGFACE:
            return os.environ.get("HF_API_KEY")
        else:
            return None
            
    def _generate_cache_key(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a cache key from the prompts."""
        import hashlib
        combined = f"{prompt}|{system_prompt}|{self.model_name}|{self.temperature}|{self.max_tokens}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _call_openai_api(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call the OpenAI API using the current SDK.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            
        Returns:
            Model response
        """
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key)
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
                
            messages.append({"role": "user", "content": prompt})
            
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error calling OpenAI API: {e}")
                return ""
        except ImportError:
            logger.error("OpenAI package not installed. Install with: pip install openai")
            return "ERROR: OpenAI package not installed. Install with: pip install openai"
    
    def _call_anthropic_api(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call the Anthropic API.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            
        Returns:
            Model response
        """
        url = "https://api.anthropic.com/v1/complete"
        
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        system = system_prompt if system_prompt else ""
        
        data = {
            "model": self.model_name,
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
            "max_tokens_to_sample": self.max_tokens,
            "temperature": self.temperature,
            "system": system
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            
            return response_data.get("completion", "")
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            return ""
    
    def _call_huggingface_api(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call the Hugging Face Inference API.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            
        Returns:
            Model response
        """
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
            
        api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": self.max_tokens,
                "temperature": self.temperature,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            
            if isinstance(response_data, list) and len(response_data) > 0:
                return response_data[0].get("generated_text", "")
            else:
                return response_data.get("generated_text", "")
        except Exception as e:
            logger.error(f"Error calling Hugging Face API: {e}")
            return ""
            
    def _call_local_model_llama_cpp(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call a local model using llama.cpp.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            
        Returns:
            Model response
        """
        if system_prompt:
            formatted_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
        else:
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            prompt_file = f.name
            f.write(formatted_prompt)

        output_file = os.path.join(self.cache_dir, f"response_{int(time.time())}.txt")

        model_path = self.local_model_path
        if not model_path or not os.path.exists(model_path):
            logger.warning("Model path not specified or not found.")
            if os.path.exists("models"):
                model_files = list(Path("models").glob("*.gguf"))
                if model_files:
                    model_path = str(model_files[0])
                    logger.info(f"Using model found at: {model_path}")
                else:
                    logger.error("No model files found in models directory.")
                    return "ERROR: No model files found."
            else:
                logger.error("No model path specified and no models directory found.")
                return "ERROR: No model path specified."
                
        try:
            cmd = [
                "llama-cli", "generate",
                "--model", model_path,
                "--file", prompt_file,
                "--temp", str(self.temperature),
                "--max-tokens", str(self.max_tokens),
                "--no-display-prompt"
            ]
            
            with open(output_file, 'w') as out_f:
                process = subprocess.run(
                    cmd, 
                    stdout=out_f,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                
            if process.returncode != 0:
                logger.error(f"Error running llama.cpp: {process.stderr}")
                return f"ERROR: Failed to run local model. {process.stderr}"

            with open(output_file, 'r') as f:
                response = f.read().strip()

            try:
                os.unlink(prompt_file)
            except:
                pass
                
            return response
        except Exception as e:
            logger.error(f"Error generating with local model: {e}")
            return f"ERROR: {str(e)}"
            
    def _call_local_model_transformers(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call a local model using Hugging Face transformers.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            
        Returns:
            Model response
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch

            if system_prompt:
                if "mistral" in self.model_name.lower() or "llama" in self.model_name.lower():
                    formatted_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
                else:
                    formatted_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            else:
                if "mistral" in self.model_name.lower() or "llama" in self.model_name.lower():
                    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
                else:
                    formatted_prompt = f"User: {prompt}\n\nAssistant:"

            model_name_or_path = self.local_model_path or self.model_name

            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    load_in_8bit=True,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            except Exception as e:
                logger.warning(f"Failed to load with 8-bit precision: {e}. Trying standard loading.")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    device_map="auto"
                )

            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True
            )

            with torch.no_grad():
                result = generator(formatted_prompt)
                
            response = result[0]["generated_text"]

            if "mistral" in self.model_name.lower() or "llama" in self.model_name.lower():
                response = response.split("[/INST]")[-1].strip()
            else:
                response = response.split("Assistant:")[-1].strip()
                
            return response
        except Exception as e:
            logger.error(f"Error generating with transformers: {e}")
            return f"ERROR: {str(e)}"
            
    def generate(self, prompt: str, system_prompt: Optional[str] = None, use_cache: bool = True) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            use_cache: Whether to use cached responses
            
        Returns:
            Generated text
        """
        cache_key = self._generate_cache_key(prompt, system_prompt)
        
        if use_cache and cache_key in self.cache:
            logger.info("Using cached response")
            return self.cache[cache_key]
            
        if use_cache and self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                        self.cache[cache_key] = cached_data["response"]
                        logger.info("Using cached response from disk")
                        return cached_data["response"]
                except Exception as e:
                    logger.warning(f"Error reading cache: {e}")
                    
        logger.info(f"Generating response using {self.provider.value}")
        
        if self.provider == LLMProvider.OPENAI:
            response = self._call_openai_api(prompt, system_prompt)
        elif self.provider == LLMProvider.ANTHROPIC:
            response = self._call_anthropic_api(prompt, system_prompt)
        elif self.provider == LLMProvider.HUGGINGFACE:
            response = self._call_huggingface_api(prompt, system_prompt)
        elif self.provider == LLMProvider.LOCAL:
            if self._local_implementation == "llama.cpp":
                response = self._call_local_model_llama_cpp(prompt, system_prompt)
            elif self._local_implementation == "transformers":
                response = self._call_local_model_transformers(prompt, system_prompt)
            else:
                logger.error("No viable local model implementation available")
                response = "ERROR: No viable local model implementation available."
        else:
            logger.error(f"Unknown provider: {self.provider}")
            return "ERROR: Unknown provider."
            
        if not response:
            logger.warning("Empty response from model")
            return "No response generated. Please try again with a different prompt or model."
            
        self.cache[cache_key] = response
        
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
            try:
                cache_data = {
                    "provider": self.provider.value,
                    "model": self.model_name,
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "timestamp": time.time(),
                    "response": response
                }
                
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Error saving cache: {e}")
        
        return response
    
    def generate_with_context(self, 
                             question: str, 
                             context_docs: List[Dict[str, Any]],
                             system_prompt: Optional[str] = None,
                             use_cache: bool = True) -> str:
        """
        Generate a response with context documents.
        
        Args:
            question: User question
            context_docs: List of context documents
            system_prompt: System prompt
            use_cache: Whether to use cached responses
            
        Returns:
            Generated response
        """
        formatted_context = self._format_context(context_docs)
        
        # Debug the context format and length
        logger.info(f"Context format complete. Total context length: {len(formatted_context)} characters")
        if len(formatted_context) < 100:
            logger.warning(f"Context is very short: {formatted_context}")
        
        prompt = f"""
Please answer the following question based on the provided context information.
If the context doesn't contain the answer, say so without making up information.

Context:
{formatted_context}

Question: {question}

Answer:
"""
        
        return self.generate(prompt, system_prompt, use_cache)
    
    def _format_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Format context documents for the prompt."""
        formatted_context = ""
        
        for i, doc in enumerate(context_docs):
            # Make sure to include the actual text content
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            title = metadata.get("title", f"Document {i+1}")
            source = metadata.get("source", "Unknown")
            
            # Debug info to verify text is present
            logger.info(f"DEBUG: Document {i+1} text length: {len(text)}")
            
            # Check if text is missing or too short, try to find it in metadata
            if len(text) < 50 and "text" in metadata:
                logger.info(f"Using text from metadata ({len(metadata['text'])} chars) instead of short document text")
                text = metadata["text"]
            
            formatted_context += f"Document {i+1}: {title} (Source: {source})\n{text}\n\n"
            
        return formatted_context.strip()
    
    def clear_cache(self):
        """Clear the response cache."""
        self.cache = {}
        
        if self.cache_dir and os.path.exists(self.cache_dir):
            try:
                for file_path in Path(self.cache_dir).glob("*.json"):
                    os.remove(file_path)
                    
                logger.info("Cleared response cache")
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
