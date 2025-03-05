"""
Query Engine - Handles generation of responses using LLMs.
"""

import os
import logging
from typing import List, Dict, Any, Optional

from ..storage.vector_store import VectorStore

# Set up logging
logger = logging.getLogger(__name__)

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """
You are Codex, an AI assistant specialized in helping developers understand codebases.
You are given context from the codebase including code snippets and documentation.
Use this context to provide accurate, helpful and concise answers.

When referencing code:
- Mention the file path
- Explain the purpose of the code
- Be specific about functions, classes, and variables
- Show brief code examples when helpful

Your goal is to help new developers understand the codebase without having to bother senior engineers.
"""


class QueryEngine:
    """Engine for querying LLMs with codebase context."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        data_dir: str = ".codex_data",
        openai_api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the query engine.

        Args:
            vector_store: Vector store for retrieving context
            data_dir: Directory containing vector store data
            openai_api_key: OpenAI API key (will use environment variable if not provided)
            system_prompt: Custom system prompt
        """
        # Set up the vector store
        self.vector_store = vector_store or VectorStore(data_dir)

        # Set up OpenAI API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

        # Set system prompt
        self.system_prompt = system_prompt or os.environ.get(
            "CODEX_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT
        )

        # Initialize chat history
        self.chat_history = []
        
        # Try loading langchain components
        self._setup_llm()

    def _setup_llm(self) -> None:
        """Set up language model."""
        self.llm = None
        self.llm_type = None

        # First try local model if path is provided
        local_model_path = os.environ.get("LOCAL_MODEL_PATH")
        if local_model_path:
            try:
                # Try to import langchain_community for LlamaCpp
                from langchain_community.llms import LlamaCpp

                # Log the model being used
                logger.info(f"Using local model at: {local_model_path}")

                # Check if running on Jetson
                is_jetson = False
                try:
                    with open("/proc/device-tree/model", "r") as f:
                        model_info = f.read()
                        is_jetson = "NVIDIA Jetson" in model_info
                except:
                    # If we can't read the file, assume not a Jetson
                    pass
                
                # Get GPU acceleration parameters from environment variables
                n_gpu_layers = int(os.environ.get("N_GPU_LAYERS", "0"))
                gpu_layers_draft = int(os.environ.get("GPU_LAYERS_DRAFT", "0"))
                n_batch = int(os.environ.get("N_BATCH", "512"))
                
                # Get context window size from environment variable
                context_window_size = int(os.environ.get("CONTEXT_WINDOW_SIZE", "200000"))
                logger.info(f"Using context window size: {context_window_size}")
                
                # Log GPU acceleration settings
                if n_gpu_layers > 0:
                    logger.info(f"GPU acceleration enabled: {n_gpu_layers} layers")
                    logger.info(f"GPU draft layers: {gpu_layers_draft}")
                    logger.info(f"Batch size: {n_batch}")
                
                # Initialize the local model with Jetson-optimized settings if needed
                if is_jetson:
                    logger.info("Detected Jetson platform, using optimized settings")
                    print("Detected Jetson platform, using optimized settings")
                    
                    # If GPU layers not explicitly set but we're on Jetson, use a default
                    if n_gpu_layers == 0:
                        n_gpu_layers = 24  # Default for Jetson Orin
                        gpu_layers_draft = 24
                        logger.info(f"Using default GPU layers for Jetson: {n_gpu_layers}")
                    
                    # Check if we're using a Hermes model
                    is_hermes_model = "hermes" in local_model_path.lower()
                    if is_hermes_model:
                        logger.info("Detected Hermes model, using ChatML format")
                        print("Detected Hermes model, using ChatML format")
                    
                    self.llm = LlamaCpp(
                        model_path=local_model_path,
                        temperature=0.7,
                        max_tokens=4096,  # Reduced for Jetson
                        n_ctx=min(context_window_size, 100000),  # Use environment variable
                        n_gpu_layers=n_gpu_layers,
                        n_batch=n_batch,
                        n_threads=4,      # Limit threads
                        verbose=True,
                        f16_kv=True,      # Use half precision for key/value cache
                        use_mlock=True,   # Lock memory to prevent swapping
                        seed=42,          # Fixed seed for reproducibility
                        # Add draft parameters for speculative decoding if available
                        **({"gpu_layers_draft": gpu_layers_draft} if gpu_layers_draft > 0 else {})
                    )
                else:
                    # Standard settings for other platforms
                    logger.info("Using standard settings for non-Jetson platforms")
                    print("Using standard settings for non-Jetson platforms")
                    
                    # Check if we're using a Hermes model
                    is_hermes_model = "hermes" in local_model_path.lower()
                    if is_hermes_model:
                        logger.info("Detected Hermes model, using ChatML format")
                        print("Detected Hermes model, using ChatML format")
                    
                    self.llm = LlamaCpp(
                        model_path=local_model_path,
                        temperature=0.7,
                        max_tokens=4096,
                        n_ctx=context_window_size,  # Use environment variable
                        # Only use GPU if explicitly configured
                        **({"n_gpu_layers": n_gpu_layers} if n_gpu_layers > 0 else {}),
                        **({"n_batch": n_batch} if n_batch > 0 else {"n_batch": 512}),
                        verbose=True,
                        f16_kv=True,      # Use half precision for key/value cache
                        use_mlock=True,   # Lock memory to prevent swapping
                        seed=42,          # Fixed seed for reproducibility
                        # Add draft parameters for speculative decoding if available
                        **({"gpu_layers_draft": gpu_layers_draft} if gpu_layers_draft > 0 else {})
                    )
                self.llm_type = "local"
                return
            except (ImportError, Exception) as e:
                logger.warning(
                    f"Failed to initialize local model: {str(e)}. Falling back to OpenAI if available."
                )

        # Try OpenAI if local model is not available or failed
        try:
            # Try to import langchain_openai for ChatOpenAI
            from langchain_openai import ChatOpenAI

            # Check if OpenAI API key is available
            if os.environ.get("OPENAI_API_KEY"):
                logger.info("Using OpenAI API")
                self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
                self.llm_type = "openai"
                return
            else:
                logger.warning("OPENAI_API_KEY not found in environment variables")
        except ImportError:
            logger.warning("langchain_openai not installed")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI: {str(e)}")

        # If we get here, no LLM is available
        logger.warning(
            "No LLM available. Please set LOCAL_MODEL_PATH or OPENAI_API_KEY environment variables."
        )

    def _format_context(self, context_results: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        Format context results for the prompt.

        Args:
            context_results: Dictionary with code and docs results

        Returns:
            Formatted context string
        """
        formatted = []

        # Format code snippets
        if context_results.get("code"):
            formatted.append("## Code Snippets")
            for i, result in enumerate(context_results["code"]):
                metadata = result.get("metadata", {})
                formatted.append(f"### File: {metadata.get('file_path', 'Unknown')}")
                formatted.append(f"Language: {metadata.get('language', 'Unknown')}")
                formatted.append("```")
                formatted.append(result.get("content", "").strip())
                formatted.append("```")
                formatted.append("")

        # Format documentation
        if context_results.get("docs"):
            formatted.append("## Documentation")
            for i, result in enumerate(context_results["docs"]):
                metadata = result.get("metadata", {})
                formatted.append(
                    f"### Document: {metadata.get('file_path', 'Unknown')}"
                )
                formatted.append("")
                formatted.append(result.get("content", "").strip())
                formatted.append("")

        return "\n".join(formatted)

    def _create_prompt(self, query: str, context: str) -> Dict[str, str]:
        """
        Create a prompt for the LLM.

        Args:
            query: User query
            context: Context from vector store

        Returns:
            Prompt dictionary with system and user messages
        """
        return {
            "system": self.system_prompt,
            "user": f"Question: {query}\n\nRelevant information from the codebase:\n{context}",
        }

    def _query_openai(self, prompt: Dict[str, str]) -> str:
        """Query OpenAI API."""
        try:
            from langchain.schema import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=prompt["system"]),
                HumanMessage(content=prompt["user"]),
            ]

            result = self.llm.invoke(messages)
            return result.content
        except Exception as e:
            logger.error(f"Error querying OpenAI: {str(e)}")
            return f"Error generating response: {str(e)}"

    def _query_local(self, prompt: Dict[str, str]) -> str:
        """Query local LLM."""
        try:
            # Check if we're using a Hermes model (based on filename)
            model_path = os.environ.get("LOCAL_MODEL_PATH", "")
            is_hermes_model = "hermes" in model_path.lower()
            is_mistral_small_24b = "mistral-small-24b" in model_path.lower()
            is_phi_4_model = "phi-4" in model_path.lower()
            
            # Check if we should use chat history
            use_chat_history = len(self.chat_history) > 0 and os.environ.get("USE_CHAT_HISTORY", "true").lower() == "true"
            
            if is_mistral_small_24b:
                logger.info("Detected Mistral Small 24B model, using V7-Tekken prompt format")
                
                # V7-Tekken format: <s>[SYSTEM_PROMPT]<system prompt>[/SYSTEM_PROMPT][INST]<user message>[/INST]<assistant response></s>[INST]<user message>[/INST]
                if use_chat_history:
                    # Format prompt with history using V7-Tekken format
                    # Note: We're using a space before <s> to prevent llama-cpp-python from detecting it as a duplicate token
                    formatted_prompt = f" <s>[SYSTEM_PROMPT]{prompt['system']}[/SYSTEM_PROMPT]"
                    
                    # Add first user message and response if available
                    if len(self.chat_history) >= 2:
                        first_user_msg = self.chat_history[0][6:]  # Remove "User: " prefix
                        first_assistant_msg = self.chat_history[1][11:]  # Remove "Assistant: " prefix
                        formatted_prompt += f"[INST]{first_user_msg}[/INST]{first_assistant_msg}"
                        
                        # Add subsequent exchanges
                        for i in range(2, len(self.chat_history), 2):
                            if i+1 < len(self.chat_history):
                                user_msg = self.chat_history[i][6:]
                                assistant_msg = self.chat_history[i+1][11:]
                                formatted_prompt += f"</s>[INST]{user_msg}[/INST]{assistant_msg}"
                    
                    # Add current query
                    formatted_prompt += f"</s>[INST]{prompt['user']}[/INST]"
                else:
                    # Format prompt without history using V7-Tekken format
                    # Note: We're using a space before <s> to prevent llama-cpp-python from detecting it as a duplicate token
                    formatted_prompt = f" <s>[SYSTEM_PROMPT]{prompt['system']}[/SYSTEM_PROMPT][INST]{prompt['user']}[/INST]"
            elif is_hermes_model or is_phi_4_model:
                if is_phi_4_model:
                    logger.info("Detected Phi-4 model, using ChatML format")
                else:
                    logger.info("Detected Hermes model, using ChatML format")
                
                if use_chat_history:
                    # Format prompt using ChatML format with history
                    formatted_prompt = f"""<|im_start|>system<|im_sep|>
{prompt["system"]}<|im_end|>
"""
                    # Add chat history
                    for entry in self.chat_history:
                        if entry.startswith("User: "):
                            formatted_prompt += f"<|im_start|>user<|im_sep|>\n{entry[6:]}<|im_end|>\n"
                        elif entry.startswith("Assistant: "):
                            formatted_prompt += f"<|im_start|>assistant<|im_sep|>\n{entry[11:]}<|im_end|>\n"
                    
                    # Add current query
                    formatted_prompt += f"""<|im_start|>user<|im_sep|>
{prompt["user"]}<|im_end|>
<|im_start|>assistant<|im_sep|>
"""
                else:
                    # Format prompt using ChatML format without history
                    formatted_prompt = f"""<|im_start|>system<|im_sep|>
{prompt["system"]}<|im_end|>
<|im_start|>user<|im_sep|>
{prompt["user"]}<|im_end|>
<|im_start|>assistant<|im_sep|>
"""
            else:
                # Format prompt for standard local LLM
                if use_chat_history:
                    # Format with history
                    formatted_prompt = f"""
System: {prompt["system"]}

"""
                    # Add chat history
                    for entry in self.chat_history:
                        formatted_prompt += f"{entry}\n\n"
                    
                    # Add current query
                    formatted_prompt += f"""User: {prompt["user"]}
"""
                else:
                    # Format without history
                    formatted_prompt = f"""
System: {prompt["system"]}

User: {prompt["user"]}
"""
            
            result = self.llm.invoke(formatted_prompt)

            # Handle both string and object returns
            if isinstance(result, str):
                response = result
            elif hasattr(result, "content"):
                response = result.content
            else:
                # Try to convert to string as fallback
                response = str(result)
                
            # Post-process response for Mistral Small 24B model
            if is_mistral_small_24b:
                logger.debug(f"Raw Mistral Small 24B response: {response}")
                import re
                
                # Check if the response contains the original prompt
                if "[/INST]" in response:
                    # Extract everything after the last [/INST] tag
                    response = response.split("[/INST]")[-1]
                
                # Remove any trailing </s> token
                response = re.sub(r'</s>\s*$', '', response)
                
                # Remove any trailing [INST] token (for next query)
                response = re.sub(r'\[INST\]\s*$', '', response)
                
                # Remove any leading/trailing whitespace
                response = response.strip()
                
                logger.debug(f"Cleaned Mistral Small 24B response: {response}")
            
            # Post-process response for Phi-4 model
            elif is_phi_4_model:
                logger.debug(f"Raw Phi-4 response: {response}")
                import re
                
                # Check if the response contains ChatML tags
                if "<|im_end|>" in response:
                    # Extract content between assistant<|im_sep|> and <|im_end|>
                    match = re.search(r'<\|im_start\|>assistant<\|im_sep\|>(.*?)<\|im_end\|>', response, re.DOTALL)
                    if match:
                        response = match.group(1).strip()
                    else:
                        # If no match, just remove all ChatML tags
                        response = re.sub(r'<\|im_(start|sep|end)\|>', '', response)
                
                # Remove any leading/trailing whitespace
                response = response.strip()
                
                logger.debug(f"Cleaned Phi-4 response: {response}")
                
            # Update chat history if enabled
            if os.environ.get("USE_CHAT_HISTORY", "true").lower() == "true":
                # Limit history to last 10 exchanges to prevent context overflow
                max_history = int(os.environ.get("MAX_CHAT_HISTORY", "10"))
                
                # Add current exchange
                self.chat_history.append(f"User: {prompt['user']}")
                self.chat_history.append(f"Assistant: {response}")
                
                # Trim history if needed
                if len(self.chat_history) > max_history * 2:
                    # Keep the most recent exchanges
                    self.chat_history = self.chat_history[-max_history * 2:]
            
            return response

        except Exception as e:
            logger.error(f"Error querying local LLM: {str(e)}")
            return f"Error generating response: {str(e)}"

    def query(self, query_text: str, code_limit: int = 3, doc_limit: int = 3) -> str:
        """
        Query the LLM with a given user query.

        Args:
            query_text: User query
            code_limit: Maximum number of code contexts to retrieve
            doc_limit: Maximum number of documentation contexts to retrieve

        Returns:
            Response from the LLM
        """
        # Check if LLM is available
        if not self.llm:
            return (
                "No LLM available for querying. Please set either LOCAL_MODEL_PATH or OPENAI_API_KEY in your environment.\n\n"
                "For local models, please run the download_model.py script first:\n"
                "poetry run python download_model.py\n\n"
                f"Query: {query_text}\n\n"
                "Context retrieval still works, but no response generation is possible without an LLM."
            )

        # Retrieve context from vector store
        context_results = self.vector_store.search_all(
            query_text, code_limit=code_limit, doc_limit=doc_limit
        )

        # Format context
        context = self._format_context(context_results)

        # Create prompt for the LLM
        prompt = self._create_prompt(query_text, context)

        # Query the LLM
        if self.llm_type == "openai":
            return self._query_openai(prompt)
        elif self.llm_type == "local":
            return self._query_local(prompt)
        else:
            # This should never happen because we check for self.llm above
            return (
                "Unknown LLM type. Please check your configuration.\n\n"
                f"Query: {query_text}\n\n"
                f"Retrieved context:\n{context}"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the query engine."""
        vs_stats = self.vector_store.get_stats()

        return {
            **vs_stats,
            "llm_type": self.llm_type,
            "llm_model": getattr(self.llm, "model_name", "None")
            if self.llm
            else "None",
        }
