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
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

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

                # Initialize the local model
                self.llm = LlamaCpp(
                    model_path=local_model_path,
                    temperature=0.7,
                    max_tokens=2000,
                    n_ctx=4096,
                    verbose=False,
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
            # Format prompt for local LLM
            formatted_prompt = f"""
System: {prompt["system"]}

User: {prompt["user"]}
"""
            result = self.llm.invoke(formatted_prompt)
            return result.content
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
