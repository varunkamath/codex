"""
Vector Store - Handles storing and retrieving embeddings for code and documentation chunks.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any


# Import optional dependencies at runtime
def import_vector_store_deps():
    global chromadb, get_embedding
    try:
        import chromadb
    except ImportError:
        chromadb = None

    # Import embedding function - try multiple options
    try:
        from langchain_openai import OpenAIEmbeddings

        def get_embedding(texts, model="text-embedding-3-small"):
            client = OpenAIEmbeddings(model=model)
            return client.embed_documents(texts)
    except ImportError:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")

            def get_embedding(texts, model_name=None):
                return model.encode(texts).tolist()
        except ImportError:

            def get_embedding(texts, model=None):
                raise ImportError(
                    "No embedding library found. Install either langchain_openai or sentence_transformers."
                )


logger = logging.getLogger(__name__)


class VectorStore:
    """Store and retrieve code and documentation embeddings."""

    def __init__(self, storage_dir: str = ".codex_data"):
        """
        Initialize the vector store.

        Args:
            storage_dir: Directory to store the vector database
        """
        self.storage_dir = Path(storage_dir)
        self.chroma_dir = self.storage_dir / "chroma"
        self.metadata_path = self.storage_dir / "metadata.json"

        # Create directories if they don't exist
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        self.chroma_dir.mkdir(exist_ok=True, parents=True)

        # Import dependencies
        import_vector_store_deps()

        # Check if ChromaDB is available
        if chromadb is None:
            logger.error("ChromaDB not available. Install with 'pip install chromadb'")
            raise ImportError(
                "ChromaDB not available. Install with 'pip install chromadb'"
            )

        # Initialize ChromaDB client - Using new client construction method
        # The old method using persistence_directory is deprecated
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))

        # Initialize collections
        self.code_collection = self._get_or_create_collection("code_chunks")
        self.doc_collection = self._get_or_create_collection("doc_chunks")

        # Initialize metadata
        self.metadata = self._load_metadata()

    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection."""
        try:
            return self.client.get_collection(name=name)
        except:  # noqa: E722
            return self.client.create_collection(name=name)

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file or create default."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {str(e)}")
                return self._create_default_metadata()
        else:
            return self._create_default_metadata()

    def _create_default_metadata(self) -> Dict[str, Any]:
        """Create default metadata structure."""
        return {
            "code_chunks": {"count": 0, "files": {}},
            "doc_chunks": {"count": 0, "files": {}},
            "last_updated": "",
            "root_directory": "",
        }

    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def add_code_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Add code chunks to the vector store.

        Args:
            chunks: List of code chunk dictionaries

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"code_{chunk['file_path']}_{chunk['chunk_index']}"

            # Track file in metadata
            file_path = chunk["file_path"]
            if file_path not in self.metadata["code_chunks"]["files"]:
                self.metadata["code_chunks"]["files"][file_path] = {
                    "language": chunk["language"],
                    "chunks": 0,
                }

            self.metadata["code_chunks"]["files"][file_path]["chunks"] += 1

            # Prepare for batch addition
            ids.append(chunk_id)
            documents.append(chunk["content"])
            metadatas.append(
                {
                    "file_path": chunk["file_path"],
                    "language": chunk["language"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk["total_chunks"],
                    "size": chunk["size"],
                    "line_count": chunk.get("line_count", 0),
                }
            )

        # Generate embeddings and add to collection
        try:
            self.code_collection.add(ids=ids, documents=documents, metadatas=metadatas)

            # Update metadata
            self.metadata["code_chunks"]["count"] += len(chunks)
            self._save_metadata()

            return len(chunks)
        except Exception as e:
            logger.error(f"Error adding code chunks: {str(e)}")
            return 0

    def add_doc_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Add documentation chunks to the vector store.

        Args:
            chunks: List of documentation chunk dictionaries

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"doc_{chunk['file_path']}_{chunk['chunk_index']}"

            # Track file in metadata
            file_path = chunk["file_path"]
            if file_path not in self.metadata["doc_chunks"]["files"]:
                self.metadata["doc_chunks"]["files"][file_path] = {
                    "format": chunk["format"],
                    "chunks": 0,
                }

            self.metadata["doc_chunks"]["files"][file_path]["chunks"] += 1

            # Prepare for batch addition
            ids.append(chunk_id)
            documents.append(chunk["content"])
            metadatas.append(
                {
                    "file_path": chunk["file_path"],
                    "format": chunk["format"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk["total_chunks"],
                    "size": chunk["size"],
                }
            )

        # Generate embeddings and add to collection
        try:
            self.doc_collection.add(ids=ids, documents=documents, metadatas=metadatas)

            # Update metadata
            self.metadata["doc_chunks"]["count"] += len(chunks)
            self._save_metadata()

            return len(chunks)
        except Exception as e:
            logger.error(f"Error adding doc chunks: {str(e)}")
            return 0

    def search_code(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for code chunks relevant to a query.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of relevant code chunks with metadata
        """
        try:
            results = self.code_collection.query(query_texts=[query], n_results=limit)

            formatted_results = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    formatted_results.append(
                        {
                            "content": doc,
                            "metadata": results["metadatas"][0][i]
                            if results["metadatas"]
                            else {},
                            "id": results["ids"][0][i]
                            if results["ids"]
                            else f"result_{i}",
                        }
                    )

            return formatted_results
        except Exception as e:
            logger.error(f"Error searching code: {str(e)}")
            return []

    def search_docs(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documentation chunks relevant to a query.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of relevant documentation chunks with metadata
        """
        try:
            results = self.doc_collection.query(query_texts=[query], n_results=limit)

            formatted_results = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    formatted_results.append(
                        {
                            "content": doc,
                            "metadata": results["metadatas"][0][i]
                            if results["metadatas"]
                            else {},
                            "id": results["ids"][0][i]
                            if results["ids"]
                            else f"result_{i}",
                        }
                    )

            return formatted_results
        except Exception as e:
            logger.error(f"Error searching docs: {str(e)}")
            return []

    def search_all(
        self, query: str, code_limit: int = 3, doc_limit: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for both code and documentation chunks relevant to a query.

        Args:
            query: Search query
            code_limit: Maximum number of code results
            doc_limit: Maximum number of documentation results

        Returns:
            Dictionary with code and documentation results
        """
        code_results = self.search_code(query, code_limit)
        doc_results = self.search_docs(query, doc_limit)

        return {"code": code_results, "docs": doc_results}

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "code_chunks": self.metadata["code_chunks"]["count"],
            "doc_chunks": self.metadata["doc_chunks"]["count"],
            "code_files": len(self.metadata["code_chunks"]["files"]),
            "doc_files": len(self.metadata["doc_chunks"]["files"]),
            "last_updated": self.metadata.get("last_updated", "Never"),
            "storage_path": str(self.storage_dir),
        }
