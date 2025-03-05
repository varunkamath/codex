"""
Vector Store - Handles storing and retrieving embeddings for code and documentation chunks.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

# Set environment variables to disable CPU affinity for ONNX runtime
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["OMP_PROC_BIND"] = "FALSE"
os.environ["ONNXRUNTIME_DISABLE_CPU_AFFINITY"] = "1"

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
            import logging

            # Use a more compatible model configuration for Jetson
            try:
                # First try with a smaller, more stable model for Jetson
                model = SentenceTransformer(
                    "paraphrase-MiniLM-L3-v2",  # Smaller model that's more stable
                    device="cpu",
                    quantize=True
                )
                logger.info("Loaded smaller embedding model (paraphrase-MiniLM-L3-v2) with quantization")
            except Exception as e:
                logging.warning(f"Error loading smaller model: {e}")
                try:
                    # Fallback to standard model with explicit settings
                    model = SentenceTransformer(
                        "all-MiniLM-L6-v2", 
                        device="cpu",
                        quantize=True
                    )
                    logger.info("Loaded standard embedding model with quantization")
                except Exception as e:
                    logging.warning(f"Error loading model with custom settings: {e}")
                    # Fallback to basic initialization
                    model = SentenceTransformer("all-MiniLM-L6-v2")
                    logger.info("Loaded embedding model with default settings")

            def get_embedding(texts, model_name=None):
                # Process in extremely small batches to avoid memory issues
                batch_size = 1  # Process one at a time for maximum stability
                all_embeddings = []
                
                # Handle empty input case
                if not texts:
                    logger.warning("Empty texts provided to get_embedding")
                    return []
                
                try:
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i+batch_size]
                        try:
                            # Add explicit error handling for each batch
                            logger.debug(f"Encoding batch {i//batch_size} of {len(texts)//batch_size if len(texts) > 0 else 0}")
                            batch_embeddings = model.encode(batch, show_progress_bar=False).tolist()
                            all_embeddings.extend(batch_embeddings)
                            logger.debug(f"Successfully encoded batch {i//batch_size}")
                        except Exception as e:
                            logger.error(f"Error encoding batch {i//batch_size}: {str(e)}")
                            # Add zero embeddings as fallback for failed batches
                            # Get embedding dimension from model
                            if all_embeddings:
                                dim = len(all_embeddings[0])
                            else:
                                # Default dimensions for different models
                                if "L3-v2" in str(model):
                                    dim = 384  # Default dimension for paraphrase-MiniLM-L3-v2
                                else:
                                    dim = 384  # Default dimension for all-MiniLM-L6-v2
                            
                            # Add zero embeddings for each text in the failed batch
                            for _ in batch:
                                all_embeddings.append([0.0] * dim)
                    
                    return all_embeddings
                except Exception as e:
                    logger.error(f"Error in embedding generation: {str(e)}")
                    # Return zero embeddings as fallback
                    dim = 384  # Default dimension for embedding models
                    return [[0.0] * dim for _ in texts]
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

        # Initialize ChromaDB client with new API format
        try:
            # Configure ChromaDB with settings optimized for Jetson
            self.client = chromadb.PersistentClient(
                path=str(self.chroma_dir),
            )
            logger.info(f"ChromaDB initialized at {self.chroma_dir}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise

        # Initialize collections with HNSW parameters for better recall
        try:
            self.code_collection = self.client.get_or_create_collection(
                name="code_chunks",
                metadata={"hnsw:search_ef": 50}  # Increase search_ef for better recall
            )
            self.doc_collection = self.client.get_or_create_collection(
                name="doc_chunks",
                metadata={"hnsw:search_ef": 50}  # Increase search_ef for better recall
            )
        except Exception as e:
            logger.error(f"Error creating collections: {e}")
            raise

        # Initialize metadata
        self.metadata = self._load_metadata()

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

        # Process in smaller batches to avoid memory issues on Jetson
        batch_size = 4  # Very small batch size for Jetson
        total_added = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []

            for chunk in batch:
                chunk_id = f"code_{chunk['file_path']}_{chunk['chunk_index']}"

                # Track file in metadata
                file_path = chunk["file_path"]
                if file_path not in self.metadata["code_chunks"]["files"]:
                    self.metadata["code_chunks"]["files"][file_path] = {
                        "language": chunk["language"],
                        "chunks": 0,
                    }

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
                # Try adding one by one for maximum stability
                for j in range(len(batch)):
                    try:
                        self.code_collection.add(
                            ids=[ids[j]],
                            documents=[documents[j]],
                            metadatas=[metadatas[j]]
                        )
                        file_path = batch[j]["file_path"]
                        self.metadata["code_chunks"]["files"][file_path]["chunks"] += 1
                        total_added += 1
                        logger.debug(f"Added individual code chunk {j} from batch {i//batch_size}")
                    except Exception as e2:
                        logger.error(f"Error adding individual code chunk {j} from batch {i//batch_size}: {str(e2)}")
                
                logger.info(f"Processed batch {i//batch_size} with {len(batch)} code chunks, added {total_added} chunks")
            except Exception as e:
                logger.error(f"Error in batch processing for code chunks batch {i//batch_size}: {str(e)}")
                
        # Update metadata
        self.metadata["code_chunks"]["count"] = total_added  # Set exact count
        self._save_metadata()

        return total_added

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

        # Process in smaller batches to avoid memory issues on Jetson
        batch_size = 4  # Very small batch size for Jetson
        total_added = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []

            for chunk in batch:
                chunk_id = f"doc_{chunk['file_path']}_{chunk['chunk_index']}"

                # Track file in metadata
                file_path = chunk["file_path"]
                if file_path not in self.metadata["doc_chunks"]["files"]:
                    self.metadata["doc_chunks"]["files"][file_path] = {
                        "format": chunk["format"],
                        "chunks": 0,
                    }

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
                # Try adding one by one for maximum stability
                for j in range(len(batch)):
                    try:
                        self.doc_collection.add(
                            ids=[ids[j]],
                            documents=[documents[j]],
                            metadatas=[metadatas[j]]
                        )
                        file_path = batch[j]["file_path"]
                        self.metadata["doc_chunks"]["files"][file_path]["chunks"] += 1
                        total_added += 1
                        logger.debug(f"Added individual doc chunk {j} from batch {i//batch_size}")
                    except Exception as e2:
                        logger.error(f"Error adding individual doc chunk {j} from batch {i//batch_size}: {str(e2)}")
                
                logger.info(f"Processed batch {i//batch_size} with {len(batch)} doc chunks, added {total_added} chunks")
            except Exception as e:
                logger.error(f"Error in batch processing for doc chunks batch {i//batch_size}: {str(e)}")
                
        # Update metadata
        self.metadata["doc_chunks"]["count"] = total_added  # Set exact count
        self._save_metadata()

        return total_added

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
            # Use a higher n_results value to improve recall
            actual_limit = max(limit * 3, 20)  # Request more results than needed
            logger.debug(f"Searching code with query: {query}, requesting {actual_limit} results")
            
            results = self.code_collection.query(
                query_texts=[query], 
                n_results=actual_limit,
                include=["documents", "metadatas", "distances"]
            )

            formatted_results = []
            if results and results["documents"]:
                # Sort by distance to ensure we get the most relevant results
                combined_results = []
                for i, doc in enumerate(results["documents"][0]):
                    combined_results.append({
                        "content": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "id": results["ids"][0][i] if results["ids"] else f"result_{i}",
                        "distance": results["distances"][0][i] if "distances" in results and results["distances"] else 1.0
                    })
                
                # Sort by distance (lower is better)
                combined_results.sort(key=lambda x: x["distance"])
                
                # Take only the requested number of results
                formatted_results = combined_results[:limit]
                
                # Remove the distance field from the final results
                for result in formatted_results:
                    result.pop("distance", None)
                
                logger.debug(f"Found {len(formatted_results)} relevant code chunks")
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
            # Use a higher n_results value to improve recall
            actual_limit = max(limit * 3, 20)  # Request more results than needed
            logger.debug(f"Searching docs with query: {query}, requesting {actual_limit} results")
            
            results = self.doc_collection.query(
                query_texts=[query], 
                n_results=actual_limit,
                include=["documents", "metadatas", "distances"]
            )

            formatted_results = []
            if results and results["documents"]:
                # Sort by distance to ensure we get the most relevant results
                combined_results = []
                for i, doc in enumerate(results["documents"][0]):
                    combined_results.append({
                        "content": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "id": results["ids"][0][i] if results["ids"] else f"result_{i}",
                        "distance": results["distances"][0][i] if "distances" in results and results["distances"] else 1.0
                    })
                
                # Sort by distance (lower is better)
                combined_results.sort(key=lambda x: x["distance"])
                
                # Take only the requested number of results
                formatted_results = combined_results[:limit]
                
                # Remove the distance field from the final results
                for result in formatted_results:
                    result.pop("distance", None)
                
                logger.debug(f"Found {len(formatted_results)} relevant doc chunks")
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
