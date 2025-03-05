"""
Documentation Parser - Handles parsing of documentation files.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set


# Optional imports - will be imported when needed
def import_doc_parsers():
    global BeautifulSoup, pypdf, docx2txt
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        BeautifulSoup = None
    try:
        import pypdf
    except ImportError:
        pypdf = None
    try:
        import docx2txt
    except ImportError:
        docx2txt = None


logger = logging.getLogger(__name__)

# Documentation file extensions to process
DOC_EXTENSIONS = {
    # Markdown
    ".md": "markdown",
    ".markdown": "markdown",
    # ReStructuredText
    ".rst": "rst",
    # HTML
    ".html": "html",
    ".htm": "html",
    # Text
    ".txt": "text",
    # PDF
    ".pdf": "pdf",
    # Office documents
    ".docx": "docx",
    ".doc": "doc",
    # Other
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
    ".csv": "csv",
}

# Files to ignore
IGNORE_FILES = {
    ".git",
    ".github",
    "node_modules",
    "__pycache__",
    "venv",
    "env",
    "build",
    "dist",
}


class DocParser:
    """Parse documentation files and extract relevant information."""

    def __init__(
        self,
        root_dir: str,
        ignore_dirs: Optional[Set[str]] = None,
        file_extensions: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the documentation parser.

        Args:
            root_dir: Root directory of the documentation
            ignore_dirs: Directories to ignore
            file_extensions: File extensions to process with their format
        """
        self.root_dir = Path(root_dir)
        self.ignore_dirs = ignore_dirs or IGNORE_FILES
        self.file_extensions = file_extensions or DOC_EXTENSIONS

        # Import optional dependencies
        import_doc_parsers()

    def is_ignored(self, path: Path) -> bool:
        """Check if a path should be ignored."""
        for ignore_pattern in self.ignore_dirs:
            if ignore_pattern in path.parts:
                return True
        return False

    def get_doc_format(self, file_path: Path) -> Optional[str]:
        """Get the document format based on file extension."""
        extension = file_path.suffix.lower()
        return self.file_extensions.get(extension)

    def find_doc_files(self) -> List[Dict[str, Any]]:
        """
        Find all documentation files in the root directory.

        Returns:
            List of dictionaries containing file_path and doc_format
        """
        doc_files = []

        for path in self.root_dir.rglob("*"):
            if path.is_file() and not self.is_ignored(path):
                doc_format = self.get_doc_format(path)
                if doc_format:
                    doc_files.append(
                        {
                            "file_path": path,
                            "doc_format": doc_format,
                            "relative_path": str(path.relative_to(self.root_dir)),
                        }
                    )

        logger.info(f"Found {len(doc_files)} documentation files")
        return doc_files

    def parse_markdown(self, content: str) -> str:
        """Parse Markdown content."""
        # For now, we just return the content directly
        # In the future, we could parse the markdown structure
        return content

    def parse_rst(self, content: str) -> str:
        """Parse ReStructuredText content."""
        # Simple parsing for now
        return content

    def parse_html(self, content: str) -> str:
        """Parse HTML content."""
        if BeautifulSoup:
            try:
                soup = BeautifulSoup(content, "html.parser")
                # Extract text without HTML tags
                return soup.get_text(separator=" ", strip=True)
            except Exception as e:
                logger.error(f"Error parsing HTML: {str(e)}")
                return content
        else:
            logger.warning(
                "BeautifulSoup not available. Install with 'pip install beautifulsoup4'"
            )
            # Simple HTML tag removal if BeautifulSoup is not available
            return re.sub(r"<[^>]*>", "", content)

    def parse_pdf(self, file_path: Path) -> str:
        """Parse PDF content."""
        if pypdf:
            try:
                text = ""
                with open(file_path, "rb") as file:
                    reader = pypdf.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() + "\n\n"
                return text
            except Exception as e:
                logger.error(f"Error parsing PDF: {str(e)}")
                return f"Error parsing PDF: {str(e)}"
        else:
            logger.warning("pypdf not available. Install with 'pip install pypdf'")
            return "PDF parsing not available. Install pypdf."

    def parse_docx(self, file_path: Path) -> str:
        """Parse DOCX content."""
        if docx2txt:
            try:
                return docx2txt.process(file_path)
            except Exception as e:
                logger.error(f"Error parsing DOCX: {str(e)}")
                return f"Error parsing DOCX: {str(e)}"
        else:
            logger.warning(
                "docx2txt not available. Install with 'pip install docx2txt'"
            )
            return "DOCX parsing not available. Install docx2txt."

    def parse_file(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a documentation file and extract content.

        Args:
            file_info: Dictionary with file_path and doc_format

        Returns:
            Dictionary containing file metadata and content
        """
        file_path = file_info["file_path"]
        doc_format = file_info["doc_format"]

        try:
            content = ""
            if doc_format in ["pdf"]:
                # Binary formats require special handling
                content = self.parse_pdf(file_path)
            elif doc_format in ["docx"]:
                content = self.parse_docx(file_path)
            else:
                # Text-based formats
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    file_content = f.read()

                if doc_format == "markdown":
                    content = self.parse_markdown(file_content)
                elif doc_format == "rst":
                    content = self.parse_rst(file_content)
                elif doc_format == "html":
                    content = self.parse_html(file_content)
                else:
                    # Default: just use the content as is
                    content = file_content

            # Extract metadata
            result = {
                "path": file_info["relative_path"],
                "format": doc_format,
                "content": content,
                "size": len(content),
                "line_count": content.count("\n") + 1,
            }

            return result
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
            return {
                "path": file_info["relative_path"],
                "format": doc_format,
                "content": f"Error parsing file: {str(e)}",
                "error": str(e),
            }

    def chunk_text(
        self, content: str, chunk_size: int = 1500, overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks for processing.

        Args:
            content: Content of the documentation file
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters

        Returns:
            List of dictionaries with chunk info
        """
        # Handle empty or very small content
        if not content or len(content) <= chunk_size:
            return [
                {
                    "content": content,
                    "size": len(content),
                    "line_count": content.count("\n") + 1 if content else 0,
                }
            ]

        # Split by paragraphs first (double newlines)
        paragraphs = content.split("\n\n")

        chunks = []
        current_chunk = []
        current_size = 0

        # Safety check for very long paragraphs
        max_paragraph_length = max(len(p) for p in paragraphs) if paragraphs else 0
        if max_paragraph_length > chunk_size:
            logger.warning(
                f"File contains paragraphs longer than chunk_size ({max_paragraph_length} > {chunk_size}). "
                f"Will split paragraphs as needed."
            )

        for paragraph in paragraphs:
            paragraph_size = len(paragraph) + 2  # +2 for the "\n\n"

            # If this paragraph alone exceeds chunk size, split it further
            if paragraph_size > chunk_size:
                # If current chunk has content, add it to chunks
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(
                        {
                            "content": chunk_text,
                            "size": current_size,
                            "line_count": chunk_text.count("\n") + 1,
                        }
                    )
                    current_chunk = []
                    current_size = 0

                # Split large paragraph by sentences or just characters if needed
                sentences = paragraph.replace(". ", ".\n").split("\n")
                sentence_chunk = []
                sentence_size = 0

                for sentence in sentences:
                    sentence_len = len(sentence) + 1  # +1 for newline

                    if sentence_len > chunk_size:
                        # This single sentence is too long, split by characters
                        if sentence_chunk:
                            # Add accumulated sentences first
                            sentence_text = " ".join(sentence_chunk)
                            chunks.append(
                                {
                                    "content": sentence_text,
                                    "size": sentence_size,
                                    "line_count": sentence_text.count("\n") + 1,
                                }
                            )
                            sentence_chunk = []
                            sentence_size = 0

                        # Split the long sentence into fixed-size chunks
                        for i in range(0, len(sentence), chunk_size - 100):
                            chunk_text = sentence[i : i + chunk_size - 100]
                            chunks.append(
                                {
                                    "content": chunk_text,
                                    "size": len(chunk_text),
                                    "line_count": chunk_text.count("\n") + 1,
                                }
                            )
                    elif sentence_size + sentence_len > chunk_size and sentence_chunk:
                        # This sentence would make the chunk too big
                        sentence_text = " ".join(sentence_chunk)
                        chunks.append(
                            {
                                "content": sentence_text,
                                "size": sentence_size,
                                "line_count": sentence_text.count("\n") + 1,
                            }
                        )
                        sentence_chunk = [sentence]
                        sentence_size = sentence_len
                    else:
                        # Add to current sentence chunk
                        sentence_chunk.append(sentence)
                        sentence_size += sentence_len

                # Add any remaining sentences
                if sentence_chunk:
                    sentence_text = " ".join(sentence_chunk)
                    chunks.append(
                        {
                            "content": sentence_text,
                            "size": sentence_size,
                            "line_count": sentence_text.count("\n") + 1,
                        }
                    )

            # Normal case: paragraph fits in a chunk
            elif current_size + paragraph_size > chunk_size and current_chunk:
                # This paragraph would make the chunk too big
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(
                    {
                        "content": chunk_text,
                        "size": current_size,
                        "line_count": chunk_text.count("\n") + 1,
                    }
                )

                # Start a new chunk with overlap
                overlap_size = 0
                overlap_paragraphs = []

                # Add paragraphs from the end until we reach desired overlap
                for p in reversed(current_chunk):
                    p_size = len(p) + 2  # +2 for "\n\n"
                    if overlap_size + p_size <= overlap:
                        overlap_paragraphs.insert(0, p)
                        overlap_size += p_size
                    else:
                        break

                current_chunk = overlap_paragraphs + [paragraph]
                current_size = overlap_size + paragraph_size
            else:
                # Add to current chunk
                current_chunk.append(paragraph)
                current_size += paragraph_size

        # Add the last chunk if it's not empty
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(
                {
                    "content": chunk_text,
                    "size": current_size,
                    "line_count": chunk_text.count("\n") + 1,
                }
            )

        # Ensure we have at least one chunk
        if not chunks:
            chunks.append(
                {
                    "content": "",
                    "size": 0,
                    "line_count": 0,
                }
            )

        return chunks

    def process_docs(
        self, chunk_size: int = 1500, overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Process all documentation files.

        Args:
            chunk_size: Maximum chunk size for splitting text
            overlap: Overlap between chunks

        Returns:
            List of dictionaries with processed doc chunks
        """
        doc_files = self.find_doc_files()
        all_chunks = []

        for file_info in doc_files:
            file_data = self.parse_file(file_info)

            if "error" in file_data:
                # Skip files with errors
                continue

            # Get file content and metadata
            file_chunks = self.chunk_text(file_data["content"], chunk_size, overlap)

            # Add file metadata to each chunk
            for i, chunk in enumerate(file_chunks):
                chunk["file_path"] = file_data["path"]
                chunk["format"] = file_data["format"]
                chunk["chunk_index"] = i
                chunk["total_chunks"] = len(file_chunks)
                all_chunks.append(chunk)

        logger.info(
            f"Generated {len(all_chunks)} chunks from {len(doc_files)} documentation files"
        )
        return all_chunks
