"""
Code Parser - Handles parsing of code files and extraction of relevant information.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# File extensions to process
CODE_EXTENSIONS = {
    # Python
    ".py": "python",
    # JavaScript/TypeScript
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    # Web
    ".html": "html",
    ".css": "css",
    # Other common languages
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".rs": "rust",
}

# Files to ignore
IGNORE_FILES = {
    ".git",
    ".idea",
    ".vscode",
    "node_modules",
    "__pycache__",
    "venv",
    "env",
    "build",
    "dist",
    ".pytest_cache",
    ".mypy_cache",
}


class CodeParser:
    """Parse code files and extract relevant information."""

    def __init__(
        self,
        root_dir: str,
        ignore_dirs: Optional[Set[str]] = None,
        file_extensions: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the code parser.

        Args:
            root_dir: Root directory of the codebase
            ignore_dirs: Directories to ignore
            file_extensions: File extensions to process with their language
        """
        self.root_dir = Path(root_dir)
        self.ignore_dirs = ignore_dirs or IGNORE_FILES
        self.file_extensions = file_extensions or CODE_EXTENSIONS

    def is_ignored(self, path: Path) -> bool:
        """Check if a path should be ignored."""
        for ignore_pattern in self.ignore_dirs:
            if ignore_pattern in path.parts:
                return True
        return False

    def get_file_language(self, file_path: Path) -> Optional[str]:
        """Get the programming language based on file extension."""
        extension = file_path.suffix.lower()
        return self.file_extensions.get(extension)

    def find_code_files(self) -> List[Tuple[Path, str]]:
        """
        Find all code files in the root directory.

        Returns:
            List of tuples containing (file_path, language)
        """
        code_files = []

        for path in self.root_dir.rglob("*"):
            if path.is_file() and not self.is_ignored(path):
                language = self.get_file_language(path)
                if language:
                    code_files.append((path, language))

        logger.info(f"Found {len(code_files)} code files")
        return code_files

    def parse_file(self, file_path: Path, language: str) -> Dict[str, Any]:
        """
        Parse a code file and extract metadata and content.

        Args:
            file_path: Path to the file
            language: Programming language of the file

        Returns:
            Dictionary containing file metadata and content
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            relative_path = file_path.relative_to(self.root_dir)

            # Basic metadata
            result = {
                "path": str(relative_path),
                "language": language,
                "content": content,
                "size": len(content),
                "line_count": content.count("\n") + 1,
            }

            return result
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
            return {
                "path": str(file_path.relative_to(self.root_dir)),
                "language": language,
                "content": f"Error parsing file: {str(e)}",
                "error": str(e),
            }

    def chunk_code(
        self,
        code_content: str,
        language: str,
        chunk_size: int = 1000,
        overlap: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Split code into overlapping chunks for processing.

        Args:
            code_content: Content of the code file
            language: Programming language
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters

        Returns:
            List of dictionaries with chunk info
        """
        # Simple line-based chunking for now
        lines = code_content.split("\n")
        chunks = []
        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line) + 1  # +1 for newline

            # If adding this line would exceed chunk size, create a new chunk
            if current_size + line_size > chunk_size and current_chunk:
                chunk_text = "\n".join(current_chunk)
                chunks.append(
                    {
                        "content": chunk_text,
                        "language": language,
                        "size": current_size,
                        "line_count": len(current_chunk),
                    }
                )

                # Keep last few lines for overlap
                overlap_lines = []
                overlap_size = 0
                for line in reversed(current_chunk):
                    if overlap_size + len(line) + 1 <= overlap:
                        overlap_lines.insert(0, line)
                        overlap_size += len(line) + 1
                    else:
                        break

                current_chunk = overlap_lines
                current_size = overlap_size

            current_chunk.append(line)
            current_size += line_size

        # Add the last chunk if it's not empty
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunks.append(
                {
                    "content": chunk_text,
                    "language": language,
                    "size": current_size,
                    "line_count": len(current_chunk),
                }
            )

        return chunks

    def process_codebase(
        self, chunk_size: int = 1000, overlap: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Process the entire codebase.

        Args:
            chunk_size: Maximum chunk size for splitting code
            overlap: Overlap between chunks

        Returns:
            List of dictionaries with processed code chunks
        """
        code_files = self.find_code_files()
        all_chunks = []

        for file_path, language in code_files:
            file_data = self.parse_file(file_path, language)

            if "error" in file_data:
                # Skip files with errors
                continue

            # Get file content and metadata
            file_chunks = self.chunk_code(
                file_data["content"], language, chunk_size, overlap
            )

            # Add file metadata to each chunk
            for i, chunk in enumerate(file_chunks):
                chunk["file_path"] = file_data["path"]
                chunk["chunk_index"] = i
                chunk["total_chunks"] = len(file_chunks)
                all_chunks.append(chunk)

        logger.info(f"Generated {len(all_chunks)} chunks from {len(code_files)} files")
        return all_chunks
