"""
Basic tests for Codex functionality.
"""

import os
import sys
import unittest
from pathlib import Path

# Add parent directory to path to import codex
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from codex.ingestion.code_parser import CodeParser
from codex.ingestion.doc_parser import DocParser


class TestCodeParser(unittest.TestCase):
    """Test code parser functionality."""

    def test_code_extensions(self):
        """Test that code extensions are correctly recognized."""
        parser = CodeParser(".")

        # Python file
        self.assertEqual(parser.get_file_language(Path("test.py")), "python")

        # JavaScript file
        self.assertEqual(parser.get_file_language(Path("test.js")), "javascript")

        # Unknown extension
        self.assertIsNone(parser.get_file_language(Path("test.unknown")))

    def test_is_ignored(self):
        """Test that ignored directories are correctly identified."""
        parser = CodeParser(".")

        # Ignored directory
        self.assertTrue(parser.is_ignored(Path("venv/test.py")))

        # Non-ignored directory
        self.assertFalse(parser.is_ignored(Path("src/test.py")))


class TestDocParser(unittest.TestCase):
    """Test documentation parser functionality."""

    def test_doc_extensions(self):
        """Test that documentation extensions are correctly recognized."""
        parser = DocParser(".")

        # Markdown file
        self.assertEqual(parser.get_doc_format(Path("test.md")), "markdown")

        # HTML file
        self.assertEqual(parser.get_doc_format(Path("test.html")), "html")

        # Unknown extension
        self.assertIsNone(parser.get_doc_format(Path("test.unknown")))

    def test_is_ignored(self):
        """Test that ignored directories are correctly identified."""
        parser = DocParser(".")

        # Ignored directory
        self.assertTrue(parser.is_ignored(Path(".git/README.md")))

        # Non-ignored directory
        self.assertFalse(parser.is_ignored(Path("docs/README.md")))


if __name__ == "__main__":
    unittest.main()
