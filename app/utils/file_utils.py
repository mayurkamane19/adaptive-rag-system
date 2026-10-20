"""File utility functions for file path management, validation, and naming."""

import re
from pathlib import Path
from typing import Set, Union


ALLOWED_DOCUMENT_EXTENSIONS: Set[str] = {".pdf", ".txt", ".docx", ".md"}


def ensure_directory_exists(directory_path: Union[str, Path]) -> Path:
    """Ensures that a specified directory exists, creating it if necessary."""
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_allowed_file_extension(
    filename: str, allowed_extensions: Set[str] = ALLOWED_DOCUMENT_EXTENSIONS
) -> bool:
    """Checks if the file extension is within the allowed set."""
    ext = Path(filename).suffix.lower()
    return ext in allowed_extensions


def generate_safe_filename(original_filename: str) -> str:
    """Generates a safe filename by stripping unsafe characters and path traversal risks."""
    filename = Path(original_filename).name
    safe_name = re.sub(r"[^\w\.-]", "_", filename)
    return safe_name


def file_exists(file_path: Union[str, Path]) -> bool:
    """Checks if a file exists on disk."""
    return Path(file_path).is_file()
