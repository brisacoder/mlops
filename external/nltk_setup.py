#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nltk_setup.py
=============

Simplified NLTK setup providing a single get_wordnet() function.

This module ensures NLTK WordNet data is available in the virtual environment
and returns the ready-to-use wordnet corpus. If anything goes wrong, it fails fast.

Public API:
- get_wordnet() -> wordnet corpus object
"""

from __future__ import annotations

import sys
import zipfile
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Any

# Check NLTK availability at module import time
try:
    import nltk
except ImportError:
    raise ImportError("NLTK is not installed. Please install it: pip install nltk")

logger = logging.getLogger(__name__)


class WordNetManager:
    """
    Singleton manager for NLTK WordNet setup and access.

    This class ensures NLTK data is properly initialized in the virtual environment
    and provides thread-safe access to the WordNet corpus.
    """

    _instance: Optional["WordNetManager"] = None
    _wordnet: Optional[Any] = None
    _initialized: bool = False

    # Required NLTK data packages
    REQUIRED_DATA: List[Tuple[str, str]] = [
        ("punkt", "tokenizers/punkt"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ]

    def __new__(cls) -> "WordNetManager":
        """Ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the manager (called only once due to singleton)."""
        # Prevent re-initialization
        if WordNetManager._initialized:
            return

        # Set up venv-local NLTK data path
        self._setup_nltk_paths()

        # Initialize NLTK data
        self._initialize_nltk_data()

        # Import wordnet after ensuring data is available
        from nltk.corpus import wordnet

        WordNetManager._wordnet = wordnet
        WordNetManager._initialized = True

        logger.info("NLTK WordNet initialized successfully")

    def _setup_nltk_paths(self) -> None:
        """Configure NLTK to use venv-local data directory."""
        venv_root = Path(sys.prefix)
        self.nltk_data_dir = venv_root / "nltk_data"

        # Ensure NLTK looks in venv first
        if str(self.nltk_data_dir) not in nltk.data.path:
            nltk.data.path.insert(0, str(self.nltk_data_dir))

    def _initialize_nltk_data(self) -> None:
        """Check and download/extract required NLTK data."""
        for alias, category_path in self.REQUIRED_DATA:
            try:
                nltk.data.find(category_path)
                logger.debug(f"NLTK data '{alias}' already available")
            except LookupError:
                self._ensure_data_available(alias, category_path)

    def _ensure_data_available(self, alias: str, category_path: str) -> None:
        """Download or extract NLTK data package."""
        # Check for existing zip file
        category = category_path.split("/")[0]
        zip_path = self.nltk_data_dir / category / f"{alias}.zip"

        if zip_path.exists():
            logger.info(f"Extracting {alias} from existing zip")
            self._extract_nltk_zip(zip_path, alias, category_path)
        else:
            logger.info(f"Downloading NLTK data: {alias}")
            nltk.download(alias, download_dir=str(self.nltk_data_dir), quiet=True)

    def _extract_nltk_zip(self, zip_path: Path, alias: str, category_path: str) -> None:
        """Extract NLTK zip file to correct location."""
        category, subdir = category_path.split("/", 1)
        extract_to = zip_path.parent / subdir

        extract_to.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to.parent)

        logger.info(f"Extracted {alias} to {extract_to}")

    @classmethod
    def get_wordnet(cls) -> Any:
        """
        Get the initialized NLTK WordNet corpus.

        This method ensures NLTK data is set up on first call and returns
        the wordnet corpus object. Subsequent calls return the cached object.

        Returns
        -------
        wordnet
            The NLTK wordnet corpus object, ready to use.

        Raises
        ------
        RuntimeError
            If WordNet initialization failed.
        """
        # Ensure singleton is created (triggers __init__ if needed)
        _ = cls()

        if cls._wordnet is None:
            raise RuntimeError("WordNet initialization failed")
        return cls._wordnet


# Public API - direct access to the class method
get_wordnet = WordNetManager.get_wordnet
