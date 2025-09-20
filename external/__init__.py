"""
NLP Setup Module

This module provides singleton managers for various NLP libraries used in the project.
Each setup module ensures the required data is downloaded and provides a simple API
for accessing the initialized NLP tools.

Available setup functions:
- get_wordnet(): NLTK WordNet corpus
- get_spacy(): spaCy English model
- get_symspell(): SymSpellPy spell checker
- get_rapidfuzz(): RapidFuzz process utilities
"""

from .nltk_setup import get_wordnet

__all__ = [
    "get_wordnet",
]
