# external/concept_extractor.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Any

from external.spacy_setup import get_spacy

class ConceptExtractor:
    """
    Parses a natural language query to extract core concept sets.

    This class identifies the primary "head" concept (the main noun) and
    associated "modifier" concepts (adjectives, related nouns, verbs) from a query.
    """

    def __init__(self, artifacts_dir: Path | None = None):
        """
        Initializes the ConceptExtractor.

        In a future implementation, this would load pre-computed corpus statistics
        (like collocations or word vectors) from the artifacts_dir to improve
        concept expansion.

        Args:
            artifacts_dir (Path | None): Path to the directory containing corpus stats.
        """
        # This is where you would load pre-computed phrase lists, sense maps, etc.
        # For now, we will rely on lemmatization for expansion.
        self.artifacts_dir = artifacts_dir
        self.nlp = get_spacy()

    def _expand_term(self, term: str) -> Set[str]:
        """
        Expands a single term into a set of related terms.

        Currently uses lemmatization. This can be extended to use corpus stats
        for finding synonyms and related phrases.

        Args:
            term (str): The term to expand.

        Returns:
            Set[str]: A set of related terms, including the original term and its lemma.
        """
        # Process the term to get the lemma
        doc = self.nlp(term.lower())
        # The set includes the original term and the lemma of the first token
        return {token.text.lower() for token in doc} | {
            doc[0].lemma_.lower()
        }

    def build_concept_sets(self, query: str) -> Dict[str, Set[str]]:
        """
        Analyzes a query to build sets for head and modifier concepts.

        It uses dependency parsing to find the main subject of the query (head)
        and the words that describe or modify it.

        Args:
            query (str): The user's search query.

        Returns:
            Dict[str, Set[str]]: A dictionary with "head" and "modifiers" concept sets.
        """
        doc = self.nlp(query.lower())

        head: Optional[Any] = None
        modifiers: List[Any] = []

        # Find the root of the sentence, which is often the main action or concept.
        root = next((tok for tok in doc if tok.dep_ == "root"), None)

        if root:
            # The head is often the direct object (dobj) or nominal subject (nsubj)
            # of the root verb.
            noun_deps = [
                child for child in root.children if child.dep_ in ("dobj", "nsubj")
            ]
            if noun_deps:
                head = noun_deps[0]
            # If the root itself is a noun, it's the head.
            elif root.pos_ in ("NOUN", "PROPN"):
                head = root

        # If the dependency parse doesn't yield a clear head, fall back to the last noun.
        if not head:
            nouns = [tok for tok in doc if tok.pos_ in ("NOUN", "PROPN")]
            if nouns:
                head = nouns[-1]

        if not head:
            # If no noun is found, we can't determine concepts.
            return {"head": set(), "modifiers": set()}

        # Gather modifiers: compound nouns, adjectives, and other related nouns.
        for child in head.children:
            if child.dep_ in ("compound", "amod", "nmod"):
                modifiers.append(child)

        # The root verb is often an important modifier (e.g., "change", "replace").
        if root and root.pos_ == "VERB":
            modifiers.append(root)

        # Expand the identified head and modifier tokens into concept sets.
        head_set = self._expand_term(head.text)

        modifier_set = set()
        for mod in modifiers:
            modifier_set.update(self._expand_term(mod.text))

        return {"head": head_set, "modifiers": modifier_set}
