#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spacy_setup.py
==============

Simplified spaCy setup providing a single get_spacy() function.

This module ensures the spaCy transformer model is available and returns
a ready-to-use spaCy nlp object. If anything goes wrong, it fails fast.

Public API:
- get_spacy() -> spacy nlp object with en_core_web_trf model loaded
"""

from __future__ import annotations

import sys
import subprocess
import logging
import spacy
from spacy.language import Language
from spacy.cli.download import download


logger = logging.getLogger(__name__)

# Required model
_MODEL_NAME = "en_core_web_trf"


class SpacySingleton:
    """Singleton class to manage the spaCy NLP object."""

    _instance = None
    _nlp: Language | None = None

    def __new__(cls):
        """Create a new instance of the SpacySingleton class."""
        if cls._instance is None:
            cls._instance = super(SpacySingleton, cls).__new__(cls)
            cls._instance._initialize_spacy()
        return cls._instance

    def _initialize_spacy(self):
        """
        Initialize the spaCy model.

        This method is called once on the first instantiation of the class.
        It loads the spaCy model, downloading it if necessary.
        """
        # Try to load the model
        try:
            self._nlp = spacy.load(_MODEL_NAME)
            logger.info(f"Loaded spaCy model: {_MODEL_NAME}")
        except OSError:
            # Model not installed, download it
            logger.info(f"Downloading spaCy model: {_MODEL_NAME}")
            try:
                # Use spacy's download function
                download(_MODEL_NAME, direct=True)

                # Try loading again
                self._nlp = spacy.load(_MODEL_NAME)
                logger.info(f"Successfully downloaded and loaded: {_MODEL_NAME}")
            except (OSError, subprocess.CalledProcessError) as e:
                # Fallback to subprocess
                logger.warning(f"Direct download failed: {e}. Trying subprocess...")
                cmd = [sys.executable, "-m", "spacy", "download", _MODEL_NAME]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=False
                )

                if result.returncode != 0:
                    raise RuntimeError(
                        f"Failed to download spaCy model {_MODEL_NAME}:\n"
                        f"stdout: {result.stdout}\n"
                        f"stderr: {result.stderr}"
                    ) from e

                # Final attempt to load
                self._nlp = spacy.load(_MODEL_NAME)

        logger.info("spaCy initialized successfully")

    def get_nlp(self) -> Language:
        """
        Get the initialized spaCy NLP object.

        Returns
        -------
        Language
            The initialized spaCy NLP object.

        Raises
        ------
        RuntimeError
            If the spaCy model is not initialized.
        """
        if self._nlp is None:
            raise RuntimeError("spaCy model not initialized")
        return self._nlp


def get_spacy(gpu_id: int | None = None) -> Language:
    """
    Get the initialized spaCy NLP object with transformer model.

    This function ensures the en_core_web_trf model is downloaded and loaded
    on first call. Subsequent calls return the cached nlp object.

    Returns
    -------
    nlp
        The spaCy nlp object with en_core_web_trf model loaded.

    Raises
    ------
    ImportError
        If spaCy is not installed.
    RuntimeError
        If model download or loading fails.
    """
    if gpu_id is not None:
        # Attempt to use GPU if specified
        try:
            spacy.require_gpu(gpu_id)
            logger.info(f"spaCy configured to use GPU {gpu_id}")
        except Exception as e:
            logger.warning(f"Failed to set spaCy to use GPU {gpu_id}: {e}")

    return SpacySingleton().get_nlp()
