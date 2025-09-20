"""
concept_gate.py
================

Dynamic concept-set gating and sense-aware re-scoring for manual/document search.

This module provides a *domain-agnostic* heuristic layer that you can place
between your candidate generation (e.g., BM25/ANN) and reranking. It builds
query-specific "concept sets" from the user's query and your own corpus—
*without* hardcoded token lists—then filters/rescores chunk candidates based
on proximity coverage and sense disambiguation.

Only standard libraries plus NumPy, Pandas, scikit-learn, and PyTorch are used.
No spaCy/NLTK is required.

Usage (high level)
------------------
1) Fit once on your corpus (list of strings; ideally each is a chunk):
    gate = ConceptGate()
    gate.fit(corpus_texts)

2) At query time, after you have candidate chunk indices and base scores:
    mask, new_scores, debug = gate.apply_gate(
        query="How do I change the cabin air filter?",
        tokens_per_chunk=gate.tokens_per_chunk,
        base_scores=base_scores,         # np.ndarray aligned to candidate indices
        cand_indices=cand_indices,       # np.ndarray of candidate indices (ints)
        window=8,
        backoff_top_k=5
    )

3) Use `mask` to keep only candidates that pass. Then use `new_scores` for
   reweighting before or after your cross-encoder step (your choice).

Design notes
------------
- Concept sets are derived from:
  (a) Query content terms (simple lemmatization + stopword removal)
  (b) Corpus n-gram PMI phrases (2–4 grams)
  (c) Distributional neighbors from a small skip-gram trained on the corpus
  (d) Morphological variants via simple suffix rules
- Coverage requires at least one *head* term and at least one *modifier* term
  to co-occur within a configurable token window; graceful backoff is provided.
- Sense disambiguation: frequent ambiguous heads (e.g., "filter") get a
  data-driven context split (k-means over collocate vectors), and chunks are
  boosted when their context matches the query-selected sense.

This module is self-contained and PEP 8 compliant.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize as sp_normalize
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Tokenization, stopwords, light lemmatization/inflection
# ---------------------------------------------------------------------------

_DEFAULT_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "how", "i", "in", "into", "is", "it", "of", "on", "or", "so", "such",
    "that", "the", "their", "then", "there", "these", "this", "those", "to",
    "we", "what", "when", "where", "which", "who", "why", "will", "with",
    "do", "does", "did", "can", "could", "should", "would", "you", "your",
    "ours", "ourselves", "my", "mine", "me", "myself", "yours", "they",
    "them", "themselves", "he", "she", "his", "her", "hers", "its", "our",
    "was", "were", "been", "being", "if", "than", "up", "down", "over",
    "under", "again", "further", "then", "once", "because",
}

_WORD_RE = re.compile(r"[A-Za-z0-9\-]+")


def tokenize(text: str) -> List[str]:
    """Lowercase tokenization keeping letters, digits and dashes."""
    return _WORD_RE.findall((text or "").lower())


def simple_lemma(token: str) -> str:
    """
    Very light lemmatizer to reduce common English inflections.
    Not linguistically perfect but robust and fast for manuals.
    """
    t = token.lower()
    if len(t) <= 3:
        return t
    # Common endings
    for suf in ("ing", "ed", "ies", "s"):
        if t.endswith(suf):
            if suf == "ies" and len(t) > 4:
                return t[:-3] + "y"
            if suf == "s" and len(t) <= 4:
                # cars -> car, but keep 'bus' -> 'bus'
                return t[:-1] if t[-2] != "u" else t
            if suf in ("ing", "ed"):
                # handling doubled consonants crudely (e.g., 'fitted' -> 'fit')
                base = t[:-len(suf)]
                if len(base) > 2 and base[-1] == base[-2]:
                    base = base[:-1]
                if suf == "ing" and base.endswith("e"):
                    base = base[:-1]
                return base
            return t[:-len(suf)]
    return t


def normalize_tokens(tokens: Iterable[str]) -> List[str]:
    """Tokenize + lowercase + light lemmatize + stopword filter."""
    out = []
    for tok in tokens:
        lem = simple_lemma(tok)
        if lem and lem not in _DEFAULT_STOP:
            out.append(lem)
    return out


def generate_inflections(token: str) -> Set[str]:
    """Generate simple inflectional variants for surface matching."""
    t = token
    forms = {t}
    if len(t) >= 3:
        if t.endswith("y"):
            forms.add(t[:-1] + "ies")
        if t.endswith("e"):
            forms.update({t[:-1] + "ing", t + "d", t + "s"})
        else:
            forms.update({t + "ing", t + "ed", t + "s"})
    return forms


# ---------------------------------------------------------------------------
# Phrase inventory via PMI (2–4-grams)
# ---------------------------------------------------------------------------

@dataclass
class Phrase:
    phrase: Tuple[str, ...]
    pmi: float
    freq: int


class PhraseInventory:
    """
    Build a phrase inventory with PMI for n-grams (2–4).

    Parameters
    ----------
    min_freq : int
        Minimum frequency for an n-gram to be considered.
    max_ngrams : int
        Limit stored phrases by highest PMI to control memory.
    """

    def __init__(self, min_freq: int = 3, max_ngrams: int = 100_000) -> None:
        self.min_freq = int(min_freq)
        self.max_ngrams = int(max_ngrams)
        self.phrases: List[Phrase] = []
        self._tok_counts: Counter[str] = Counter()
        self._ngram_counts: Counter[Tuple[str, ...]] = Counter()
        self._vocab_size: int = 0
        self._total_tokens: int = 0

    @staticmethod
    def _ngrams(tokens: Sequence[str], n: int) -> Iterable[Tuple[str, ...]]:
        for i in range(len(tokens) - n + 1):
            yield tuple(tokens[i : i + n])

    def fit(self, tokenized_docs: Iterable[List[str]]) -> "PhraseInventory":
        """
        Compute unigram and n-gram counts and PMI scores.
        """
        for toks in tokenized_docs:
            self._tok_counts.update(toks)
            self._total_tokens += len(toks)
            for n in (2, 3, 4):
                self._ngram_counts.update(self._ngrams(toks, n))

        self._vocab_size = len(self._tok_counts)

        # Compute PMI and keep top by PMI (with min_freq)
        phrases: List[Phrase] = []
        for ng, c_xy in self._ngram_counts.items():
            if c_xy < self.min_freq:
                continue
            # Use pairwise product for n-grams; approximate PMI by average pairwise PMI
            denom = 1.0
            valid = True
            for w in ng:
                c_x = self._tok_counts.get(w, 0)
                if c_x == 0:
                    valid = False
                    break
                denom *= c_x
            if not valid:
                continue
            # PMI ~ log( (c_xy * N^(n-1)) / product(c_x) )
            n = len(ng)
            pmi = math.log((c_xy * (self._total_tokens ** (n - 1))) / max(1.0, denom) + 1e-9)
            phrases.append(Phrase(phrase=ng, pmi=pmi, freq=c_xy))

        phrases.sort(key=lambda p: (-p.pmi, -p.freq))
        self.phrases = phrases[: self.max_ngrams]
        return self

    def expand_for_term(self, term: str, top_n: int = 20) -> List[str]:
        """
        Return top phrases (as strings) that contain the given term, by PMI.
        """
        term = simple_lemma(term)
        hits = [p for p in self.phrases if term in p.phrase]
        hits.sort(key=lambda p: (-p.pmi, -p.freq))
        out = [" ".join(p.phrase) for p in hits[:top_n]]
        return out


# ---------------------------------------------------------------------------
# Distributional neighbors via tiny skip-gram (PyTorch)
# ---------------------------------------------------------------------------

class SkipGramNeg(nn.Module):
    """Minimal skip-gram with negative sampling."""

    def __init__(self, vocab_size: int, dim: int = 64) -> None:
        super().__init__()
        self.input = nn.Embedding(vocab_size, dim)
        self.output = nn.Embedding(vocab_size, dim)

    def forward(self, center: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        v = self.input(center)       # (B, D)
        u_pos = self.output(pos)     # (B, D)
        u_neg = self.output(neg)     # (B, D)

        pos_score = (v * u_pos).sum(dim=1)
        neg_score = (v * u_neg).sum(dim=1)

        loss = -torch.log(torch.sigmoid(pos_score) + 1e-9).mean() - \
               torch.log(1 - torch.sigmoid(neg_score) + 1e-9).mean()
        return loss


class DistributionalNeighbors:
    """
    Train a tiny skip-gram model to get distributional neighbors.

    Notes
    -----
    - Designed for *fast* training on a few epochs; this is not word2vec.
    - Uses a simple unigram negative sampler.
    """

    def __init__(
        self,
        window: int = 4,
        dim: int = 64,
        epochs: int = 2,
        lr: float = 0.025,
        negatives: int = 1,
        seed: int = 13,
    ) -> None:
        self.window = int(window)
        self.dim = int(dim)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.negatives = int(negatives)
        self.seed = int(seed)

        self.vocab: Dict[str, int] = {}
        self.ivocab: List[str] = []
        self.unigram: np.ndarray = np.array([])
        self.model: Optional[SkipGramNeg] = None

    def _build_vocab(self, tokenized_docs: Iterable[List[str]]) -> List[List[int]]:
        counts = Counter()
        seqs: List[List[int]] = []
        # First pass to collect
        for toks in tokenized_docs:
            counts.update(toks)
        self.ivocab = [w for w, _ in counts.most_common()]
        self.vocab = {w: i for i, w in enumerate(self.ivocab)}
        total = sum(counts.values())
        probs = np.array([counts[w] for w in self.ivocab], dtype=np.float64)
        self.unigram = probs / max(1.0, probs.sum())

        # Second pass to map
        for toks in tokenized_docs:
            seqs.append([self.vocab[t] for t in toks if t in self.vocab])
        return seqs

    def fit(self, tokenized_docs: Iterable[List[str]]) -> "DistributionalNeighbors":
        rng = np.random.RandomState(self.seed)
        torch.manual_seed(self.seed)
        seqs = self._build_vocab(tokenized_docs)
        vocab_size = len(self.ivocab)
        if vocab_size == 0:
            self.model = None
            return self

        model = SkipGramNeg(vocab_size=vocab_size, dim=self.dim)
        self.model = model
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        def sample_negative(batch_size: int) -> torch.Tensor:
            idx = rng.choice(vocab_size, size=batch_size, p=self.unigram)
            return torch.tensor(idx, dtype=torch.long)

        # Build simple (center, positive) pairs
        pairs: List[Tuple[int, int]] = []
        for seq in seqs:
            L = len(seq)
            for i, c in enumerate(seq):
                left = max(0, i - self.window)
                right = min(L, i + self.window + 1)
                for j in range(left, right):
                    if j == i:
                        continue
                    pairs.append((c, seq[j]))

        if not pairs:
            return self

        batch_size = 1024
        rng.shuffle(pairs)
        for _epoch in range(self.epochs):
            loss_ema = None
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                c = torch.tensor([p[0] for p in batch], dtype=torch.long)
                pos = torch.tensor([p[1] for p in batch], dtype=torch.long)
                neg = sample_negative(len(batch))
                opt.zero_grad()
                loss = model(c, pos, neg)
                loss.backward()
                opt.step()
                # cheap trace
                loss_v = float(loss.detach().cpu().item())
                loss_ema = loss_v if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_v

        return self

    def neighbors(self, term: str, top_k: int = 10) -> List[str]:
        """Return top-k nearest neighbors to the term in embedding space."""
        if not self.model or term not in self.vocab:
            return []
        idx = self.vocab[term]
        vec = self.model.input.weight.detach().cpu().numpy()[idx : idx + 1]
        all_vecs = self.model.input.weight.detach().cpu().numpy()
        sims = (all_vecs @ vec.T).ravel()
        order = np.argsort(-sims)
        result = []
        for j in order[1 : top_k + 1]:
            result.append(self.ivocab[j])
        return result


# ---------------------------------------------------------------------------
# Concept building, coverage gating, and sense disambiguation
# ---------------------------------------------------------------------------

@dataclass
class ConceptSets:
    head: Set[str]
    modifiers: List[Set[str]]  # zero or more sets
    debug: Dict[str, Any]


def _content_terms(tokens: List[str]) -> List[str]:
    """Keep content-like terms (roughly nouns/verbs/adjectives by shape)."""
    out = []
    for t in tokens:
        if t in _DEFAULT_STOP:
            continue
        # accept tokens with letters or digits; drop tiny dashes
        if len(t) <= 1:
            continue
        out.append(simple_lemma(t))
    return out


def _pick_head_and_modifiers(content_terms: List[str]) -> Tuple[str, List[str]]:
    """
    Heuristic: choose the *rightmost* noun-ish term as head.
    Since we lack POS, approximate by preferring tokens that are not common verbs.
    Fall back to the last content term.
    """
    # A tiny list of frequent action verbs we do not want as heads.
    frequent_verbs = {"change", "replace", "adjust", "remove", "install", "open", "close", "check", "set"}

    head = None
    for t in reversed(content_terms):
        if t not in frequent_verbs:
            head = t
            break
    if head is None and content_terms:
        head = content_terms[-1]

    modifiers = [t for t in content_terms if t != head]
    return head or "", modifiers


def _collapse_by_lemma(terms: Iterable[str]) -> Set[str]:
    """Collapse duplicates by lemma (using simple_lemma)."""
    out = set()
    seen = set()
    for t in terms:
        lem = simple_lemma(t)
        if lem in seen:
            continue
        seen.add(lem)
        out.add(lem)
    return out


class ConceptGate:
    """
    End-to-end concept builder and gate.

    Call `fit(corpus_texts)` once to prepare phrase inventory, distributional
    neighbors, and collocate statistics; then `apply_gate(...)` per query.

    Attributes
    ----------
    tokens_per_chunk : List[List[str]]
        Normalized tokens for each chunk; populated after `fit`.
    """

    def __init__(
        self,
        min_phrase_freq: int = 3,
        max_phrases: int = 120_000,
        skipgram_dim: int = 64,
        skipgram_epochs: int = 2,
        window: int = 4,
        seed: int = 13,
    ) -> None:
        self.phrases = PhraseInventory(min_freq=min_phrase_freq, max_ngrams=max_phrases)
        self.neighbors = DistributionalNeighbors(
            window=window, dim=skipgram_dim, epochs=skipgram_epochs, seed=seed
        )
        self.tokens_per_chunk: List[List[str]] = []
        self._collocates: Dict[str, Counter[str]] = defaultdict(Counter)
        self._seed = int(seed)

    # ---------------------- fitting ----------------------

    def fit(self, corpus_texts: Sequence[str]) -> "ConceptGate":
        """
        Fit the gate on the corpus: tokenize, build phrase PMI, train neighbors,
        and record collocate counts for sense disambiguation.
        """
        rng = random.Random(self._seed)

        # Tokenize and normalize
        self.tokens_per_chunk = [normalize_tokens(tokenize(t)) for t in corpus_texts]

        # Phrase inventory
        self.phrases.fit(self.tokens_per_chunk)

        # Train tiny skip-gram
        self.neighbors.fit(self.tokens_per_chunk)

        # Collocate counts for sense disambiguation (±20 window)
        window = 20
        for toks in self.tokens_per_chunk:
            for i, w in enumerate(toks):
                left = max(0, i - window)
                right = min(len(toks), i + window + 1)
                ctx = toks[left:i] + toks[i + 1:right]
                self._collocates[w].update(ctx)

        # Optionally prune collocates to top-N for memory
        for w, cnt in list(self._collocates.items()):
            most = cnt.most_common(100)
            self._collocates[w] = Counter(dict(most))

        # Small shuffle to avoid any pathological order
        rng.shuffle(self.tokens_per_chunk)
        return self

    # ---------------------- concept building ----------------------

    def _expand_term(self, term: str, top_phrases: int = 20, top_neighbors: int = 10) -> Set[str]:
        """Expand a term with PMI phrases, neighbors, and inflections."""
        expanded: Set[str] = set()
        expanded.add(term)

        # PMI phrases that contain the term
        for ph in self.phrases.expand_for_term(term, top_n=top_phrases):
            expanded.add(ph)

        # Distributional neighbors
        for nb in self.neighbors.neighbors(term, top_k=top_neighbors):
            expanded.add(nb)

        # Inflections for each atomic token in expansions
        atomic = set()
        for item in list(expanded):
            for tok in item.split():
                atomic.add(tok)
        for a in list(atomic):
            expanded.update(generate_inflections(a))

        return _collapse_by_lemma(expanded)

    def build_concept_sets(self, query: str) -> ConceptSets:
        """
        Build dynamic concept sets (head + modifiers) from the query and corpus.
        """
        q_toks = normalize_tokens(tokenize(query))
        content = _content_terms(q_toks)
        head, mods = _pick_head_and_modifiers(content)

        head_set = self._expand_term(head) if head else set()
        mod_sets: List[Set[str]] = []
        for m in mods:
            mod_sets.append(self._expand_term(m))

        # Limit size to avoid blow-up
        def _topn(s: Set[str], n: int = 50) -> Set[str]:
            # approximate: keep shorter strings first, then lexicographic
            arr = sorted(s, key=lambda x: (len(x), x))[:n]
            return set(arr)

        head_set = _topn(head_set, 60)
        mod_sets = [_topn(ms, 40) for ms in mod_sets]

        debug = {
            "query_tokens": q_toks,
            "content_terms": content,
            "head": head,
            "modifiers": mods,
            "head_candidates": sorted(head_set),
            "modifier_candidates": [sorted(ms) for ms in mod_sets],
        }
        return ConceptSets(head=head_set, modifiers=mod_sets, debug=debug)

    # ---------------------- coverage + proximity ----------------------

    @staticmethod
    def _positions(tokens: List[str], terms: Set[str]) -> List[int]:
        pos = []
        # match both atomic terms and multi-word phrases
        term_atoms = {t for t in terms if " " not in t}
        phrases = [t.split() for t in terms if " " in t]
        # atomic
        for i, w in enumerate(tokens):
            if w in term_atoms:
                pos.append(i)
        # phrases
        for i in range(len(tokens)):
            for ph in phrases:
                n = len(ph)
                if i + n <= len(tokens) and tokens[i : i + n] == ph:
                    pos.append(i)
        return pos

    @staticmethod
    def _has_proximity(
        head_pos: List[int],
        mod_pos: List[int],
        window: int,
    ) -> bool:
        """Return True if any head position is within `window` of any modifier pos."""
        if not head_pos or not mod_pos:
            return False
        i, j = 0, 0
        head_pos = sorted(head_pos)
        mod_pos = sorted(mod_pos)
        while i < len(head_pos) and j < len(mod_pos):
            if abs(head_pos[i] - mod_pos[j]) <= window:
                return True
            if head_pos[i] < mod_pos[j]:
                i += 1
            else:
                j += 1
        return False

    def coverage_mask(
        self,
        concept: ConceptSets,
        tokens_per_chunk: Sequence[List[str]],
        window: int = 8,
        backoff_top_k: int = 5,
        candidate_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Build a boolean mask of chunks that satisfy the concept-coverage rule.

        Coverage rule:
            - Head coverage: at least one token/phrase from `concept.head` appears
            - Modifier coverage: at least one token/phrase from *any* modifier set
              appears near a head occurrence (within `window` tokens)

        Graceful backoff if too few chunks survive:
            - First widen window to 16
            - Then, allow head-only coverage

        Parameters
        ----------
        concept : ConceptSets
        tokens_per_chunk : Sequence[List[str]]
            Normalized tokens for each chunk (prepared at fit time).
        window : int
            Proximity window size (default 8).
        backoff_top_k : int
            Target count; if survivors < backoff_top_k, we relax the criteria.
        candidate_indices : Optional[Sequence[int]]
            If provided, only evaluate these chunk indices.

        Returns
        -------
        mask : np.ndarray
            Boolean mask aligned to `tokens_per_chunk` (or to `candidate_indices` if provided).
        debug : Dict[str, Any]
            Diagnostics about stages and survivor counts.
        """
        if candidate_indices is None:
            candidate_indices = list(range(len(tokens_per_chunk)))
        cand = list(candidate_indices)

        head_terms = concept.head
        mod_sets = concept.modifiers

        # Stage 1: strict coverage
        def _apply(curr_window: int, require_modifier: bool) -> np.ndarray:
            keep = np.zeros((len(cand),), dtype=bool)
            for idx_local, idx in enumerate(cand):
                toks = tokens_per_chunk[idx]
                head_pos = self._positions(toks, head_terms)
                if not head_pos:
                    continue
                if not require_modifier or not mod_sets:
                    keep[idx_local] = True
                    continue
                # any modifier set suffices
                for ms in mod_sets:
                    mod_pos = self._positions(toks, ms)
                    if self._has_proximity(head_pos, mod_pos, curr_window):
                        keep[idx_local] = True
                        break
            return keep

        keep = _apply(window, require_modifier=True)
        stage = "strict"
        if keep.sum() < backoff_top_k:
            keep = _apply(16, require_modifier=True)
            stage = "widened"
        if keep.sum() < backoff_top_k:
            keep = _apply(16, require_modifier=False)
            stage = "head_only"

        debug = {
            "stage": stage,
            "survivors": int(keep.sum()),
            "evaluated": len(cand),
            "window": window,
        }
        return keep, debug

    # ---------------------- sense disambiguation ----------------------

    def _context_vector(self, tokens: List[str], focus: str, window: int = 20) -> np.ndarray:
        """
        Build a collocate vector for tokens where `focus` appears, summing windowed contexts.
        """
        vocab = list(self._collocates.keys())
        idx = {w: i for i, w in enumerate(vocab)}
        vec = np.zeros((len(vocab),), dtype=np.float32)
        for i, w in enumerate(tokens):
            if w != focus:
                continue
            left = max(0, i - window)
            right = min(len(tokens), i + window + 1)
            ctx = tokens[left:i] + tokens[i + 1:right]
            for c in ctx:
                j = idx.get(c)
                if j is not None:
                    vec[j] += 1.0
        if vec.sum() > 0:
            vec = vec / max(1.0, np.linalg.norm(vec))
        return vec

    def _cluster_head_senses(self, head: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Cluster contexts for the head term into 2 centroids (if enough data).
        Returns (centroid_0, centroid_1) or None.
        """
        if head not in self._collocates:
            return None
        # Build a tiny matrix of top collocates as pseudo-examples
        vocab = list(self._collocates.keys())
        idx = {w: i for i, w in enumerate(vocab)}
        # Create synthetic examples: for each occurrence word c with count,
        # form a sparse basis vector along dimension idx[c].
        counts = self._collocates[head]
        if len(counts) < 10:
            return None
        words, vals = zip(*counts.most_common(60))
        mat = np.zeros((len(words), len(vocab)), dtype=np.float32)
        for r, (w, v) in enumerate(zip(words, vals)):
            j = idx.get(w, -1)
            if j >= 0:
                mat[r, j] = float(v)
        if mat.shape[0] < 4:
            return None
        km = KMeans(n_clusters=2, n_init=4, random_state=self._seed)
        labels = km.fit_predict(mat)
        cents = km.cluster_centers_
        # L2 normalize
        cents = sp_normalize(cents, axis=1)
        return cents[0], cents[1]

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        num = float((a * b).sum())
        den = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return num / den

    def sense_rescore(
        self,
        query: str,
        concept: ConceptSets,
        tokens_per_chunk: Sequence[List[str]],
        base_scores: np.ndarray,
        candidate_indices: Sequence[int],
        boost: float = 0.15,
    ) -> np.ndarray:
        """
        Rescore candidates by matching the head sense closest to the query context.

        - Build two head sense centroids (if available) from collocates.
        - Choose the sense whose centroid is closer to the query context terms.
        - For each candidate, compute cosine between its head-context vector and
          the chosen centroid; add a small boost proportional to similarity.
        """
        if not concept.head:
            return base_scores

        # Pick any head token as representative (first in sorted order).
        head = sorted(concept.head)[0].split(" ")[0]

        cents = self._cluster_head_senses(head)
        if cents is None:
            return base_scores

        c0, c1 = cents

        q_toks = normalize_tokens(tokenize(query))
        q_vec = np.zeros_like(c0)
        # Approximate query context vector using collocate basis dims
        vocab = list(self._collocates.keys())
        idx = {w: i for i, w in enumerate(vocab)}
        for t in q_toks:
            j = idx.get(t)
            if j is not None:
                q_vec[j] += 1.0
        if q_vec.sum() > 0:
            q_vec = q_vec / max(1.0, np.linalg.norm(q_vec))

        # Choose closest sense
        score0 = self._cos(q_vec, c0)
        score1 = self._cos(q_vec, c1)
        chosen = c0 if score0 >= score1 else c1

        # Chunk-wise similarity -> boost
        new_scores = base_scores.copy()
        for loc, idx_chunk in enumerate(candidate_indices):
            toks = tokens_per_chunk[idx_chunk]
            ctx_vec = self._context_vector(toks, head=head, window=20)
            sim = self._cos(ctx_vec, chosen)
            new_scores[loc] += float(boost * max(0.0, sim))
        return new_scores

    # ---------------------- public API ----------------------

    def apply_gate(
        self,
        query: str,
        tokens_per_chunk: Sequence[List[str]],
        base_scores: np.ndarray,
        cand_indices: Sequence[int],
        window: int = 8,
        backoff_top_k: int = 5,
        apply_sense: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Build concept sets for the query, apply coverage filter with backoff,
        and optionally rescore by head sense proximity.

        Parameters
        ----------
        query : str
            User query string.
        tokens_per_chunk : Sequence[List[str]]
            Normalized tokens for each chunk, aligned to global chunk indices.
        base_scores : np.ndarray
            Base scores aligned to `cand_indices` (same length).
        cand_indices : Sequence[int]
            Global chunk indices corresponding to base_scores.
        window : int
            Proximity window for coverage (default 8).
        backoff_top_k : int
            Minimum survivors before relaxing.
        apply_sense : bool
            Whether to apply sense-aware reweighting.

        Returns
        -------
        mask : np.ndarray
            Boolean mask aligned to `cand_indices`; True = keep.
        new_scores : np.ndarray
            Reweighted scores aligned to `cand_indices` (unchanged where masked False).
        debug : Dict[str, Any]
            Rich diagnostics for logging.
        """
        concept = self.build_concept_sets(query)
        # Coverage filter
        mask, cov_dbg = self.coverage_mask(
            concept=concept,
            tokens_per_chunk=tokens_per_chunk,
            window=window,
            backoff_top_k=backoff_top_k,
            candidate_indices=cand_indices,
        )

        # Re-score survivors by sense proximity (optional)
        rescored = base_scores.copy()
        if apply_sense and mask.any():
            survivors = [i for k, i in enumerate(cand_indices) if mask[k]]
            loc_map = {i: pos for pos, i in enumerate(cand_indices)}
            # slice arrays for survivors in order
            sub_scores = np.array([base_scores[loc_map[i]] for i in survivors], dtype=np.float32)
            sub_scores = self.sense_rescore(
                query=query,
                concept=concept,
                tokens_per_chunk=tokens_per_chunk,
                base_scores=sub_scores,
                candidate_indices=survivors,
                boost=0.15,
            )
            # write back
            for i, s in zip(survivors, sub_scores):
                rescored[loc_map[i]] = s

        debug = {"concept": concept.debug, "coverage": cov_dbg}
        return mask, rescored, debug


# ---------------------------------------------------------------------------
# TOC/Index hygiene (generic, query-agnostic)
# ---------------------------------------------------------------------------

def toc_index_penalty(text: str) -> float:
    """
    Heuristic penalty for TOC/Index/spec pages. Use as a kill switch upstream.
    """
    if not text:
        return 0.0
    s = text.lower()
    pen = 0.0
    toc_words = ("table of contents", "contents", "index", "glossary", "specifications", "dimensions", "bulb wattage")
    if any(w in s for w in toc_words):
        pen += 0.25
    if s.count("..") >= 10:
        pen += 0.2
    num = sum(ch.isdigit() for ch in s)
    if num > max(30, len(s) // 10):
        pen += 0.1
    return float(min(pen, 0.5))


# ---------------------------------------------------------------------------
# Convenience: fitting helper that accepts raw texts
# ---------------------------------------------------------------------------

def fit_concept_gate(corpus_texts: Sequence[str]) -> ConceptGate:
    """
    One-liner to construct and fit a ConceptGate on your corpus.
    """
    gate = ConceptGate()
    gate.fit(corpus_texts)
    return gate
