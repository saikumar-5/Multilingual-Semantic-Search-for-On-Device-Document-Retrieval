"""
Text Preprocessor for multilingual document processing.
Handles tokenization, stopword removal, and normalization
for English, Hindi, and Telugu text.

CO1 Alignment: Document Representation - preparing text for indexing.
"""

import re
import string
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

# ── Stopword Lists ──────────────────────────────────────────────────────────

ENGLISH_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "it", "its", "this", "that", "these", "those", "i", "me", "my",
    "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "because", "as", "until", "while",
    "about", "between", "through", "during", "before", "after", "above",
    "below", "up", "down", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "if", "also", "into",
}

HINDI_STOPWORDS = {
    "का", "के", "की", "में", "है", "हैं", "को", "पर", "से", "ने", "और",
    "एक", "यह", "नहीं", "तो", "भी", "कि", "हो", "था", "थी", "थे",
    "या", "इस", "उस", "कर", "हुए", "जो", "अपने", "वह", "जा", "रहे",
    "रहा", "रही", "अब", "जब", "सब", "कुछ", "बहुत", "होता", "होती",
    "होते", "करता", "करती", "करते", "गया", "गई", "गए", "लिए", "दो",
    "तक", "ही", "बाद", "साथ", "उन", "इन", "कोई", "कैसे", "क्या",
    "वो", "मैं", "हम", "तुम", "आप", "वे", "उसे", "इसे", "किसी",
    "सकता", "सकती", "सकते", "होगा", "होगी", "होंगे", "अगर", "मगर",
    "लेकिन", "फिर", "ऐसे", "वैसे", "जैसे", "जिस", "जिन", "कहा",
}

TELUGU_STOPWORDS = {
    "మరియు", "ఒక", "ఈ", "ఆ", "ఇది", "అది", "కు", "లో", "తో",
    "యొక్క", "గా", "నుండి", "కోసం", "పై", "వల్ల", "ద్వారా",
    "అయిన", "అయితే", "కానీ", "లేదా", "మాత్రమే", "కూడా", "ఇంకా",
    "అన్ని", "ఏ", "ఎవరు", "ఏమి", "ఎక్కడ", "ఎప్పుడు", "ఎలా",
    "చేయడం", "చేసిన", "అవి", "వారు", "మేము", "నేను", "మీరు",
    "అతను", "ఆమె", "వాళ్ళు", "తన", "దాని", "వారి", "మా", "మీ",
    "నా", "ఉంది", "ఉన్న", "ఉన్నాయి", "ఉంటుంది", "చేస్తుంది",
    "అయింది", "కాదు", "లేదు", "అవును", "ఇప్పుడు", "ముందు",
}

ALL_STOPWORDS = ENGLISH_STOPWORDS | HINDI_STOPWORDS | TELUGU_STOPWORDS


class Preprocessor:
    """Multilingual text preprocessor for IR indexing."""

    def __init__(self, remove_stopwords: bool = True, min_length: int = 2):
        self.remove_stopwords = remove_stopwords
        self.min_length = min_length
        self.stopwords = ALL_STOPWORDS

        # Regex patterns
        self._url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self._email_pattern = re.compile(r"\S+@\S+\.\S+")
        self._number_pattern = re.compile(r"^\d+$")
        # Matches English words, Hindi (Devanagari), Telugu characters, and numbers
        self._token_pattern = re.compile(
            r"[a-zA-Z0-9\u0900-\u097F\u0C00-\u0C7F]+"
        )

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into individual words/tokens.
        Handles English, Hindi (Devanagari), and Telugu scripts.

        Returns:
            List of lowercased tokens.
        """
        if not text:
            return []

        # Remove URLs and emails
        text = self._url_pattern.sub(" ", text)
        text = self._email_pattern.sub(" ", text)

        # Extract tokens using multilingual regex
        tokens = self._token_pattern.findall(text)

        # Lowercase English tokens (Devanagari/Telugu don't have case)
        tokens = [t.lower() if t.isascii() else t for t in tokens]

        return tokens

    def preprocess(self, text: str) -> List[str]:
        """
        Full preprocessing pipeline: tokenize → filter → clean.

        Returns:
            List of cleaned, filtered tokens ready for indexing.
        """
        tokens = self.tokenize(text)

        filtered = []
        for token in tokens:
            # Skip short tokens
            if len(token) < self.min_length:
                continue
            # Skip very long tokens (likely garbage)
            if len(token) > 50:
                continue
            # Skip pure numbers (keep alphanumeric like 'covid19')
            if self._number_pattern.match(token):
                # Keep numbers that look like IDs (e.g., PAN numbers)
                if len(token) >= 5:
                    filtered.append(token)
                continue
            # Skip stopwords
            if self.remove_stopwords and token in self.stopwords:
                continue
            filtered.append(token)

        return filtered

    def preprocess_with_positions(self, text: str) -> List[Tuple[str, int]]:
        """
        Tokenize and return tokens with their position indices.
        Used for positional inverted index.

        Returns:
            List of (token, position) tuples.
        """
        tokens = self.tokenize(text)
        result = []
        pos = 0
        for token in tokens:
            if len(token) < self.min_length or len(token) > 50:
                pos += 1
                continue
            if self.remove_stopwords and token in self.stopwords:
                pos += 1
                continue
            result.append((token, pos))
            pos += 1
        return result

    def detect_language(self, text: str) -> str:
        """
        Detect the primary language of the text.
        Simple heuristic based on character ranges.

        Returns:
            Language code: 'en', 'hi', 'te', or 'mixed'
        """
        if not text:
            return "en"

        devanagari_count = len(re.findall(r"[\u0900-\u097F]", text))
        telugu_count = len(re.findall(r"[\u0C00-\u0C7F]", text))
        latin_count = len(re.findall(r"[a-zA-Z]", text))

        total = devanagari_count + telugu_count + latin_count
        if total == 0:
            return "en"

        # Determine dominant script
        if devanagari_count / total > 0.3:
            return "hi"
        elif telugu_count / total > 0.3:
            return "te"
        elif latin_count / total > 0.5:
            return "en"
        else:
            return "mixed"

    def get_ngrams(self, text: str, n: int = 2) -> List[str]:
        """
        Generate character n-grams from text.
        Used for wildcard query support (bigram index).

        CO2 Alignment: n-gram query formation.
        """
        tokens = self.preprocess(text)
        ngrams = []
        for token in tokens:
            padded = f"${token}$"
            for i in range(len(padded) - n + 1):
                ngrams.append(padded[i : i + n])
        return ngrams
