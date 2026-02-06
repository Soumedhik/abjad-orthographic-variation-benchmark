from __future__ import annotations

import random
import unicodedata
from typing import Dict, Iterable, List, Sequence

import pandas as pd
from tqdm.auto import tqdm

ARABIC_DIACRITICS = tuple(chr(c) for c in range(0x064B, 0x0653))

URDU_ROMAN = {"ا": "a", "آ": "aa", "ب": "b", "پ": "p", "ت": "t", "ٹ": "t", "ث": "s", "ج": "j", "چ": "ch", "ح": "h", "خ": "kh", "د": "d", "ڈ": "d", "ذ": "z", "ر": "r", "ڑ": "r", "ز": "z", "ژ": "zh", "س": "s", "ش": "sh", "ص": "s", "ض": "z", "ط": "t", "ظ": "z", "ع": "a", "غ": "gh", "ف": "f", "ق": "q", "ک": "k", "گ": "g", "ل": "l", "م": "m", "ن": "n", "ں": "n", "و": "w", "ؤ": "o", "ہ": "h", "ء": "", "ی": "y", "ے": "e", "ۓ": "e"}
PASHTO_ROMAN = {"ا": "a", "آ": "aa", "ب": "b", "پ": "p", "ت": "t", "ټ": "tt", "ث": "s", "ج": "j", "ځ": "dz", "چ": "ch", "ح": "h", "خ": "kh", "د": "d", "ډ": "dd", "ذ": "z", "ر": "r", "ړ": "rr", "ز": "z", "ژ": "zh", "ږ": "gh", "س": "s", "ش": "sh", "ښ": "x", "ص": "s", "ض": "z", "ط": "t", "ظ": "z", "ع": "a", "غ": "gh", "ف": "f", "ق": "q", "ک": "k", "ګ": "g", "گ": "g", "ل": "l", "م": "m", "ن": "n", "ڼ": "nn", "و": "w", "ؤ": "o", "ه": "h", "ۀ": "e", "ی": "y", "ې": "e", "ۍ": "ai"}


def strip_diacritics(text: str) -> str:
    """Remove all diacritical marks from Arabic text.
    
    Args:
        text: Input text with potential diacritics.
        
    Returns:
        Text with all Unicode combining marks (Mn category) removed.
    """
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def partial_diacritics(text: str) -> str:
    """Retain only final-position diacritical marks in Arabic text.
    
    This simulates natural Arabic writing where case markers are often
    retained while internal vowel marks are omitted.
    
    Args:
        text: Input Arabic text with diacritics.
        
    Returns:
        Text with only word-final diacritical marks preserved.
    """
    processed = []
    for tok in text.split():
        if not tok:
            continue
        last_base = None
        for i in range(len(tok) - 1, -1, -1):
            if unicodedata.category(tok[i]) != "Mn":
                last_base = i
                break
        trailing = "".join(
            ch
            for ch in tok[last_base + 1 :]
            if last_base is not None and unicodedata.category(ch) == "Mn"
        ) if last_base is not None else ""
        base = "".join(ch for ch in tok if unicodedata.category(ch) != "Mn")
        processed.append(base + trailing)
    return " ".join(processed)


def romanize(text: str, language: str) -> str:
    """Transliterate Perso-Arabic script to Latin script.
    
    Args:
        text: Input text in Perso-Arabic script.
        language: Language code ('ur' for Urdu, 'ps' for Pashto).
        
    Returns:
        Romanized text using language-specific character mappings.
    """
    table = URDU_ROMAN if language == "ur" else PASHTO_ROMAN
    return "".join(table.get(ch, ch) for ch in text)


def romanize_ratio(text: str, language: str, ratio: float, rng: random.Random) -> str:
    """Romanize a random subset of words at a specified rate.
    
    Args:
        text: Input text in Perso-Arabic script.
        language: Language code for romanization table.
        ratio: Proportion of words to romanize (0.0-1.0).
        rng: Random number generator for reproducibility.
        
    Returns:
        Text with randomly selected words romanized.
    """
    words = text.split()
    out = []
    for word in words:
        if rng.random() < ratio:
            out.append(romanize(word, language))
        else:
            out.append(word)
    return " ".join(out)


def mix_with_tokens(text: str, donor_tokens: Sequence[str], ratio: float, rng: random.Random) -> str:
    """Simulate code-switching by injecting donor language tokens.
    
    Args:
        text: Original text.
        donor_tokens: Pool of tokens from donor language.
        ratio: Proportion of words to replace (0.0-1.0).
        rng: Random number generator for reproducibility.
        
    Returns:
        Text with randomly replaced tokens creating mixed-script output.
    """
    words = text.split()
    out = []
    for word in words:
        if rng.random() < ratio and donor_tokens:
            out.append(rng.choice(donor_tokens))
        else:
            out.append(word)
    return " ".join(out)


def build_token_pool(sentences: Iterable[str]) -> List[str]:
    tokens: List[str] = []
    for sent in sentences:
        tokens.extend(sent.split())
    return tokens


def make_variants(
    df: pd.DataFrame,
    en_tokens: Sequence[str],
    ur_tokens: Sequence[str],
    rng: random.Random,
    romanize_ratios: Sequence[float] = (0.25, 0.5, 1.0),
    mix_ratios: Sequence[float] = (0.25, 0.5),
) -> pd.DataFrame:
    records: List[Dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        base = {
            "premise": row.premise,
            "hypothesis": row.hypothesis,
            "label": row.label_text,
            "language": row.language,
        }
        records.append({**base, "condition": "clean"})
        if row.language == "ar":
            records.append({
                **base,
                "premise": strip_diacritics(row.premise),
                "hypothesis": strip_diacritics(row.hypothesis),
                "condition": "no_diacritics",
            })
            records.append({
                **base,
                "premise": partial_diacritics(row.premise),
                "hypothesis": partial_diacritics(row.hypothesis),
                "condition": "partial_diacritics",
            })
        if row.language == "ur":
            for ratio in romanize_ratios:
                label = f"R{int(ratio * 100)}"
                records.append({
                    **base,
                    "premise": romanize_ratio(row.premise, "ur", ratio, rng),
                    "hypothesis": romanize_ratio(row.hypothesis, "ur", ratio, rng),
                    "condition": label,
                })
            for ratio in mix_ratios:
                label = f"M{int(ratio * 100)}"
                records.append({
                    **base,
                    "premise": mix_with_tokens(row.premise, en_tokens, ratio, rng),
                    "hypothesis": mix_with_tokens(row.hypothesis, en_tokens, ratio, rng),
                    "condition": label,
                })
        if row.language == "sw":
            records.append({**base, "condition": "romanized"})
            for ratio in mix_ratios:
                label = f"M{int(ratio * 100)}"
                records.append({
                    **base,
                    "premise": mix_with_tokens(row.premise, en_tokens, ratio, rng),
                    "hypothesis": mix_with_tokens(row.hypothesis, en_tokens, ratio, rng),
                    "condition": label,
                })
        if row.language == "en":
            for ratio in mix_ratios:
                label = f"M{int(ratio * 100)}"
                records.append({
                    **base,
                    "premise": mix_with_tokens(row.premise, ur_tokens, ratio, rng),
                    "hypothesis": mix_with_tokens(row.hypothesis, ur_tokens, ratio, rng),
                    "condition": label,
                })
    return pd.DataFrame.from_records(records)
