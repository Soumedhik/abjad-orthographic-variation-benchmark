from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import cycle
from typing import Dict, List

import requests

LABEL_ORDER = ["entailment", "neutral", "contradiction"]
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


@dataclass
class ModelSpec:
    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 4


def format_prompt(premise: str, hypothesis: str) -> List[Dict[str, str]]:
    """Format NLI premise-hypothesis pair as chat messages.
    
    Args:
        premise: The premise statement.
        hypothesis: The hypothesis statement.
        
    Returns:
        List of chat messages with system and user roles.
    """
    prompt = (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Answer with one of: entailment, neutral, contradiction."
    )
    return [
        {"role": "system", "content": "You are a precise NLI classifier."},
        {"role": "user", "content": prompt},
    ]


def post_with_retry(url: str, headers: Dict[str, str], payload: Dict, max_retries: int = 3) -> Dict:
    for attempt in range(max_retries):
        resp = requests.post(url, headers=headers, json=payload, timeout=45)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in {429, 503}:
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
    raise RuntimeError(f"Failed after {max_retries} retries")


def build_key_cycle(keys: List[str]) -> cycle:
    if not keys:
        raise ValueError("Set GROQ_API_KEYS in your environment or .env file.")
    return cycle(keys)


def call_groq(spec: ModelSpec, messages: List[Dict[str, str]], key_cycle: cycle) -> str:
    key = next(key_cycle)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": spec.model,
        "messages": messages,
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
    }
    data = post_with_retry(GROQ_URL, headers, payload)
    return data["choices"][0]["message"]["content"]


def run_model(spec: ModelSpec, premise: str, hypothesis: str, key_cycle: cycle) -> str:
    """Run NLI inference and extract normalized prediction.
    
    Args:
        spec: Model specification (provider, model name, parameters).
        premise: The premise text.
        hypothesis: The hypothesis text.
        key_cycle: Cycling iterator over API keys.
        
    Returns:
        Normalized prediction label (entailment, neutral, or contradiction).
    """
    messages = format_prompt(premise, hypothesis)
    raw = call_groq(spec, messages, key_cycle)
    text = raw.strip().lower()
    for label in LABEL_ORDER:
        if label in text:
            return label
    return text.split()[0] if text else "unknown"
