"""
=============================================================================
  Sarcasm-Aware Explainable Early Detection of Cyberbullying
  Module 3: LLM-Based Sarcasm Detection Engine
  IEEE Research Paper Implementation

  Architecture:
    ┌─────────────────────────────────────────────┐
    │  Input Text                                  │
    │       │                                       │
    │       ▼                                       │
    │  ┌──────────────────────────────┐            │
    │  │  LLM Sarcasm Detector        │            │
    │  │  (Gemini / GPT / Local BERT) │            │
    │  └──────────────────────────────┘            │
    │       │                                       │
    │       ▼                                       │
    │  sarcasm_score ∈ [0, 1]                       │
    │  sarcasm_explanation (natural language)       │
    │  sarcasm_indicators (list of cues)            │
    └─────────────────────────────────────────────┘
=============================================================================
"""

import os
import re
import json
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from config import (
    LLM_PROVIDER, LLM_MODEL, GEMINI_API_KEY, OPENAI_API_KEY,
    LOCAL_SARCASM_MODEL, SARCASM_THRESHOLD
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SarcasmResult:
    """Structured output from sarcasm detection."""
    text               : str
    sarcasm_score      : float          # probability ∈ [0, 1]
    is_sarcastic       : bool
    confidence         : str            # "high" | "medium" | "low"
    explanation        : str            # LLM-generated reasoning
    indicators         : List[str]      # specific sarcastic cues found
    provider           : str            # which model/provider was used
    latency_ms         : float = 0.0

    def to_dict(self) -> Dict:
        return {
            "sarcasm_score" : round(self.sarcasm_score, 4),
            "is_sarcastic"  : self.is_sarcastic,
            "confidence"    : self.confidence,
            "explanation"   : self.explanation,
            "indicators"    : self.indicators,
            "provider"      : self.provider,
            "latency_ms"    : round(self.latency_ms, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROMPT ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

SARCASM_SYSTEM_PROMPT = """You are an expert computational linguist specialising in
Hinglish (Hindi-English code-mixed) social media text analysis, particularly
sarcasm and irony detection in the context of cyberbullying.

Hinglish is written in Roman script and mixes Hindi and English freely.
Examples of sarcastic Hinglish cyberbullying:
  - "Wah wah! Kya genius hai tu. Tabhi toh fail ho gaya."
  - "Haan haan, bahut talented hai tu. Tabhi koi nahi sunta."
  - "Arre waah! SOOOO smart. Sirf 3 din lage 2+2 samajhne mein."
  - "Badhai ho! Tu nahi sudharega kabhi."
  - "Oh wow kya confidence hai! Mirror dekh kabhi."

You must respond in valid JSON with this exact schema:
{
  "sarcasm_score": <float between 0.0 and 1.0>,
  "is_sarcastic": <true|false>,
  "confidence": <"high"|"medium"|"low">,
  "explanation": "<one concise sentence explaining your reasoning>",
  "indicators": ["<cue1>", "<cue2>", ...]
}

Hinglish Sarcasm Guidelines:
- sarcasm_score: 0.0 = definitely sincere; 1.0 = definitely sarcastic/ironic
- Common Hinglish sarcastic openers: 'wah wah', 'haan haan', 'arre waah',
  'kya baat hai', 'badhai ho', 'janaab', 'expert aa gaye', 'bilkul sahi',
  'oh wow', 'oh sure bhai', 'haan sure'
- Watch for: exaggerated praise (shandar, kamaal, genius) followed by insults
- CAPITAL LETTERS in Hinglish (BAHUT, SOOOO, ITNA) signal exaggerated sarcasm
- Code-switching mid-sentence often signals irony (mixing English insults into Hindi praise)
- Differentiate benign situational sarcasm (about weather, traffic, internet)
  from weapon-sarcasm targeting a specific person
- indicators: specific Hinglish phrases or patterns that signal sarcasm
- Return ONLY the JSON object, no extra text."""

SARCASM_USER_PROMPT_TEMPLATE = """Analyse the following Hinglish (Hindi-English code-mixed)
social media text for sarcasm:

TEXT: "{text}"

Respond with the JSON schema described."""


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOCAL BERT-BASED SARCASM DETECTOR (Fallback)
# ─────────────────────────────────────────────────────────────────────────────

class LocalSarcasmDetector:
    """
    Uses cardiffnlp/twitter-roberta-base-irony (fine-tuned on Twitter irony)
    as a lightweight local fallback when no LLM API key is available.
    """

    def __init__(self, model_name: str = LOCAL_SARCASM_MODEL):
        logger.info(f"Loading local sarcasm model: {model_name}")
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=0 if torch.cuda.is_available() else -1,
            return_all_scores=True,
        )
        self.provider = f"local:{model_name.split('/')[-1]}"
        logger.info("Local sarcasm model loaded.")

    def detect(self, text: str) -> SarcasmResult:
        t0 = time.time()
        results = self.classifier(text[:512])[0]  # truncate for safety

        # cardiffnlp irony model has labels: irony | non_irony
        score_map = {r["label"].lower(): r["score"] for r in results}
        sarcasm_score = score_map.get("irony", score_map.get("label_1", 0.5))

        is_sarcastic = sarcasm_score >= SARCASM_THRESHOLD
        confidence   = (
            "high"   if sarcasm_score > 0.75 else
            "medium" if sarcasm_score > 0.5  else
            "low"
        )
        latency = (time.time() - t0) * 1000

        return SarcasmResult(
            text          = text,
            sarcasm_score = float(sarcasm_score),
            is_sarcastic  = is_sarcastic,
            confidence    = confidence,
            explanation   = (
                "Local model detected irony/sarcasm patterns." if is_sarcastic
                else "Text appears sincere according to local model."
            ),
            indicators    = self._extract_surface_cues(text),
            provider      = self.provider,
            latency_ms    = latency,
        )

    @staticmethod
    def _extract_surface_cues(text: str) -> List[str]:
        """Heuristic Hinglish + English sarcasm cue extraction."""
        cues = []
        text_low = text.lower()

        # English cues
        if re.search(r"\b[A-Z]{3,}\b", text):
            cues.append("ALL-CAPS word(s)")
        if text.count("!") >= 2:
            cues.append("Multiple exclamation marks")
        if re.search(
            r"\b(oh wow|sure|yeah right|brilliant|great job|"
            r"congratulations|amazing|so smart|totally|absolutely)\b",
            text_low,
        ):
            cues.append("Sarcastic English trigger phrase")

        # Hinglish cues
        hinglish_triggers = [
            (r"\bwah wah\b",       "'wah wah' (mock praise)"),
            (r"\bhaan haan\b",     "'haan haan' (dismissive agreement)"),
            (r"\bbadhai ho\b",     "'badhai ho' (mock congratulation)"),
            (r"\barre waah\b",     "'arre waah' (exaggerated awe)"),
            (r"\bkya genius\b",    "'kya genius' (ironic label)"),
            (r"\bexpert aa gaye\b","'expert aa gaye' (dismissive)"),
            (r"\bjanaab\b",        "'janaab' (mock formal address)"),
            (r"\bkya baat hai\b",  "'kya baat hai' (ironic praise)"),
            (r"\bbilkul sahi\b",   "'bilkul sahi' (sarcastic agreement)"),
            (r"\bshandar\b",       "'shandar' used ironically"),
        ]
        for pattern, label in hinglish_triggers:
            if re.search(pattern, text_low):
                cues.append(label)

        if re.search(r"SOOO+|BAHUT|ITNA|REALL?Y+", text):
            cues.append("Exaggerated Hinglish intensifier (CAPS)")
        return cues

    def batch_detect(self, texts: List[str]) -> List[SarcasmResult]:
        return [self.detect(t) for t in texts]


# ─────────────────────────────────────────────────────────────────────────────
# 4. GEMINI LLM SARCASM DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class GeminiSarcasmDetector:
    """
    Uses Google Gemini API for zero-shot sarcasm detection with
    structured JSON output and rich linguistic explanations.
    """

    def __init__(self, model_name: str = LLM_MODEL, api_key: str = GEMINI_API_KEY):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "temperature"      : 0.1,   # low temp for consistent outputs
                    "top_p"            : 0.9,
                    "max_output_tokens": 512,
                },
                system_instruction=SARCASM_SYSTEM_PROMPT,
            )
            self.provider = f"gemini:{model_name}"
            logger.info(f"Gemini sarcasm detector initialised with {model_name}")
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. Run: pip install google-generativeai"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialise Gemini: {e}")

    def detect(self, text: str, retries: int = 3) -> SarcasmResult:
        prompt = SARCASM_USER_PROMPT_TEMPLATE.format(text=text[:1000])
        t0 = time.time()

        for attempt in range(retries):
            try:
                response = self.model.generate_content(prompt)
                raw      = response.text.strip()

                # Robustly extract JSON from response
                json_match = re.search(r"\{.*\}", raw, re.DOTALL)
                if not json_match:
                    raise ValueError(f"No JSON found in response: {raw}")
                data = json.loads(json_match.group())

                latency = (time.time() - t0) * 1000
                score   = float(data.get("sarcasm_score", 0.5))
                return SarcasmResult(
                    text          = text,
                    sarcasm_score = score,
                    is_sarcastic  = bool(data.get("is_sarcastic", score >= SARCASM_THRESHOLD)),
                    confidence    = str(data.get("confidence", "medium")),
                    explanation   = str(data.get("explanation", "")),
                    indicators    = list(data.get("indicators", [])),
                    provider      = self.provider,
                    latency_ms    = latency,
                )

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Attempt {attempt+1}/{retries}: JSON parse error: {e}")
                time.sleep(1.5 ** attempt)    # exponential back-off
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{retries}: API error: {e}")
                time.sleep(2 ** attempt)

        # All retries exhausted – return neutral result
        logger.error(f"All retries failed for text: {text[:80]}…")
        return SarcasmResult(
            text=text, sarcasm_score=0.5, is_sarcastic=False,
            confidence="low", explanation="API call failed after retries.",
            indicators=[], provider=self.provider,
            latency_ms=(time.time() - t0) * 1000,
        )

    def batch_detect(
        self,
        texts: List[str],
        delay_ms: float = 200
    ) -> List[SarcasmResult]:
        results = []
        for i, text in enumerate(texts):
            result = self.detect(text)
            results.append(result)
            if delay_ms > 0 and i < len(texts) - 1:
                time.sleep(delay_ms / 1000)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. OPENAI LLM SARCASM DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class OpenAISarcasmDetector:
    """
    Uses OpenAI ChatCompletion API (GPT-4o-mini or GPT-3.5-turbo)
    for structured sarcasm detection.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = OPENAI_API_KEY):
        try:
            from openai import OpenAI
            self.client   = OpenAI(api_key=api_key)
            self.model    = model_name
            self.provider = f"openai:{model_name}"
            logger.info(f"OpenAI sarcasm detector initialised with {model_name}")
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")

    def detect(self, text: str, retries: int = 3) -> SarcasmResult:
        prompt = SARCASM_USER_PROMPT_TEMPLATE.format(text=text[:1000])
        t0 = time.time()

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SARCASM_SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=512,
                    response_format={"type": "json_object"},
                )
                raw  = response.choices[0].message.content.strip()
                data = json.loads(raw)

                latency = (time.time() - t0) * 1000
                score   = float(data.get("sarcasm_score", 0.5))
                return SarcasmResult(
                    text          = text,
                    sarcasm_score = score,
                    is_sarcastic  = bool(data.get("is_sarcastic", score >= SARCASM_THRESHOLD)),
                    confidence    = str(data.get("confidence", "medium")),
                    explanation   = str(data.get("explanation", "")),
                    indicators    = list(data.get("indicators", [])),
                    provider      = self.provider,
                    latency_ms    = latency,
                )
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{retries} failed: {e}")
                time.sleep(2 ** attempt)

        return SarcasmResult(
            text=text, sarcasm_score=0.5, is_sarcastic=False,
            confidence="low", explanation="API call failed.",
            indicators=[], provider=self.provider,
            latency_ms=(time.time() - t0) * 1000,
        )

    def batch_detect(self, texts: List[str], delay_ms: float = 300) -> List[SarcasmResult]:
        results = []
        for i, text in enumerate(texts):
            results.append(self.detect(text))
            if delay_ms > 0 and i < len(texts) - 1:
                time.sleep(delay_ms / 1000)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. UNIFIED SARCASM ENGINE (Factory + Cache)
# ─────────────────────────────────────────────────────────────────────────────

class SarcasmEngine:
    """
    Unified sarcasm detection engine with:
    - Automatic provider selection (Gemini > OpenAI > Local)
    - In-memory caching to avoid redundant API calls
    - Fallback chain for robustness
    """

    def __init__(
        self,
        provider: str = LLM_PROVIDER,
        use_cache: bool = True,
        fallback_to_local: bool = True,
    ):
        self._cache: Dict[str, SarcasmResult] = {}
        self._use_cache = use_cache
        self._fallback  = fallback_to_local
        self._local: Optional[LocalSarcasmDetector] = None
        self.primary_provider = provider

        self.detector = self._init_detector(provider)
        logger.info(f"SarcasmEngine ready. Provider: {provider}")

    def _init_detector(self, provider: str):
        if provider == "gemini" and GEMINI_API_KEY:
            try:
                return GeminiSarcasmDetector()
            except Exception as e:
                logger.warning(f"Gemini init failed ({e}). Falling back…")

        if provider in ("openai", "gemini") and OPENAI_API_KEY:
            try:
                return OpenAISarcasmDetector()
            except Exception as e:
                logger.warning(f"OpenAI init failed ({e}). Falling back…")

        logger.info("No LLM API key found. Using local BERT-based sarcasm model.")
        return LocalSarcasmDetector()

    def detect(self, text: str) -> SarcasmResult:
        if self._use_cache and text in self._cache:
            return self._cache[text]

        try:
            result = self.detector.detect(text)
        except Exception as e:
            logger.warning(f"Primary detector failed: {e}. Using local fallback.")
            if self._fallback:
                if self._local is None:
                    self._local = LocalSarcasmDetector()
                result = self._local.detect(text)
            else:
                raise

        if self._use_cache:
            self._cache[text] = result
        return result

    def batch_detect(self, texts: List[str]) -> List[SarcasmResult]:
        return [self.detect(t) for t in texts]

    def clear_cache(self):
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)


# ─────────────────────────────────────────────────────────────────────────────
# 7. QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_texts = [
        "Oh wow, you're SO smart! Only took you three days to figure that out.",
        "You are absolute garbage and everyone would be happier without you.",
        "I love your new project, you've worked really hard on it!",
        "Sure, your code is 'great'. That's why it crashes every single time.",
        "Have a wonderful day! Keep up the amazing work.",
    ]

    engine = SarcasmEngine(provider="local")  # use local for offline testing
    print("\n" + "="*70)
    print("  SARCASM ENGINE — DETECTION RESULTS")
    print("="*70)
    for text in test_texts:
        result = engine.detect(text)
        icon   = "🎭" if result.is_sarcastic else "💬"
        print(f"\n{icon} Text    : {text[:70]}…" if len(text) > 70 else f"\n{icon} Text    : {text}")
        print(f"   Score   : {result.sarcasm_score:.3f}  ({result.confidence} confidence)")
        print(f"   Sarcasm : {result.is_sarcastic}")
        print(f"   Cues    : {', '.join(result.indicators) or 'None detected'}")
        print(f"   Source  : {result.provider}")
