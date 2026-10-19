"""
=============================================================================
  Sarcasm-Aware Explainable Early Detection of Cyberbullying
  Module 4: Main Detection Model (SACENet)
  IEEE Research Paper Implementation

  Architecture — SACENet (Sarcasm-Aware Cyberbullying Early-detection Network):
  ┌─────────────────────────────────────────────────────────────────────┐
  │                         SACENet Pipeline                            │
  │                                                                      │
  │  Input Text                                                          │
  │      │                                                               │
  │      ├──────────────────────────────────────────────────────────┐   │
  │      │                                                           │   │
  │      ▼                                                           ▼   │
  │  ┌──────────────────┐                          ┌──────────────────┐  │
  │  │ Twitter-RoBERTa  │                          │  LLM Sarcasm     │  │
  │  │ (Cyberbullying   │                          │  Engine          │  │
  │  │  Classifier)     │                          │  (Gemini/GPT/    │  │
  │  └──────────────────┘                          │  Local-BERT)     │  │
  │      │                                         └──────────────────┘  │
  │      │ P(bully)                                     │ sarcasm_score  │
  │      │                                              │                │
  │      └──────────────┬───────────────────────────────┘               │
  │                     │                                                │
  │                     ▼                                                │
  │         ┌──────────────────────────┐                                │
  │         │  Sarcasm-Aware Fusion    │                                │
  │         │  Score Combiner          │                                │
  │         │  (calibrated boost)      │                                │
  │         └──────────────────────────┘                                │
  │                     │                                                │
  │                     ▼                                                │
  │         ┌──────────────────────────┐                                │
  │         │  SHAP Explainer          │                                │
  │         │  (token attribution)     │                                │
  │         └──────────────────────────┘                                │
  │                     │                                                │
  │                     ▼                                                │
  │         ┌──────────────────────────┐                                │
  │         │  Structured Output       │                                │
  │         │  DetectionResult         │                                │
  │         └──────────────────────────┘                                │
  └─────────────────────────────────────────────────────────────────────┘
=============================================================================
"""

import os
import json
import time
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

import torch
import numpy as np
import shap
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline as hf_pipeline,
)

from config import (
    CYBERBULLYING_MODEL, CYBERBULLYING_THRESHOLD, SARCASM_BOOST,
    TOKENIZER_MAX_LEN, SHAP_MAX_EVALS, SHAP_BATCH_SIZE, TOP_K_TOKENS,
    ID2LABEL, MODELS_DIR, EARLY_DETECTION_K,
)
from sarcasm_engine import SarcasmEngine, SarcasmResult

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenAttribution:
    """SHAP attribution for a single token."""
    token      : str
    shap_value : float
    position   : int


@dataclass
class ExplanationResult:
    """Full SHAP explainability output."""
    top_tokens         : List[TokenAttribution]
    base_value         : float
    predicted_class    : int
    class_name         : str
    shap_narrative     : str          # human-readable explanation
    sarcasm_narrative  : str          # why sarcasm affected prediction


@dataclass
class DetectionResult:
    """
    Complete detection output for a single text input.
    Includes cyberbullying prediction, sarcasm analysis,
    and SHAP-based token-level explanations.
    """
    # Input
    text                    : str
    platform                : str = "unknown"

    # Cyberbullying prediction
    cyberbullying_prob      : float = 0.0
    base_cyberbullying_prob : float = 0.0      # before sarcasm boost
    is_cyberbullying        : bool = False
    prediction_label        : str = "non_cyberbullying"
    confidence              : str = "low"

    # Sarcasm analysis
    sarcasm_result          : Optional[SarcasmResult] = None
    sarcasm_was_applied     : bool = False
    sarcasm_boost_applied   : float = 0.0

    # Explainability
    explanation             : Optional[ExplanationResult] = None

    # Metadata
    detection_time_ms       : float = 0.0
    model_version           : str = "SACENet-v1.0"

    def to_dict(self) -> Dict:
        d = {
            "text"                    : self.text,
            "platform"                : self.platform,
            "cyberbullying_prob"      : round(self.cyberbullying_prob, 4),
            "base_cyberbullying_prob" : round(self.base_cyberbullying_prob, 4),
            "is_cyberbullying"        : self.is_cyberbullying,
            "prediction_label"        : self.prediction_label,
            "confidence"              : self.confidence,
            "sarcasm_score"           : round(
                self.sarcasm_result.sarcasm_score if self.sarcasm_result else 0.0, 4
            ),
            "is_sarcastic"            : (
                self.sarcasm_result.is_sarcastic if self.sarcasm_result else False
            ),
            "sarcasm_boost_applied"   : round(self.sarcasm_boost_applied, 4),
            "sarcasm_explanation"     : (
                self.sarcasm_result.explanation if self.sarcasm_result else ""
            ),
            "sarcasm_indicators"      : (
                self.sarcasm_result.indicators if self.sarcasm_result else []
            ),
            "detection_time_ms"       : round(self.detection_time_ms, 1),
            "model_version"           : self.model_version,
        }
        if self.explanation:
            d["shap_explanation"] = {
                "top_tokens"        : [
                    {"token": t.token, "shap_value": round(t.shap_value, 4)}
                    for t in self.explanation.top_tokens
                ],
                "base_value"        : round(self.explanation.base_value, 4),
                "shap_narrative"    : self.explanation.shap_narrative,
                "sarcasm_narrative" : self.explanation.sarcasm_narrative,
            }
        return d

    def summary(self) -> str:
        icon = "🚨" if self.is_cyberbullying else "✅"
        sar  = f"  🎭 Sarcasm boost (+{self.sarcasm_boost_applied:.3f})" \
               if self.sarcasm_was_applied else ""
        return (
            f"{icon} PREDICTION : {self.prediction_label.upper()}  "
            f"(prob={self.cyberbullying_prob:.3f}, {self.confidence} confidence)\n"
            f"   Base prob  : {self.base_cyberbullying_prob:.3f}{sar}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. CYBERBULLYING CLASSIFIER (Fine-tuned RoBERTa)
# ─────────────────────────────────────────────────────────────────────────────

class CyberbullyingClassifier:
    """
    Wraps a fine-tuned Twitter-RoBERTa model for cyberbullying classification.
    Falls back to zero-shot HuggingFace pipeline if fine-tuned weights missing.
    """

    def __init__(
        self,
        model_name: str = CYBERBULLYING_MODEL,
        finetuned_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name

        # Prefer fine-tuned weights if available
        load_path = (
            finetuned_path
            if finetuned_path and os.path.isdir(finetuned_path)
            else model_name
        )
        logger.info(f"Loading cyberbullying classifier from: {load_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            load_path, num_labels=2
        ).to(device)
        self.model.eval()
        logger.info("Cyberbullying classifier ready.")

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Return probability matrix of shape (N, 2): [P(benign), P(bully)]."""
        all_probs = []
        batch_size = SHAP_BATCH_SIZE

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc   = self.tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=TOKENIZER_MAX_LEN,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**enc).logits
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

        return np.vstack(all_probs)

    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Return (predicted_labels, probabilities_for_bully_class)."""
        probs    = self.predict_proba(texts)
        bully_p  = probs[:, 1]
        labels   = (bully_p >= CYBERBULLYING_THRESHOLD).astype(int)
        return labels, bully_p


# ─────────────────────────────────────────────────────────────────────────────
# 3. SARCASM-AWARE SCORE FUSION
# ─────────────────────────────────────────────────────────────────────────────

def fuse_scores(
    bully_prob    : float,
    sarcasm_result: SarcasmResult,
    boost         : float = SARCASM_BOOST,
) -> Tuple[float, float, bool]:
    """
    Combine cyberbullying probability with sarcasm score.

    When sarcasm is detected at high confidence, the bullying probability
    is boosted (sarcastic bullying is often misclassified as benign).

    Args:
        bully_prob     : P(cyberbullying) from classifier ∈ [0, 1]
        sarcasm_result : SarcasmResult from engine
        boost          : maximum additive boost applied when sarcasm=1.0

    Returns:
        (fused_prob, boost_applied, was_boosted)
    """
    if not sarcasm_result.is_sarcastic:
        return bully_prob, 0.0, False

    # Scale boost by sarcasm confidence score
    dynamic_boost  = boost * sarcasm_result.sarcasm_score
    fused          = min(1.0, bully_prob + dynamic_boost)
    return fused, dynamic_boost, True


# ─────────────────────────────────────────────────────────────────────────────
# 4. SHAP EXPLAINER
# ─────────────────────────────────────────────────────────────────────────────

class SHAPExplainer:
    """
    Token-level SHAP explanations using the Partition explainer
    (works best with transformer models on text data).
    """

    def __init__(self, classifier: CyberbullyingClassifier):
        self.classifier = classifier
        self._explainer: Optional[shap.Explainer] = None
        logger.info("SHAP explainer initialised (lazy load).")

    def _get_explainer(self) -> shap.Explainer:
        if self._explainer is None:
            logger.info("Building SHAP partition explainer …")

            def predict_fn(texts):
                # SHAP passes numpy arrays of strings; convert to list
                if isinstance(texts, np.ndarray):
                    texts = texts.tolist()
                return self.classifier.predict_proba(texts)

            self._explainer = shap.Explainer(
                predict_fn,
                self.classifier.tokenizer,
                output_names=["non_cyberbullying", "cyberbullying"],
            )
        return self._explainer

    def explain(
        self,
        text            : str,
        sarcasm_result  : Optional[SarcasmResult] = None,
        max_evals       : int = SHAP_MAX_EVALS,
        top_k           : int = TOP_K_TOKENS,
        pred_label      : int = 1,
    ) -> ExplanationResult:
        """
        Compute SHAP values for a single text and return
        a structured ExplanationResult.
        """
        explainer   = self._get_explainer()
        shap_values = explainer([text], max_evals=max_evals)

        # shap_values[0].values shape: (n_tokens, n_classes)
        vals    = shap_values[0].values[:, pred_label]
        tokens  = shap_values[0].data
        base_v  = float(shap_values[0].base_values[pred_label])

        # Build sorted token attribution list
        attributions = [
            TokenAttribution(token=str(t), shap_value=float(v), position=i)
            for i, (t, v) in enumerate(zip(tokens, vals))
        ]
        top_tokens = sorted(attributions, key=lambda x: abs(x.shap_value), reverse=True)[:top_k]

        # Build narrative
        positive = [a for a in top_tokens if a.shap_value > 0][:5]
        negative = [a for a in top_tokens if a.shap_value < 0][:3]

        pos_str  = ", ".join(f'"{a.token}"' for a in positive) or "none"
        neg_str  = ", ".join(f'"{a.token}"' for a in negative) or "none"

        class_name = ID2LABEL.get(pred_label, str(pred_label))
        narrative  = (
            f"The model predicted '{class_name}' primarily because of tokens: "
            f"{pos_str}. These tokens pushed the prediction towards "
            f"cyberbullying. Tokens {neg_str} pushed against the prediction."
        )

        # Sarcasm narrative
        sarcasm_narrative = ""
        if sarcasm_result and sarcasm_result.is_sarcastic:
            indicators = ", ".join(sarcasm_result.indicators) or "exaggerated language"
            sarcasm_narrative = (
                f"⚠️ Sarcasm detected (score={sarcasm_result.sarcasm_score:.2f}): "
                f"{sarcasm_result.explanation} "
                f"Key sarcastic cues: {indicators}. "
                f"The cyberbullying score was boosted to account for covert toxicity."
            )

        return ExplanationResult(
            top_tokens        = top_tokens,
            base_value        = base_v,
            predicted_class   = pred_label,
            class_name        = class_name,
            shap_narrative    = narrative,
            sarcasm_narrative = sarcasm_narrative,
        )

    def visualise(self, shap_values, notebook: bool = True):
        """Render SHAP text plot (call inside Jupyter notebook)."""
        try:
            if notebook:
                shap.plots.text(shap_values)
            else:
                shap.summary_plot(shap_values)
        except Exception as e:
            logger.warning(f"SHAP visualisation failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN SACENet PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class SACENet:
    """
    SACENet — Sarcasm-Aware Cyberbullying Early-detection Network

    End-to-end pipeline combining:
    1. LLM-based sarcasm detection (Gemini / GPT / Local-BERT)
    2. Twitter-RoBERTa cyberbullying classifier
    3. Sarcasm-aware score fusion
    4. SHAP token-level explainability
    5. Early detection (analyse first-K messages of a conversation)

    Usage:
        model = SACENet()
        result = model.detect("Oh wow, you're SO brilliant! No wonder you failed.")
        print(result.summary())
        print(result.to_dict())
    """

    VERSION = "SACENet-v1.0"

    def __init__(
        self,
        cyberbullying_model : str = CYBERBULLYING_MODEL,
        finetuned_path      : Optional[str] = None,
        sarcasm_provider    : str = "local",   # start with local; switch to gemini/openai
        enable_shap         : bool = True,
        device              : Optional[str] = None,
    ):
        logger.info("="*60)
        logger.info(f"  Initialising SACENet {self.VERSION}")
        logger.info("="*60)

        # Core components
        self.classifier = CyberbullyingClassifier(
            model_name=cyberbullying_model,
            finetuned_path=finetuned_path,
            device=device,
        )
        self.sarcasm_engine = SarcasmEngine(provider=sarcasm_provider)
        self.enable_shap    = enable_shap
        self._explainer     = SHAPExplainer(self.classifier) if enable_shap else None

        logger.info("SACENet initialisation complete.")

    # ── Core Detection ────────────────────────────────────────────────────────

    def detect(
        self,
        text          : str,
        platform      : str = "unknown",
        explain       : bool = True,
    ) -> DetectionResult:
        """
        Detect cyberbullying in a single text with sarcasm-aware adjustment
        and SHAP explanation.
        """
        t_start = time.time()

        # 1. LLM sarcasm detection
        sarcasm_result = self.sarcasm_engine.detect(text)

        # 2. Cyberbullying classification
        _, bully_probs   = self.classifier.predict([text])
        base_bully_prob  = float(bully_probs[0])

        # 3. Sarcasm-aware score fusion
        fused_prob, boost_val, was_boosted = fuse_scores(
            base_bully_prob, sarcasm_result
        )

        # 4. Final decision
        is_bully = fused_prob >= CYBERBULLYING_THRESHOLD
        label    = "cyberbullying" if is_bully else "non_cyberbullying"
        conf     = (
            "high"   if fused_prob > 0.8  or (1 - fused_prob) > 0.8  else
            "medium" if fused_prob > 0.65 or (1 - fused_prob) > 0.65 else
            "low"
        )

        # 5. SHAP explanation (optional)
        explanation = None
        if explain and self._explainer:
            try:
                explanation = self._explainer.explain(
                    text, sarcasm_result, pred_label=int(is_bully)
                )
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")

        t_end = time.time()
        return DetectionResult(
            text                    = text,
            platform                = platform,
            cyberbullying_prob      = fused_prob,
            base_cyberbullying_prob = base_bully_prob,
            is_cyberbullying        = is_bully,
            prediction_label        = label,
            confidence              = conf,
            sarcasm_result          = sarcasm_result,
            sarcasm_was_applied     = was_boosted,
            sarcasm_boost_applied   = boost_val,
            explanation             = explanation,
            detection_time_ms       = (t_end - t_start) * 1000,
            model_version           = self.VERSION,
        )

    def batch_detect(
        self,
        texts     : List[str],
        platforms : Optional[List[str]] = None,
        explain   : bool = False,    # disable SHAP for batch (expensive)
    ) -> List[DetectionResult]:
        """Detect cyberbullying in a batch of texts."""
        if platforms is None:
            platforms = ["unknown"] * len(texts)
        return [
            self.detect(t, p, explain=explain)
            for t, p in zip(texts, platforms)
        ]

    # ── Early Detection ───────────────────────────────────────────────────────

    def early_detect_conversation(
        self,
        messages    : List[str],
        platform    : str = "unknown",
        k           : int = EARLY_DETECTION_K,
        explain     : bool = True,
    ) -> Dict[str, Any]:
        """
        Perform early detection on the first k messages of a conversation.
        Returns detection result + per-message risk scores.

        Args:
            messages : list of conversation messages (in chronological order)
            platform : social media platform
            k        : number of first messages to analyse
            explain  : whether to generate SHAP explanation

        Returns:
            dict with keys: 'early_result', 'per_message_results',
                            'risk_trajectory', 'escalation_detected'
        """
        window    = messages[:k]
        per_msg   = []
        risk_traj = []

        for msg in window:
            r = self.detect(msg, platform=platform, explain=False)
            per_msg.append(r)
            risk_traj.append(r.cyberbullying_prob)

        # Joint decision on concatenated window
        joined_text  = " [SEP] ".join(window)
        early_result = self.detect(joined_text, platform=platform, explain=explain)

        # Detect escalation (rising risk trajectory)
        escalation = (
            len(risk_traj) >= 2 and
            risk_traj[-1] > risk_traj[0] and
            (risk_traj[-1] - risk_traj[0]) > 0.15
        )

        return {
            "early_result"         : early_result,
            "per_message_results"  : per_msg,
            "risk_trajectory"      : risk_traj,
            "escalation_detected"  : escalation,
            "messages_analysed"    : len(window),
        }

    # ── Model Persistence ────────────────────────────────────────────────────

    def save_classifier(self, path: str = MODELS_DIR):
        os.makedirs(path, exist_ok=True)
        self.classifier.model.save_pretrained(path)
        self.classifier.tokenizer.save_pretrained(path)
        logger.info(f"Classifier saved to {path}")

    @classmethod
    def load(cls, path: str, **kwargs) -> "SACENet":
        return cls(finetuned_path=path, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 6. QUICK DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = SACENet(sarcasm_provider="local", enable_shap=False)

    test_cases = [
        ("Oh wow, you are SO smart. That's why you failed every test.",  "twitter"),
        ("You are absolute garbage, nobody likes you.",                  "formspring"),
        ("Great work on your project, I'm really impressed!",            "instagram"),
        ("Sure, your code is AMAZING. It crashes every single time.",    "reddit"),
        ("Keep up the good work, you're doing really well!",             "youtube"),
    ]

    print("\n" + "="*70)
    print("  SACENet — DETECTION DEMO (IEEE Research Prototype)")
    print("="*70)
    for text, platform in test_cases:
        result = model.detect(text, platform=platform, explain=False)
        print(f"\n📝 [{platform.upper()}] {text[:75]}…" if len(text) > 75
              else f"\n📝 [{platform.upper()}] {text}")
        print(f"   {result.summary()}")
