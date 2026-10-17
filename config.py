"""
=============================================================================
  Sarcasm-Aware Explainable Early Detection of Cyberbullying
  Module 1: Configuration & Constants
  IEEE Research Paper Implementation
=============================================================================
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# API KEYS (set via environment or .env file)
# ─────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
HF_TOKEN          = os.getenv("HF_TOKEN", "")          # HuggingFace token

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIGURATION  [Hinglish / Code-Mixed Edition]
# ─────────────────────────────────────────────────────────────────────────────
# Primary cyberbullying classifier backbone
# MuRIL: Google's multilingual model trained on 17 Indian languages
# + transliterated text — best for Hinglish code-mixed tasks.
# Alternatives (ranked for Hinglish): MuRIL > XLM-RoBERTa > IndicBERT
CYBERBULLYING_MODEL  = "google/muril-base-cased"     # PRIMARY (Hinglish)
# CYBERBULLYING_MODEL  = "xlm-roberta-base"           # Alternative 1
# CYBERBULLYING_MODEL  = "ai4bharat/indic-bert"       # Alternative 2
TOKENIZER_MAX_LEN    = 128

# ── Hinglish-specific settings ────────────────────────────────────────────
LANGUAGE_MODE   = "hinglish"    # triggers Hinglish-aware preprocessing
SCRIPT          = "roman"       # Hinglish in Roman script (not Devanagari)
# Set True to apply Devanagari→Roman transliteration on mixed-script inputs
TRANSLITERATE   = False

# LLM for sarcasm detection (choose one)
LLM_PROVIDER  = "gemini"        # Options: "gemini", "openai", "local"
LLM_MODEL     = "gemini-1.5-flash"   # Gemini handles Hinglish natively

# Local fallback sarcasm model
# xlm-roberta-large-sarcasm (multilingual, handles Hinglish better than
# English-only twitter-roberta-base-irony)
LOCAL_SARCASM_MODEL = "cardiffnlp/xlm-roberta-base-tweet-sentiment-multilingual"

# ─────────────────────────────────────────────────────────────────────────────
# EARLY DETECTION CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# "Early" means: detect within the FIRST K messages of a conversation
EARLY_DETECTION_K = 3

# Cyberbullying decision threshold
CYBERBULLYING_THRESHOLD = 0.55

# Sarcasm confidence threshold above which we flip or boost bully score
SARCASM_THRESHOLD = 0.45

# Sarcasm boost factor applied to cyberbullying probability
SARCASM_BOOST = 0.18

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
TRAINING_CONFIG = {
    "output_dir"                   : "./model_checkpoints",
    "num_train_epochs"             : 5,
    "per_device_train_batch_size"  : 16,
    "per_device_eval_batch_size"   : 32,
    "learning_rate"                : 2e-5,
    "weight_decay"                 : 0.01,
    "warmup_ratio"                 : 0.1,
    "lr_scheduler_type"            : "cosine",
    "eval_strategy"                : "epoch",
    "save_strategy"                : "epoch",
    "load_best_model_at_end"       : True,
    "metric_for_best_model"        : "f1",
    "fp16"                         : True,
    "seed"                         : 42,
    "logging_steps"                : 50,
    "report_to"                    : "none",
}

# ─────────────────────────────────────────────────────────────────────────────
# EXPLAINABILITY CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SHAP_MAX_EVALS     = 500      # Higher = more accurate but slower
SHAP_BATCH_SIZE    = 16
TOP_K_TOKENS       = 15       # Top influential tokens to show

# ─────────────────────────────────────────────────────────────────────────────
# DATASET CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR         = "./data"
RESULTS_DIR      = "./results"
PLOTS_DIR        = "./plots"
MODELS_DIR       = "./saved_models"

# Supported platforms for multi-platform corpus (Indian social media focus)
PLATFORMS = ["twitter", "instagram", "youtube", "sharechat", "moj", "reddit"]

# Label mapping
LABEL2ID = {"non_cyberbullying": 0, "cyberbullying": 1}
ID2LABEL = {0: "non_cyberbullying", 1: "cyberbullying"}

# Hinglish-specific label categories (for multi-class variant)
HINGLISH_CATEGORIES = [
    "non_cyberbullying",
    "mockery",           # mazak ura raha hai
    "abusive",           # gaaliyan
    "threat",            # dhamki
    "religion_hate",     # dharm pe attack
    "gender_hate",       # gender discrimination
]

# Random seed for reproducibility
SEED = 42
