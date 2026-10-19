"""
=============================================================================
  Sarcasm-Aware Explainable Early Detection of Cyberbullying
  Module 2: Dataset Loader — MC-Hinglish1.0 + Sarcasm Features
  IEEE Research Paper Implementation

  Datasets used:
    1. MC-Hinglish1.0 (Train/Test)
       Columns: tweets | cyberbullying_type
       Labels : not_cyberbullying | mockery | abusive | offensive |
                religion | gender | age
       Size   : 7400 train + 1000 test

    2. human_labelled_features.csv
       Columns: comment | human_labelled | ai_labelled | emoji_present | hindi_pct
       Labels : sarcastic | non-sarcastic
       Size   : 8885 rows (ground-truth sarcasm annotations)

    3. ai_labelled_features.csv
       Columns: comment | ai_labelled | emoji_present | hindi_pct
       Labels : sarcastic | non-sarcastic
       Size   : 12809 rows (LLM sarcasm annotations)
=============================================================================
"""

import os
import re
import json
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from config import (
    DATA_DIR, SEED, LABEL2ID, ID2LABEL, TOKENIZER_MAX_LEN,
    CYBERBULLYING_MODEL, LANGUAGE_MODE, TRANSLITERATE
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 0. DATASET FILE PATHS (relative to project root)
# ─────────────────────────────────────────────────────────────────────────────
MC_TRAIN_PATH   = os.path.join(DATA_DIR, "mc_hinglish_train.csv")
MC_TEST_PATH    = os.path.join(DATA_DIR, "mc_hinglish_test.csv")
HUMAN_SAR_PATH  = os.path.join(DATA_DIR, "human_sarcasm_features.csv")
AI_SAR_PATH     = os.path.join(DATA_DIR, "ai_sarcasm_features.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LABEL MAPPINGS
# ─────────────────────────────────────────────────────────────────────────────

# MC-Hinglish1.0 multi-class labels → binary bully/non-bully
MULTICLASS_TO_BINARY = {
    "not_cyberbullying": 0,
    "mockery"          : 1,
    "abusive"          : 1,
    "offensive"        : 1,
    "religion"         : 1,
    "gender"           : 1,
    "age"              : 1,
}

# Numeric encoding for the 7-class classification (for multi-class variant)
MULTICLASS_LABEL2ID = {
    "not_cyberbullying": 0,
    "mockery"          : 1,
    "abusive"          : 2,
    "offensive"        : 3,
    "religion"         : 4,
    "gender"           : 5,
    "age"              : 6,
}
MULTICLASS_ID2LABEL = {v: k for k, v in MULTICLASS_LABEL2ID.items()}

# Sarcasm label normalisation
def normalise_sarcasm_label(label: str) -> int:
    """Convert raw sarcasm string labels to binary 0/1."""
    l = str(label).strip().lower()
    if l in ("sarcastic",):
        return 1
    if l in ("non-sarcastic", "non_sarcastic"):
        return 0
    return -1   # error / unknown


# ─────────────────────────────────────────────────────────────────────────────
# 2. HINGLISH TEXT PREPROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

class TextPreprocessor:
    """Clean and normalise Hinglish code-mixed social-media text."""

    EMOJI_PATTERN      = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F9FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    # Named emoji text tokens (e.g., "face_with_tears_of_joy", "cross_mark")
    NAMED_EMOJI_PATTERN = re.compile(r"\b[a-z]+(?:_[a-z]+){2,}\b")
    URL_PATTERN         = re.compile(r"http\S+|www\.\S+")
    MENTION_PATTERN     = re.compile(r"@\w+")
    HASHTAG_PATTERN     = re.compile(r"#(\w+)")
    DEVANAGARI_PATTERN  = re.compile(r"[\u0900-\u097F]+")
    HIND_ELONGATION     = re.compile(r"(.)\1{2,}")   # yaaaaar → yaaar

    # Hinglish sarcasm trigger phrases (Roman script)
    SARCASM_TRIGGERS_HI = [
        r"\bwah wah\b", r"\bkya baat\b", r"\bbadhai ho\b",
        r"\bkya genius\b", r"\bexpert aa gaye\b", r"\bshandar\b",
        r"\bkamaal\b", r"\bjanaab\b", r"\bhaan haan\b",
        r"\bhaan sure\b", r"\bbilkul sahi\b", r"\boh wow\b",
        r"\barre waah\b", r"\bkya confidence\b",
    ]

    @classmethod
    def has_devanagari(cls, text: str) -> bool:
        return bool(cls.DEVANAGARI_PATTERN.search(text))

    @classmethod
    def clean(cls, text: str, preserve_caps: bool = True) -> str:
        """Full Hinglish-aware cleaning pipeline."""
        if not isinstance(text, str):
            return ""
        text = cls.URL_PATTERN.sub("[URL]", text)
        text = cls.MENTION_PATTERN.sub("[USER]", text)
        text = cls.HASHTAG_PATTERN.sub(r"\1", text)
        # Replace unicode emojis
        text = cls.EMOJI_PATTERN.sub(" [EMOJI] ", text)
        # Replace named emoji tokens from dataset (e.g., "face_with_tears_of_joy")
        text = cls.NAMED_EMOJI_PATTERN.sub("[EMOJI]", text)
        # Normalise Hinglish elongations
        text = cls.HIND_ELONGATION.sub(r"\1\1\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not preserve_caps:
            text = text.lower()
        return text

    @classmethod
    def batch_clean(cls, texts: List[str], **kwargs) -> List[str]:
        return [cls.clean(t, **kwargs) for t in texts]

    @classmethod
    def extract_sarcasm_cues_hinglish(cls, text: str) -> List[str]:
        """Detect Hinglish sarcasm surface cues."""
        cues = []
        text_low = text.lower()
        for pat in cls.SARCASM_TRIGGERS_HI:
            if re.search(pat, text_low):
                cues.append(re.sub(r"\\b", "", pat).strip())
        if re.search(r"\b[A-Z]{3,}\b", text):
            cues.append("ALL-CAPS word")
        if text.count("!") >= 2:
            cues.append("Multiple exclamation marks")
        if re.search(r"SOOO+|BAHUT|ITNA|REALL?Y+", text):
            cues.append("Exaggerated Hinglish intensifier (CAPS)")
        return cues


# ─────────────────────────────────────────────────────────────────────────────
# 3. SARCASM FEATURE LOADER
# ─────────────────────────────────────────────────────────────────────────────

class SarcasmFeatureLoader:
    """
    Loads and merges human_labelled_features.csv and ai_labelled_features.csv
    into a single sarcasm lookup keyed by cleaned comment text.

    Priority: human label > AI label (when both available)
    """

    def __init__(
        self,
        human_path: str = HUMAN_SAR_PATH,
        ai_path   : str = AI_SAR_PATH,
        use_human : bool = True,
        use_ai    : bool = True,
    ):
        self.preprocessor = TextPreprocessor()
        self._lookup: Dict[str, dict] = {}
        self._load(human_path, ai_path, use_human, use_ai)

    def _load(self, human_path, ai_path, use_human, use_ai):
        frames = []

        if use_human and os.path.exists(human_path):
            df_h = pd.read_csv(human_path)
            df_h = df_h[df_h["human_labelled"].isin(["sarcastic", "non-sarcastic", "non_sarcastic"])]
            df_h["sarcastic"] = df_h["human_labelled"].apply(normalise_sarcasm_label)
            df_h["source"]    = "human"
            df_h["text_key"]  = self.preprocessor.batch_clean(df_h["comment"].tolist())
            frames.append(df_h[["text_key", "sarcastic", "emoji_present", "hindi_pct", "source"]])
            logger.info(f"Loaded {len(df_h)} human sarcasm labels.")

        if use_ai and os.path.exists(ai_path):
            df_a = pd.read_csv(ai_path)
            df_a["sarcastic"] = df_a["ai_labelled"].apply(normalise_sarcasm_label)
            df_a["source"]    = "ai"
            df_a["text_key"]  = self.preprocessor.batch_clean(df_a["comment"].tolist())
            frames.append(df_a[["text_key", "sarcastic", "emoji_present", "hindi_pct", "source"]])
            logger.info(f"Loaded {len(df_a)} AI sarcasm labels.")

        if not frames:
            logger.warning("No sarcasm feature files found. All sarcasm flags default to 0.")
            return

        merged = pd.concat(frames, ignore_index=True)
        # Human label takes priority: sort so human rows come first
        merged = merged.sort_values("source", ascending=True)   # human < ai alphabetically
        # Keep first occurrence per text (= human if available)
        merged = merged.drop_duplicates(subset="text_key", keep="first")

        for _, row in merged.iterrows():
            self._lookup[row["text_key"]] = {
                "sarcastic"    : int(row["sarcastic"]),
                "emoji_present": str(row.get("emoji_present", "no")),
                "hindi_pct"    : float(row.get("hindi_pct", 0.0)),
                "source"       : row["source"],
            }
        logger.info(f"Sarcasm lookup built: {len(self._lookup)} unique entries.")

    def get(self, text: str) -> dict:
        """Look up sarcasm info for a cleaned text string."""
        key = self.preprocessor.clean(text)
        return self._lookup.get(key, {
            "sarcastic"    : 0,
            "emoji_present": "no",
            "hindi_pct"    : 0.0,
            "source"       : "none",
        })

    @property
    def size(self) -> int:
        return len(self._lookup)


# ─────────────────────────────────────────────────────────────────────────────
# 4. MC-HINGLISH1.0 DATASET LOADER
# ─────────────────────────────────────────────────────────────────────────────

class CyberbullyingDataset:
    """
    Multi-class Hinglish Cyberbullying Dataset Loader.

    Sources:
      - MC-Hinglish1.0 train/test (7400 + 1000 rows)
      - Human / AI sarcasm feature files (joined by text)

    Final columns per row:
      text         : cleaned Hinglish tweet
      label        : 0 = benign | 1 = cyberbullying  (binary)
      label_mc     : 0-6  (multi-class category)
      label_str    : category string
      sarcastic    : 0 | 1  (from human/AI sarcasm files)
      emoji_present: yes | no
      hindi_pct    : float (% Hindi words in the tweet)
      platform     : twitter (MC-Hinglish is from Twitter)
    """

    def __init__(
        self,
        train_path     : str = MC_TRAIN_PATH,
        test_path      : str = MC_TEST_PATH,
        use_sample_corpus: bool = False,   # fallback if CSVs missing
        balance        : bool = True,
        val_size       : float = 0.1,
        use_sarcasm_features: bool = True,
    ):
        self.preprocessor = TextPreprocessor()
        self.balance       = balance
        self.val_size      = val_size
        self.raw_train_df  = pd.DataFrame()
        self.raw_test_df   = pd.DataFrame()

        # Load sarcasm lookup
        self.sarcasm_loader = (
            SarcasmFeatureLoader() if use_sarcasm_features else None
        )

        # Load main dataset
        if os.path.exists(train_path) and os.path.exists(test_path):
            logger.info(f"Loading MC-Hinglish1.0 from {train_path} / {test_path}")
            self.raw_train_df = self._load_mc(train_path)
            self.raw_test_df  = self._load_mc(test_path)
        elif use_sample_corpus:
            logger.warning("MC-Hinglish1.0 CSVs not found — falling back to built-in sample.")
            self.raw_train_df = self._fallback_sample()
            self.raw_test_df  = pd.DataFrame()
        else:
            raise FileNotFoundError(
                f"Dataset files not found:\n  {train_path}\n  {test_path}\n"
                "Set use_sample_corpus=True to use the built-in fallback."
            )

        logger.info(
            f"Train: {len(self.raw_train_df)} rows | "
            f"Test:  {len(self.raw_test_df)} rows"
        )
        self._log_label_dist(self.raw_train_df, "Train")

    # ── loaders ──────────────────────────────────────────────────────────────

    def _load_mc(self, path: str) -> pd.DataFrame:
        """Load one MC-Hinglish CSV and enrich with sarcasm features."""
        df = pd.read_csv(path)

        # Rename to standard column names
        df = df.rename(columns={"tweets": "text", "cyberbullying_type": "label_str"})
        df["text"] = self.preprocessor.batch_clean(df["text"].tolist())
        df = df.dropna(subset=["text", "label_str"])
        df = df[df["text"].str.strip() != ""]

        # Binary label
        df["label"] = df["label_str"].map(MULTICLASS_TO_BINARY).fillna(1).astype(int)

        # Multi-class label
        df["label_mc"] = df["label_str"].map(MULTICLASS_LABEL2ID).fillna(0).astype(int)

        # Platform (MC-Hinglish1.0 is from Twitter)
        df["platform"] = "twitter"

        # ── Sarcasm labelling strategy ────────────────────────────────────
        # MC-Hinglish tweets and sarcasm feature comments come from different
        # sources, so exact-text matching yields ~0 hits.
        # Strategy (priority order):
        #   1. label_str == 'mockery'  → sarcastic=1 (mockery IS sarcasm)
        #   2. Hinglish surface-cue heuristic on the cleaned text
        #   3. Text-lookup in sarcasm feature files (catches any overlaps)
        def _assign_sarcasm(row) -> int:
            # Priority 1: mockery label
            if row["label_str"] == "mockery":
                return 1
            # Priority 2: surface cues
            if self.preprocessor.extract_sarcasm_cues_hinglish(row["text"]):
                return 1
            # Priority 3: lookup
            if self.sarcasm_loader:
                info = self.sarcasm_loader.get(row["text"])
                if info["sarcastic"] == 1:
                    return 1
            return 0

        df["sarcastic"]     = df.apply(_assign_sarcasm, axis=1)
        df["emoji_present"] = df["text"].apply(
            lambda t: "yes" if "[EMOJI]" in t else "no"
        )
        df["hindi_pct"]     = 0.0   # MC-Hinglish1.0 doesn't include this column

        return df[["text", "label", "label_mc", "label_str",
                   "sarcastic", "emoji_present", "hindi_pct", "platform"]]

    def _fallback_sample(self) -> pd.DataFrame:
        """Minimal Hinglish sample corpus when CSVs are missing."""
        samples = [
            {"text": "Wah wah! Kya genius hai tu. Tabhi toh fail hua phir se.",
             "label": 1, "label_mc": 1, "label_str": "mockery",
             "sarcastic": 1, "emoji_present": "no", "hindi_pct": 70.0, "platform": "twitter"},
            {"text": "Tu bilkul bekaar hai, teri aukaat kya hai.",
             "label": 1, "label_mc": 2, "label_str": "abusive",
             "sarcastic": 0, "emoji_present": "no", "hindi_pct": 85.0, "platform": "twitter"},
            {"text": "Bhai tera project ekdum mast tha, proud feel kiya.",
             "label": 0, "label_mc": 0, "label_str": "not_cyberbullying",
             "sarcastic": 0, "emoji_present": "no", "hindi_pct": 60.0, "platform": "twitter"},
        ]
        df = pd.DataFrame(samples)
        df["text"] = self.preprocessor.batch_clean(df["text"].tolist())
        return df

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _log_label_dist(df: pd.DataFrame, split: str):
        dist = df["label_str"].value_counts().to_dict()
        binary = df["label"].value_counts().to_dict()
        sar    = int(df["sarcastic"].sum()) if "sarcastic" in df.columns else 0
        logger.info(
            f"[{split}] Multi-class: {dist} | "
            f"Binary: {binary} | Sarcastic: {sar}"
        )

    def _balance_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Upsample minority class for binary balance."""
        counts = df["label"].value_counts()
        maj, mino = counts.idxmax(), counts.idxmin()
        maj_df  = df[df["label"] == maj]
        mino_df = df[df["label"] == mino]
        mino_up = resample(mino_df, replace=True, n_samples=len(maj_df), random_state=SEED)
        balanced = pd.concat([maj_df, mino_up]).sample(frac=1, random_state=SEED).reset_index(drop=True)
        logger.info(f"After balancing: {dict(balanced['label'].value_counts())}")
        return balanced

    # ── public API ────────────────────────────────────────────────────────────

    def get_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Returns (train_df, val_df, test_df).
        Uses the official MC-Hinglish1.0 train/test split.
        Carves val from train.
        """
        train = self.raw_train_df.copy()
        test  = self.raw_test_df.copy() if len(self.raw_test_df) > 0 else pd.DataFrame()

        if self.balance and len(train) > 0:
            train = self._balance_binary(train)

        if len(train) > 0:
            train, val = train_test_split(
                train, test_size=self.val_size,
                random_state=SEED, stratify=train["label"]
            )
        else:
            val = pd.DataFrame()

        logger.info(
            f"Splits — Train: {len(train)} | Val: {len(val)} | Test: {len(test)}"
        )
        return (
            train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True),
        )

    def get_hf_dataset(
        self,
        tokenizer_name: str = CYBERBULLYING_MODEL,
        multiclass    : bool = False,
    ) -> DatasetDict:
        """
        Return a HuggingFace DatasetDict tokenised and ready for Trainer.

        Args:
            tokenizer_name : HF model name (MuRIL recommended)
            multiclass     : If True, use 7-class label; else binary 0/1
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        train_df, val_df, test_df = self.get_splits()
        label_col = "label_mc" if multiclass else "label"

        def to_hf(df: pd.DataFrame) -> Dataset:
            if len(df) == 0:
                return Dataset.from_dict({
                    "text": [], "label": [], "sarcastic": [],
                    "emoji_present": [], "hindi_pct": [], "platform": []
                })
            return Dataset.from_dict({
                "text"         : df["text"].tolist(),
                "label"        : df[label_col].tolist(),
                "label_str"    : df["label_str"].tolist(),
                "sarcastic"    : df["sarcastic"].tolist(),
                "emoji_present": df["emoji_present"].tolist(),
                "hindi_pct"    : df["hindi_pct"].tolist(),
                "platform"     : df["platform"].tolist(),
            })

        def tokenize(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=TOKENIZER_MAX_LEN,
            )

        ds_dict = DatasetDict({
            "train": to_hf(train_df),
            "val"  : to_hf(val_df),
            "test" : to_hf(test_df),
        })
        tokenized = ds_dict.map(tokenize, batched=True)
        tokenized.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"]
        )
        return tokenized

    def get_stats(self) -> Dict:
        """Corpus statistics for the paper's dataset section."""
        all_df   = pd.concat(
            [self.raw_train_df, self.raw_test_df], ignore_index=True
        ) if len(self.raw_test_df) > 0 else self.raw_train_df

        label_dist = all_df["label_str"].value_counts().to_dict()
        sarcasm_in_bully = int(
            all_df[(all_df["label"] == 1) & (all_df["sarcastic"] == 1)].shape[0]
        )
        emoji_in_bully = int(
            all_df[(all_df["label"] == 1) & (all_df["emoji_present"] == "yes")].shape[0]
        )
        return {
            "dataset"               : "MC-Hinglish1.0 + Sarcasm Features",
            "total_samples"         : len(all_df),
            "train_samples"         : len(self.raw_train_df),
            "test_samples"          : len(self.raw_test_df),
            "cyberbullying_count"   : int((all_df["label"] == 1).sum()),
            "non_cyberbullying_count": int((all_df["label"] == 0).sum()),
            "sarcastic_bully_count" : sarcasm_in_bully,
            "emoji_in_bully_count"  : emoji_in_bully,
            "label_distribution"    : {str(k): int(v) for k, v in label_dist.items()},
            "avg_hindi_pct"         : float(all_df["hindi_pct"].mean()),
            "avg_text_length_words" : float(
                all_df["text"].str.split().apply(len).mean()
            ),
        }

    def get_sarcasm_only_df(self, source: str = "human") -> pd.DataFrame:
        """
        Return the standalone sarcasm feature dataframe.
        Useful for training / evaluating the sarcasm module separately.

        Args:
            source: 'human' | 'ai' | 'both'
        """
        frames = []
        if source in ("human", "both") and os.path.exists(HUMAN_SAR_PATH):
            df_h = pd.read_csv(HUMAN_SAR_PATH)
            df_h = df_h[df_h["human_labelled"].isin(["sarcastic", "non-sarcastic", "non_sarcastic"])].copy()
            df_h["sarcastic"] = df_h["human_labelled"].apply(normalise_sarcasm_label)
            df_h["text"]      = self.preprocessor.batch_clean(df_h["comment"].tolist())
            df_h["source"]    = "human"
            frames.append(df_h[["text", "sarcastic", "emoji_present", "hindi_pct", "source"]])

        if source in ("ai", "both") and os.path.exists(AI_SAR_PATH):
            df_a = pd.read_csv(AI_SAR_PATH)
            df_a["sarcastic"] = df_a["ai_labelled"].apply(normalise_sarcasm_label)
            df_a["text"]      = self.preprocessor.batch_clean(df_a["comment"].tolist())
            df_a["source"]    = "ai"
            frames.append(df_a[["text", "sarcastic", "emoji_present", "hindi_pct", "source"]])

        if not frames:
            return pd.DataFrame()

        merged = pd.concat(frames, ignore_index=True)
        merged = merged[merged["sarcastic"] != -1].reset_index(drop=True)
        logger.info(
            f"Sarcasm-only DF ({source}): {len(merged)} rows | "
            f"Sarcastic: {merged['sarcastic'].sum()}"
        )
        return merged


# ─────────────────────────────────────────────────────────────────────────────
# 5. EARLY DETECTION WINDOW BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_early_detection_windows(
    conversations: List[Dict],
    k: int = 3
) -> List[Dict]:
    """
    For conversation-level data, extract the first k messages.
    Uses [SEP] as turn boundary (recognised by MuRIL / XLM-RoBERTa).
    """
    preprocessor = TextPreprocessor()
    windows = []
    for conv in conversations:
        messages = conv.get("messages", [])[:k]
        cleaned  = preprocessor.batch_clean(messages)
        text     = " [SEP] ".join(cleaned)
        windows.append({
            "text"            : text,
            "label"           : conv["label"],
            "conversation_id" : conv.get("conversation_id", ""),
            "n_messages_used" : len(messages),
        })
    return windows


# ─────────────────────────────────────────────────────────────────────────────
# 6. QUICK SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    # Force UTF-8 output on Windows to avoid cp1252 emoji errors
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logger.info("=" * 65)
    logger.info("  MC-Hinglish1.0 Dataset Loader -- Smoke Test")
    logger.info("=" * 65)

    ds = CyberbullyingDataset(balance=True)
    stats = ds.get_stats()
    print("\n[STATS] Dataset Statistics:")
    print(json.dumps(stats, indent=2, ensure_ascii=True))

    train, val, test = ds.get_splits()
    print(f"\n[SAMPLE] Train rows: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(train[["text", "label_str", "sarcastic", "hindi_pct"]].head(5).to_string())

    print("\n[SARCASM] Sarcasm distribution in train:")
    print(train.groupby(["label_str", "sarcastic"]).size().to_string())

    print("\n[SARCASM-FILE] Standalone sarcasm dataframe (human labels):")
    sar_df = ds.get_sarcasm_only_df(source="human")
    print(f"   Rows: {len(sar_df)} | Sarcastic: {int(sar_df['sarcastic'].sum())}")
    print(sar_df[["text", "sarcastic", "emoji_present"]].head(3).to_string())

    print("\n[OK] Dataset module loaded successfully.")
