"""Raw-file loaders -> list[Record]. One function per verified source."""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

from .schema import CONDA_TOXIC, MC_LABELS, MC_TO_BINARY, Record
from .text import clean_text, script_of

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    pass


def _read_csv(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_mc_hinglish(raw_dir: Path) -> list[Record]:
    out: list[Record] = []
    for split in ("train", "test"):
        rows = _read_csv(raw_dir / f"{split}.csv")
        for i, r in enumerate(rows):
            lab = (r.get("cyberbullying_type") or "").strip()
            if lab not in MC_LABELS:
                raise DataValidationError(f"MC-Hinglish {split} row {i}: unknown label {lab!r}")
            raw = r.get("tweets") or ""
            out.append(Record(
                id=f"mc-{split}-{i:05d}", text=clean_text(raw), text_raw=raw, source="mc_hinglish",
                language="hi-en", script=script_of(raw), label_orig=lab, orig_split=split,
                cb_type=lab, cb_binary=MC_TO_BINARY[lab],
            ))
    return out


def _blank_line_blocks(path: Path) -> list[list[str]]:
    blocks, cur = [], []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip() == "":
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(line.rstrip("\n"))
    if cur:
        blocks.append(cur)
    return blocks


HASHTAG_RE = re.compile(r"#\w*")   # hashtags and stray "#" characters
TWEET_ID_RE = re.compile(r"^\d{5,20}$")   # ids range from 10-11 digits (2009-11) to 18 (2017)


def _swami_tweets(path: Path) -> dict[str, str]:
    """Sarcasm_tweets.txt: an id line (first line or after a blank line) followed by
    one or more text lines. Some tweets span several lines."""
    out: dict[str, list[str]] = {}
    cur, prev_blank = None, True
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if prev_blank and TWEET_ID_RE.match(s):
            cur = s
            out[cur] = []
        elif s and cur is not None:
            out[cur].append(s)
        prev_blank = (s == "")
    return {k: " ".join(v).strip() for k, v in out.items()}


def _swami_truth(path: Path) -> dict[str, str]:
    """Sarcasm_tweet_truth.txt: alternating id / YES|NO lines. (The repository README
    says blank-line separated, but the file has no blank lines.)"""
    lines = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(lines) % 2:
        raise DataValidationError("Swami truth file has an odd number of non-empty lines")
    out = {}
    for tid, lab in zip(lines[0::2], lines[1::2]):
        if not TWEET_ID_RE.match(tid) or lab.upper() not in ("YES", "NO"):
            raise DataValidationError(f"Swami truth: bad pair {tid!r}/{lab!r}")
        out[tid] = lab.upper()
    return out


def load_swami_sarcasm(raw_dir: Path) -> list[Record]:
    """Parse the id/text and id/label files and join on tweet id."""
    texts = _swami_tweets(raw_dir / "Sarcasm_tweets.txt")
    labels = _swami_truth(raw_dir / "Sarcasm_tweet_truth.txt")
    out = []
    missing = 0
    for tid, lab in labels.items():
        if tid not in texts or lab not in ("YES", "NO"):
            missing += 1
            continue
        raw = texts[tid]
        # Collection artefact: sarcastic tweets were retrieved via #sarcasm/#irony and
        # non-sarcastic ones via topic hashtags (#tripletalaq, #bollywood, #cricket ...);
        # 486/504 sarcastic tweets contain #irony or #sarcasm. All hashtag tokens are
        # removed from the model input so the label cannot be read off the query term.
        out.append(Record(
            id=f"swami-{tid}", text=clean_text(HASHTAG_RE.sub(" ", raw)), text_raw=raw, source="swami_sarcasm",
            language="hi-en", script=script_of(raw), label_orig=lab, sarcasm=int(lab == "YES"),
            meta={"tweet_id": tid},
        ))
    if missing:
        logger.warning("Swami: %d label entries without text or with invalid label", missing)
    return out


def swami_language_tags(raw_dir: Path) -> dict[str, list[tuple[str, str]]]:
    """tweet_id -> [(token, lang_tag)] from Sarcasm_tweets_with_language.txt."""
    out = {}
    for blk in _blank_line_blocks(raw_dir / "Sarcasm_tweets_with_language.txt"):
        tid, toks = blk[0].strip(), []
        for line in blk[1:]:
            parts = line.rsplit(None, 1)
            if len(parts) == 2:
                toks.append((parts[0], parts[1].lower()))
        out[tid] = toks
    return out


def load_user_sarcasm_human(path: Path) -> list[Record]:
    norm = {"sarcastic": 1, "non-sarcastic": 0, "non_sarcastic": 0}
    out = []
    for i, r in enumerate(_read_csv(path)):
        lab = (r.get("human_labelled") or "").strip().lower()
        if lab not in norm:          # 3 rows are labelled 'error'
            continue
        raw = r.get("comment") or ""
        out.append(Record(
            id=f"usr-{i:05d}", text=clean_text(raw), text_raw=raw, source="user_sarcasm_human",
            language="hi-en", script=script_of(raw), label_orig=r.get("human_labelled"),
            sarcasm=norm[lab],
            meta={"ai_label": r.get("ai_labelled"), "emoji_present": r.get("emoji_present"),
                  "hindi_pct": r.get("hindi_pct")},
        ))
    return out


def load_conda(raw_dir: Path) -> list[Record]:
    out = []
    for fname, split in (("CONDA_train.csv", "train"), ("CONDA_valid.csv", "valid")):
        for r in _read_csv(raw_dir / fname):
            intent = (r.get("intentClass") or "").strip()
            if intent not in {"E", "I", "A", "O"}:
                continue
            raw = r.get("utterance") or ""
            out.append(Record(
                id=f"conda-{r['Id']}", text=clean_text(raw.replace("[SEPA]", " | "), demojize=False),
                text_raw=raw, source="conda", language="en", script=script_of(raw),
                label_orig=intent, orig_split=split, toxic_intent=intent,
                conversation_id=f"{r['matchId']}-{r['conversationId']}",
                timestamp=float(r["chatTime"]) if r.get("chatTime") not in (None, "") else None,
                meta={"match_id": r["matchId"], "row_id": int(r["Id"]), "toxic": int(intent in CONDA_TOXIC)},
            ))
    return out


def load_tweeteval_offensive(raw_dir: Path) -> list[Record]:
    import pandas as pd

    df = pd.read_parquet(raw_dir / "test.parquet")
    out = []
    for i, r in df.iterrows():
        raw = str(r["text"])
        out.append(Record(
            id=f"olid-test-{i:04d}", text=clean_text(raw), text_raw=raw, source="tweeteval_offensive",
            language="en", script=script_of(raw), label_orig=str(int(r["label"])), orig_split="test",
            offensive=int(r["label"]),
        ))
    return out
