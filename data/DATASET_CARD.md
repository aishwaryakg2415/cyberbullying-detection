# Dataset Card — data used by SACENet

All numbers below were **measured** by `scripts/verify_datasets.py` / `scripts/prepare_data.py`
on the files whose SHA-256 hashes are listed in `dataset_manifest.json` (retrieved 2026-10-17).

---

## 1. MC-Hinglish1.0 (primary)

* **Citation**: Rahman-Laskar, Gupta, Badhani & Pinto-Avendaño (2024), *Computación y Sistemas* 28(3), doi:10.13053/CyS-28-3-4989.
* **Source**: Hugging Face `sahinurlaskar/MC-Hinglish1.0` (revision `a8ae581`), CC-BY-NC-SA-4.0, ungated.
  The project owner's local copies are byte-identical to this release.
* **Content**: Roman-script Hinglish Twitter comments, one label from 7 cyberbullying *types*.
  The paper describes manual annotation over ~60 days but reports neither the number of annotators nor
  inter-annotator agreement.
* **Columns**: `tweets`, `cyberbullying_type`.

| Label | Train (official) | Test (official) | Total |
|---|---:|---:|---:|
| not_cyberbullying | 1,491 | 200 | 1,691 |
| abusive | 989 | 148 | 1,137 |
| gender | 1,004 | 141 | 1,145 |
| religion | 979 | 131 | 1,110 |
| mockery | 980 | 127 | 1,107 |
| age | 977 | 129 | 1,106 |
| offensive | 980 | 124 | 1,104 |
| **Total** | **7,400** | **1,000** | **8,400** |

The files differ slightly from the per-class totals in the paper's table and the HF card; this card uses
the counts measured in the files.

**Our splits** (`data/splits/mc7/`): train 6,659 · val 741 (stratified 10% of official train,
near-duplicate-grouped) · test 1,000 (official, unchanged).

**Derived binary task (`mc2`)**: `not_cyberbullying` → 0, the six other types → 1. Every non-negative
label is a *type of cyberbullying* by the dataset's own definition, so the union is well-defined.
Note, however, that `offensive` and `abusive` are defined by *language*, while `age`/`gender`/`religion`
are defined by *target*, and `mockery` by *pragmatic style*. These are overlapping, non-exclusive concepts
forced into a single label.

**Quality findings**
* Missing values: 0. Script: 100% Roman (8,400/8,400). Length: median 13 words (mean 17.8, p95 34, max 1,402 — one outlier).
* Official train: 8 exact-duplicate groups, 250 normalised-duplicate groups, 1,082 near-duplicate pairs; **0 label conflicts** among exact duplicates.
* **Official test overlaps training data**: 4 exact, 38 normalised and 174 near-duplicate pairs with the official train
  split; 182 of the 1,000 test items (18.2%) have a near-duplicate in the official train data. These items are flagged and
  every MC result is also reported on the 818-item de-duplicated subset.
* After our split: train↔val exact / normalised / near-duplicate overlap = 0 / 0 / 0.
* No sarcasm annotation exists. **The original repository's `sarcastic` column (mockery + regex) is not used.**

## 2. Swami et al. (2018) English-Hindi code-mixed sarcasm corpus

* **Citation**: Swami, Khandelwal, Singh, Akhtar & Shrivastava (2018), arXiv:1805.11869.
* **Source**: GitHub `sahilswami96/SarcasmDetection_CodeMixed` (commit `7c259dc`), GPL-3.0 repository licence.
* **Content**: 5,250 tweets with text, label YES/NO for sarcasm (**504 sarcastic, 9.6%**), plus token-level
  language tags (en 11,195 · hi 84,239 · rest 15,921 tokens). Mean Code-Mixing Index (CMI computed from these tags as
  100·(1 − max-language share)) = **11.8** over 5,249 tweets.
* **Layout note**: the README says the truth file is blank-line separated; it is actually alternating id/label lines.
  Tweet IDs vary between 10 and 18 digits. The loader handles both cases and reproduces the paper's 5,250 / 504.
* **Collection artefact (important)**: sarcastic tweets were retrieved with `#sarcasm` / `#irony`, and 486 of the 504
  sarcastic tweets contain one of them. Non-sarcastic tweets carry topic query hashtags (`#tripletalaq` 280,
  `#bollywood` 271, `#cricket` 118, …). With hashtags kept, a TF-IDF LR baseline reaches 0.879 macro-F1. After
  removing them it reaches 0.739, so a large part of the "easy" signal is the retrieval query. **All `#` tokens are
  removed from the model input** (raw text is preserved in `text_raw`).
* Our splits (stratified, near-duplicate-grouped): train 3,675 (353 sarcastic) · val 525 (50) · test 1,050 (101).
  Cross-split near-duplicates: 0.

## 3. CONDA (Weld et al., 2021) — proxy for early detection

* **Citation**: Weld et al. (2021), Findings of ACL-IJCNLP 2021, doi:10.18653/v1/2021.findings-acl.213.
* **Source**: GitHub `usydnlp/CONDA` (commit `3fb9067`). **No licence file** — used for non-commercial research
  evaluation; not redistributed.
* **Content**: English Dota 2 in-game chat. Each row is an utterance with `matchId`, `conversationId`, `chatTime` and an
  utterance-level intent: **E**xplicit toxicity, **I**mplicit toxicity, **A**ction, **O**ther. We define
  toxic = E ∪ I. The official test file is unannotated and is not used.
* Annotated utterances: 35,895 (E 4,711 · I 2,274 · A 2,299 · O 26,611) in 10,902 conversations.
* **Official split is utterance-level**: 3,540 conversations appear in both the official train and valid files, and
  1,772 of 1,778 valid-file matches also occur in train. We therefore pool and re-split **by match id**:

| Split | Conversations | Utterances | Toxic conversations | Mean / median length |
|---|---:|---:|---:|---|
| train | 7,726 | 25,341 | 2,726 | 3.28 / 2 |
| val | 1,089 | 3,495 | 396 | 3.21 / 2 |
| test | 2,087 | 7,059 | 794 | 3.38 / 2 |

* **Limitations**: conversations are short (median 2 turns), and turns from the unannotated official test file are
  *missing*, which leaves gaps inside conversations. The domain (game chat, English, toxicity) differs from Hinglish
  cyberbullying. CONDA can therefore test the *mechanics* of the causal early-warning procedure, but **not** whether
  SACENet detects Hinglish cyberbullying early.

## 4. TweetEval-Offensive test (OLID / OffensEval-2019)

* **Citation**: Barbieri et al. (2020); data from Zampieri et al. (2019).
* **Source**: HF `cardiffnlp/tweet_eval` (revision `b3a375b`), config `offensive`, test split: 860 tweets
  (240 offensive / 620 not).
* **Use**: English, **offensive-language** labels — a different construct from cyberbullying. Used only as a
  cross-lingual, cross-construct stress test of the binary MC model; not used for training or model selection.

## 5. User-supplied sarcasm files (provenance unknown)

* `human_labelled_features.csv`: 8,885 rows (8,882 with a valid human label: 3,737 sarcastic / 5,145 not;
  3 rows labelled `error`). It also contains an `ai_labelled` column and `emoji_present` / `hindi_pct` features.
  382 exact-duplicate rows; **24 duplicate groups carry conflicting human labels**. After normalised
  de-duplication and dropping conflicts, 8,455 items remain (3,644 / 4,811). 318 items contain Devanagari, and some
  are Marathi rather than Hindi.
* Human-vs-AI label agreement in this file: 4,701/8,882 = **52.9%**; the AI labeller marks 86% of rows as sarcastic.
* `ai_labelled_features.csv`: 12,809 rows of LLM labels only.
* 0 texts overlap MC-Hinglish, so the original repository's text-lookup "sarcasm feature" never fired.
* **Use**: the human labels serve only as a supplementary cross-corpus sarcasm test. The AI file is not used.
  **Please document the origin, licence and annotation protocol before any publication use.**

## Ethical notes

The corpora contain slurs, sexual harassment, religious and gendered abuse, and identity attacks. Example texts quoted
in the paper are limited to short, non-identifying fragments; @-mentions and URLs are masked by the pipeline.
