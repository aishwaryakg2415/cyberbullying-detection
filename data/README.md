# Data

Raw and processed text is **not committed** (licences: CC-BY-NC-SA-4.0, GPL-3.0, unspecified).
Everything is re-created from official sources by two commands; nothing requires Kaggle,
credentials or API keys.

```bash
python scripts/download_datasets.py --user-dir ..   # --user-dir only for the two user-supplied CSVs
python scripts/verify_datasets.py                    # SHA-256 + record counts vs dataset_manifest.json
python scripts/prepare_data.py                       # clean, de-duplicate, split, statistics
```

## Layout

| Path | Tracked in git | Content |
|---|---|---|
| `data/dataset_manifest.json` | ✅ | sources, pinned revisions, licences, SHA-256 per file, measured record/label counts |
| `data/DATASET_CARD.md` | ✅ | definitions, distributions, quality findings, intended use |
| `data/splits/<task>/<split>_ids.json` | ✅ | exact record IDs of every split (IDs only, no text) |
| `data/raw/<source>/` | ❌ | files as downloaded (`download_log.json` records URL + hash + time) |
| `data/processed/<task>/{train,val,test}.jsonl` | ❌ | unified-schema records (`src/data/schema.py`) |
| `data/processed/data_statistics.json` | ❌ (regenerated) | duplicate / leakage / distribution statistics |

## Sources used

| Key | Dataset | Official source (pinned) | Licence | Role |
|---|---|---|---|---|
| `mc_hinglish` | MC-Hinglish1.0 (Rahman-Laskar et al., 2024) | HF `sahinurlaskar/MC-Hinglish1.0` @ `a8ae581` | CC-BY-NC-SA-4.0 | primary task (7-way type; derived binary) |
| `swami_sarcasm` | Hindi-English code-mixed sarcasm corpus (Swami et al., 2018) | GitHub `sahilswami96/SarcasmDetection_CodeMixed` @ `7c259dc` | GPL-3.0 | human sarcasm labels for the sarcasm component |
| `conda` | CONDA (Weld et al., 2021) | GitHub `usydnlp/CONDA` @ `3fb9067` | none stated | **proxy** sequential data for early detection (English game chat) |
| `tweeteval_offensive` | TweetEval-Offensive test = OffensEval-2019/OLID test | HF `cardiffnlp/tweet_eval` @ `b3a375b` | see OLID terms | English external stress test (different label definition) |
| `user_sarcasm_human` | user-supplied `human_labelled_features.csv` | local file, **provenance unknown** | unknown | supplementary cross-corpus sarcasm test only |
| `user_sarcasm_ai` | user-supplied `ai_labelled_features.csv` | local file, provenance unknown | unknown | **not used** (labels are LLM output, not ground truth) |

## Candidates that need manual steps (not used in the reported results)

| Dataset | Why not downloaded | What you need to do |
|---|---|---|
| **HASOC** 2019–2021 (Mandl et al., 2019) | Distributed to registered participants by the FIRE organisers. Unofficial Hugging Face re-uploads exist (`pranjali94/HASOC2020_Roman`, `nikitadesai/hasoc`) but carry no licence or provenance, so they were not used. | Request access through the track pages at <https://hasocfire.github.io/>; place the files under `data/raw/hasoc/` and add a loader in `src/data/loaders.py`. |
| **TRAC-1 / TRAC-2** aggression (Kumar et al., 2018; 2020) | Released by the organisers on request. | Follow the data-request instructions linked from the shared-task overview papers (ACL Anthology W18-4401, 2020.trac-1.1). |
| **Bohra et al. (2018)** Hinglish hate speech | Repository ships tweet **IDs + labels only** (`id_annotated.tsv`); text requires X/Twitter API hydration. | Hydrate with your own API access (subject to X's terms); many tweets will no longer exist. |
| **OLID official zip** | Hosted on a Google Sites page; TweetEval redistributes the identical OffensEval-2019 test set, which is used instead. | — |
| Session-level cyberbullying corpora (e.g. Hosseinmardi et al., 2015, Instagram) | Available from the authors on request only. | Contact the authors. |

## Processing summary (see `src/data/prepare.py`)

* Text cleaning keeps case, punctuation, Devanagari and code-switching. Emoji → CLDR names, URL/@mention
  placeholders, elongation capped at 3 repeats. Swami input additionally has **all hashtags removed**:
  they were the corpus' retrieval queries and leak the label (see DATASET_CARD).
* Duplicates: exact, normalised (case/punctuation/repeat-insensitive) and near-duplicate
  (char 3–5-gram TF-IDF cosine ≥ 0.9).
* MC-Hinglish: the official test split is kept intact. Validation (10%) is carved from the official train split with
  `StratifiedGroupKFold`, using near-duplicate clusters as groups, so no near-duplicates cross train/val.
  Test items with a near-duplicate in the official training data are flagged (`meta.near_dup_in_train`) and
  results are additionally reported on the de-duplicated test subset.
* Swami: no official split → stratified, near-duplicate-grouped 70/10/20.
* CONDA: the official train/valid files were split per utterance (conversations straddle files), so all
  annotated utterances are pooled and re-split **by match id** 70/10/20.
* No oversampling anywhere. Class weights are computed from the training split only, inside the trainer.
* Split seed: `20261017`.
