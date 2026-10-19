# Model Card — SACENet MC7 classifier (MuRIL, seed 13)

*Rendered from saved results by `scripts/build_paper.py` on 2026-10-18. Numbers trace to `results/analysis/analysis.json`.*

## Model details
- **Architecture:** `google/muril-base-cased` (revision `afd9f36c`) encoder, [CLS] pooling, dropout 0.1, linear head with 7 outputs.
  The token-embedding table is frozen; all transformer layers are fine-tuned.
- **Training:** MC-Hinglish1.0 train split (6,659 comments), class-weighted cross-entropy, AdamW (encoder lr 3e-5, head lr 1e-3),
  batch 32, ≤ 5 epochs with early stopping on validation macro-F1. Seed 13. Trained on an Apple M1 Pro (MPS).
- **Calibration:** temperature scaling fitted on the validation split, applied in the demo app. On the shifted official
  test split it did **not** improve MuRIL's calibration (ECE 0.094 → 0.103), so
  treat displayed probabilities as approximate.
- **Checkpoint:** `results/runs/mc7__muril__base__s13/checkpoint/model/` (fp16 weights; not in git).
- **Companion model:** `sarcasm__muril__base__s13` (Swami et al. 2018 sarcasm). In the app it is shown for information only.

## Intended use
Research on Hinglish cyberbullying detection, and **decision support for trained human moderators**. Not for automated
sanctions, profiling of individuals, or any decision about a person without human review.

## Labels
`not_cyberbullying`, `age`, `gender`, `religion`, `mockery`, `abusive`, `offensive` (MC-Hinglish1.0 definitions).
P(any cyberbullying) = 1 − P(`not_cyberbullying`).

## Performance (official MC-Hinglish1.0 test split, n = 1,000)
| Model | Test macro-F1 (mean ± s.d., 3 seeds) | Validation macro-F1 |
|---|---|---|
| MuRIL (this card's family) | 0.632 ± 0.016 | 0.761 |
| HingRoBERTa | 0.721 ± 0.040 | 0.793 |
| TF-IDF + SVM (baseline) | 0.743 | 0.763 |

Per-class F1 for MuRIL (seed ensemble): age 0.888, religion
0.860, not_cyberbullying 0.807,
gender 0.642, abusive 0.616,
offensive 0.396, mockery 0.288.

**This model is not the best model on the official test split.** TF-IDF + SVM and HingRoBERTa score higher. MuRIL is the
retained checkpoint because the brief specified MuRIL as the primary encoder and because the explainability study and
demo were built on it. Choosing the deployed model is the owner's decision (see `paper/Author_Review_Notes.md`).

## Known limitations
- The official test split is distribution-shifted from the training data (domain-classifier AUC 0.884).
  Expect lower performance on new data than on validation.
- `mockery` and `offensive` are unreliable. Very long comments (> 20 words) are classified poorly.
- Roman-script Hinglish from Twitter only; other scripts, platforms and languages are out of domain
  (English OLID transfer: macro-F1 0.275 for the binary variant).
- Sarcasm supervision did not improve this model (see paper §11).

## Ethical considerations
Can encode annotation and dialect biases, especially around religion and gender. Explanations (SHAP) show model sensitivity,
not intent. The demo does not store user text.

## Data licences
MC-Hinglish1.0: CC-BY-NC-SA-4.0 (non-commercial; ShareAlike). Model weights derived from it should be treated accordingly.
