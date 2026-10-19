**Error analysis by heuristic slice (MuRIL seed ensemble, MC7 test).**

| Slice | N | Accuracy | Macro-F1 |
|---|---|---|---|
| test without near-duplicate in train | 818 | 0.678 | 0.638 |
| test WITH near-duplicate in train | 182 | 0.632 | 0.629 |
| short (<=8 words) | 503 | 0.746 | 0.659 |
| medium (9-20 words) | 471 | 0.611 | 0.607 |
| long (>20 words) | 26 | 0.269 | 0.262 |
| low English function-word ratio (<5%) | 727 | 0.688 | 0.609 |
| high English function-word ratio (>=15%) | 154 | 0.656 | 0.660 |
| no OOV words vs train vocabulary | 556 | 0.669 | 0.653 |
| OOV rate >= 30% (spelling/transliteration variation) | 34 | 0.647 | 0.617 |
| contains profanity | 260 | 0.604 | 0.581 |
| no profanity | 740 | 0.693 | 0.566 |
| sarcasm model P>=0.5 (model-inferred, not gold) | 410 | 0.612 | 0.546 |
| sarcasm model P<0.5 | 590 | 0.710 | 0.681 |
