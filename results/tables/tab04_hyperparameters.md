**Model configurations and hyperparameters (shared across models).**

| Setting | Value |
|---|---|
| max_len | 128 |
| batch_size | 32 |
| freeze_word_embeddings | True |
| lr | 3e-05 |
| head_lr | 0.001 |
| epochs | 5 |
| patience | 2 |
| warmup | 0.1 |
| weight_decay | 0.01 |
| clip | 1.0 |
| dropout | 0.1 |
| class_weighting | True |
| amp | False |
| optimiser | AdamW (fused), linear warm-up/decay |
| seeds | 13, 42, 87 |
| MTL lambda | 0.5 |
| model selection | best validation macro-F1 epoch |
| TF-IDF features | word 1-2-grams + char_wb 2-5-grams, sublinear TF |
| TF-IDF C grid (val-selected) | 0.1, 0.3, 1, 3, 10 |
| encoder: muril | google/muril-base-cased @ afd9f36c |
| encoder: xlmr | FacebookAI/xlm-roberta-base @ e73636d4 |
| encoder: hingroberta | l3cube-pune/hing-roberta @ f714c4bd |
| encoder: distilroberta | distilbert/distilroberta-base @ fb53ab88 |
