**Model size, inference latency and training time (Apple M1 Pro).**

| name | params_m | trainable_params_m | size_mb | latency_ms_bs1_cpu | latency_ms_bs1_mps | throughput_per_s_bs32_mps | train_minutes_mean |
|---|---|---|---|---|---|---|---|
| MuRIL | 237.562 | 86.047 | 453.113 | 23.749 | 11.307 | 272.041 | 13.761 |
| XLM-R base | 278.049 | 86.047 | 530.336 | 30.242 | 10.828 | 333.086 | 13.327 |
| HingRoBERTa | 278.049 | 86.047 | 530.336 | 29.936 | 10.479 | 346.571 | 13.193 |
| DistilRoBERTa | 82.124 | 43.520 | 156.639 | 15.972 | 6.095 | 512.006 | – |
| TF-IDF + LR | 0.784 | – | 9.973 | 0.571 | – | – | 0.458 |
| TF-IDF + SVM | 0.784 | – | 33.913 | 2.798 | – | – | 0.811 |
