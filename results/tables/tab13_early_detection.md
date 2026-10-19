**Early-warning proxy evaluation on CONDA test conversations (running-max policy; thresholds chosen on validation).**

| Turn scorer | F1 | Precision | Recall | False-alarm rate | Mean delay | ERDE_3 | ERDE_5 | Recall @FAR<=0.10(val) | FAR @FAR<=0.10(val) |
|---|---|---|---|---|---|---|---|---|---|
| conda__distilroberta__base__s13 | 0.913 | 0.916 | 0.909 | 0.051 | 0.010 | 0.161 | 0.097 | 0.909 | 0.051 |
| conda__distilroberta__base__s42 | 0.910 | 0.930 | 0.890 | 0.041 | 0.071 | 0.165 | 0.101 | 0.890 | 0.041 |
| conda__distilroberta__base__s87 | 0.914 | 0.918 | 0.911 | 0.050 | 0.008 | 0.161 | 0.096 | 0.911 | 0.050 |
| conda__tfidf_lr | 0.912 | 0.928 | 0.897 | 0.043 | 0.022 | 0.165 | 0.098 | 0.897 | 0.043 |
| conda_ctx__distilroberta__base__s13 | 0.906 | 0.948 | 0.868 | 0.029 | 0.148 | 0.171 | 0.109 | 0.868 | 0.029 |
| conda_ctx__distilroberta__base__s42 | 0.910 | 0.926 | 0.894 | 0.044 | 0.070 | 0.165 | 0.101 | 0.894 | 0.044 |
| conda_ctx__distilroberta__base__s87 | 0.910 | 0.927 | 0.894 | 0.043 | 0.058 | 0.165 | 0.101 | 0.894 | 0.043 |
| conda_ctx__tfidf_lr | 0.853 | 0.824 | 0.884 | 0.116 | -0.044 | 0.179 | 0.118 | 0.849 | 0.083 |
