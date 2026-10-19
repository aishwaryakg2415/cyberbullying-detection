# Literature review (verified sources only)

Every work below is in `paper/references.bib`. Bibliographic data was retrieved on 2026-10-17 from the
ACL Anthology, doi.org/Crossref, the arXiv API, or the official PMLR/JMLR/NeurIPS pages. Descriptions are
limited to what the cited paper states (title/abstract/dataset card). Where we characterise a paper more
specifically, the project owner should confirm it against the PDF before submission (see
`paper/Author_Review_Notes.md`).

## 1. Cyberbullying and abusive-language detection

| Work | What it contributes | Relevance to SACENet |
|---|---|---|
| Salawu et al. (2020), *IEEE TAC* — "Approaches to Automated Detection of Cyberbullying: A Survey" | Survey of cyberbullying detection approaches | Framing; definitions |
| Van Hee et al. (2018), *PLOS ONE* — "Automatic detection of cyberbullying in social media text" | Fine-grained cyberbullying detection | Fine-grained categories, as in MC-Hinglish |
| Fortuna & Nunes (2018), *ACM CSUR* — hate-speech detection survey | Survey | Distinguishes hate speech from related constructs |
| Vidgen & Derczynski (2020), *PLOS ONE* — abusive-language training data review | Systematic review of datasets | Why label definitions differ; why datasets should not be naively merged |
| Sap et al. (2019), *ACL* — racial bias in hate-speech detection | Annotation/dialect bias | Ethics section |
| Zampieri et al. (2019), *NAACL* — OLID | Offensive-language taxonomy & dataset | External English test (via TweetEval) |
| Barbieri et al. (2020), *Findings of EMNLP* — TweetEval | Unified tweet benchmark incl. offensive | Source of the OLID test copy we use |
| Mandl et al. (2019), *FIRE* — HASOC 2019 overview | Hate/offensive content in Indo-European languages | Candidate dataset (registration required) |
| Kumar et al. (2018), *TRAC-1* — "Benchmarking Aggression Identification in Social Media" | Hindi-English aggression shared task | Candidate dataset (on request) |
| Kumar et al. (2020), *TRAC-2* — "Evaluating Aggression Identification in Social Media" | Second aggression shared task | Candidate dataset (on request) |
| Mathew et al. (2021), *AAAI* — HateXplain | Hate speech with rationales | Explainability benchmark precedent |

## 2. Hinglish / code-mixed NLP

| Work | Contribution | Relevance |
|---|---|---|
| Rahman-Laskar et al. (2024), *Computación y Sistemas* 28(3) — MC-Hinglish1.0 | 8,400 Hinglish tweets, 7 cyberbullying types; classical ML, voting ensemble reported at 0.609 accuracy | **Primary dataset** |
| Bohra et al. (2018), *PEOPLES workshop* — Hindi-English code-mixed hate speech dataset | Hinglish hate-speech tweets (IDs + labels released) | Unusable without tweet hydration |
| Khanuja et al. (2020), *ACL* — GLUECoS | Code-switching evaluation benchmark | Motivation: code-switching is hard for pretrained models |
| Khanuja et al. (2021), arXiv:2103.10730 — MuRIL | Multilingual model for 17 Indian languages, including transliterated data | **Main encoder** |
| Nayak & Joshi (2022), *WILDRE @ LREC* — L3Cube-HingCorpus and HingBERT | Hinglish corpus and Hinglish-pretrained BERT/RoBERTa models | HingRoBERTa comparison |
| Kumar (2025), arXiv:2506.16066 — MuRIL + XAI for Hinglish cyberbullying | MuRIL on six Hinglish datasets with attribution analysis (not MC-Hinglish) | Closest prior combination of MuRIL + XAI; reason we do **not** claim that combination as novel |

## 3. Sarcasm

| Work | Contribution | Relevance |
|---|---|---|
| Joshi et al. (2017), *ACM CSUR* — sarcasm detection survey | Survey | Background |
| Swami et al. (2018), arXiv:1805.11869 — English-Hindi code-mixed sarcasm corpus | 5,250 tweets, 504 sarcastic, token language tags | **Sarcasm supervision**; we identify a hashtag-retrieval artefact in it |
| Aggarwal et al. (2020), *W-NUT* — sarcasm in Hindi-English code-mixed data with bilingual embeddings | Releases its own Hinglish sarcasm dataset plus a tweet corpus for custom embeddings; reports 78.49% accuracy with attention-based BiLSTMs | Alternative Hinglish sarcasm resource (not used here) |
| Majumder et al. (2019), *IEEE Intelligent Systems* — sentiment and sarcasm with multitask learning | Multitask sarcasm/sentiment | Precedent for sarcasm as an auxiliary task |
| Majumder & Sen (2026), arXiv:2602.21933 — LLMs vs. fine-tuned DistilBERT for Hinglish sarcasm | Small fine-tuned models competitive with LLMs | Supports fine-tuning a local sarcasm model instead of relying on LLM labels |

## 4. Multilingual transformers and multitask learning

Devlin et al. (2019, NAACL) BERT; Conneau et al. (2020, ACL) XLM-R; Wolf et al. (2020, EMNLP demos)
Transformers library; Caruana (1997, *Machine Learning*) multitask learning; Ruder (2017, arXiv) overview of
multitask learning in deep networks; Loshchilov & Hutter (2017, arXiv) decoupled weight decay (AdamW).

## 5. Explainability and faithfulness

Ribeiro et al. (2016, KDD) LIME; Lundberg & Lee (2017, NeurIPS) SHAP; Sundararajan et al. (2017, ICML)
Integrated Gradients; DeYoung et al. (2020, ACL) ERASER (comprehensiveness/sufficiency); Jacovi & Goldberg
(2020, ACL) on defining and evaluating faithfulness. Our deletion test follows the ERASER comprehensiveness idea.
The "explanations ≠ intent" caveat follows Jacovi & Goldberg.

## 6. Early detection and conversation-level data

Losada & Crestani (2016, CLEF) introduced the eRisk test collection and the ERDE latency-weighted error;
López-Vizcaíno et al. (2021, *FGCS*) study early detection of cyberbullying on social networks;
Hosseinmardi et al. (2015, arXiv) annotate Instagram media sessions (available on request); Weld et al. (2021,
Findings of ACL) CONDA, an in-game chat dataset with conversation structure and utterance-level toxicity intents,
which we use as a proxy.

## 7. Evaluation methodology

Dietterich (1998, *Neural Computation*) on statistical tests for comparing classifiers (McNemar); Koehn (2004,
EMNLP) on paired bootstrap resampling in NLP; Guo et al. (2017, ICML) on calibration and temperature scaling;
Pedregosa et al. (2011, JMLR) scikit-learn.

## 8. Identified gap (stated conservatively)

1. MC-Hinglish1.0 results so far are reported for classical models on the official split. We are not aware of a
   published *leakage-audited* transformer evaluation on it. **[Owner: re-check Google Scholar for papers citing
   doi:10.13053/CyS-28-3-4989 before submission.]**
2. Sarcasm is often *assumed* to help detect indirect bullying. We found no controlled test, on Hinglish, of whether
   a separately annotated sarcasm corpus transfers to cyberbullying classification, i.e. one that separates the
   effect of sarcasm labels from that of extra auxiliary data.
3. Attribution-based explanations for Hinglish abuse classifiers are rarely checked for faithfulness.
4. No openly available Hinglish conversation dataset with turn-level cyberbullying labels was found, so
   claims of *early* Hinglish cyberbullying detection cannot currently be validated with public data.
