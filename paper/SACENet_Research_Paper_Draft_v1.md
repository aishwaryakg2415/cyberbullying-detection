# SACENet: Sarcasm-Aware Explainable Detection of Cyberbullying in Hinglish Text

**VERSION 1 — INITIAL DRAFT (written 2026-10-17, after the repository audit and dataset verification, before any
transformer result existed).** Preserved unchanged for revision tracking; the evidence-updated manuscript is
`SACENet_Research_Paper_Draft.md`. Markers: **[PENDING: …]** = result not yet available; **[VERIFY: …]** = claim
the authors must check.

*Authors: [author names and affiliations to be added by the project owner]*

---

## Abstract

Cyberbullying on Indian social media is frequently written in Hinglish, a Roman-script mixture of Hindi and English,
and is often expressed indirectly through mockery and sarcasm. We present SACENet, a reproducible study of
sarcasm-aware and explainable cyberbullying classification on the MC-Hinglish1.0 corpus (8,400 comments, seven
cyberbullying types). We compare classical TF-IDF baselines with multilingual (MuRIL, XLM-R) and Hinglish-specific
(HingRoBERTa) transformers, and test three ways of injecting sarcasm knowledge learned from a separate,
human-annotated Hinglish sarcasm corpus: multitask learning, a shuffled-label multitask control, and feature fusion.
Explanations are produced with SHAP and checked for faithfulness with a deletion test. An early-warning procedure is
evaluated on a public conversational toxicity dataset as a proxy, because no sequentially annotated Hinglish
cyberbullying data is publicly available. We also document data issues found along the way: 18.2% of the official
MC-Hinglish test items have a near-duplicate in the training data, and the sarcasm corpus's retrieval hashtags leak
its labels. **[PENDING: headline results]**

**Keywords:** cyberbullying detection; code-mixing; Hinglish; sarcasm; multitask learning; explainability; SHAP; early detection

## 1. Introduction

Automated detection of cyberbullying and abusive language has been studied for more than a decade
\cite{salawu2020survey,vanhee2018cyberbullying,fortuna2018survey}. Most of this work concerns English. In India,
however, a large share of social-media communication is code-mixed. Users alternate between Hindi and English within a
sentence and usually write Hindi in Roman script, with no standard spelling \cite{khanuja2020gluecos,bohra2018hinglish}.
This creates three difficulties for detection models. First, vocabulary is sparse and noisy because the same Hindi word
appears in many transliterations. Second, pretrained models are mostly trained on standard-script, monolingual text.
Third, cyberbullying is often indirect: users mock a target with exaggerated praise or irony instead of explicit insults
\cite{joshi2017sarcasm}.

Sarcasm is a natural candidate for auxiliary knowledge because it is one of the mechanisms behind indirect harassment.
It is also hard to supply honestly. Corpora that annotate both cyberbullying and sarcasm on the same Hinglish texts are,
to our knowledge, not publicly available **[VERIFY: literature search up to 2026]**. Sarcasm signals must therefore come
from a *separate* corpus, and whether they transfer is an empirical question.

Explainability is needed because moderation decisions affect people. Token-attribution methods such as SHAP
\cite{lundberg2017shap} can show moderators which words a model relied on. Attributions, however, describe model
behaviour, not speaker intent, and they can be unfaithful \cite{jacovi2020faithfulness}. We therefore evaluate their
faithfulness rather than assume it.

Finally, the original motivation of SACENet included *early* detection of escalating cyberbullying in conversations
\cite{lopezvizcaino2021early}. That requires turn-level annotations of real conversations. We found none for Hinglish
cyberbullying that is openly available, so we evaluate the early-warning mechanism on a public English conversational
toxicity corpus (CONDA; \citealp{weld2021conda}) and present it explicitly as a proxy study.

**Research questions.**
RQ1: How do classical and transformer models (multilingual vs. Hinglish-specific) compare on MC-Hinglish1.0 when splits are leakage-controlled?
RQ2: Does sarcasm supervision from a separate human-annotated Hinglish corpus improve cyberbullying-type classification, and is any gain attributable to the sarcasm *labels* rather than to extra data?
RQ3: Are SHAP explanations of the resulting model faithful and stable?
RQ4: How early, and with how many false alarms, can a causal turn-level detector flag toxic conversations (proxy setting)?

## 2. Related Work

**Cyberbullying and abusive language.** Surveys cover feature-engineered and neural approaches
\cite{salawu2020survey,fortuna2018survey}. \citet{vanhee2018cyberbullying} model fine-grained cyberbullying roles and
categories. \citet{vidgen2020directions} review abusive-language training data and stress how annotation and
sampling choices shape what models learn. \citet{sap2019racial} show dialect-correlated bias in hate-speech labels.
Benchmarks include OLID/OffensEval \cite{zampieri2019olid}, TweetEval \cite{barbieri2020tweeteval}, HASOC
\cite{mandl2019hasoc}, TRAC \cite{kumar2018trac,kumar2020trac2} and HateXplain \cite{mathew2021hatexplain}.

**Hinglish and code-mixed NLP.** \citet{bohra2018hinglish} released Hinglish hate-speech tweets. GLUECoS
\cite{khanuja2020gluecos} benchmarks code-switched NLP. MuRIL \cite{khanuja2021muril} is pretrained on 17 Indian
languages, including transliterated text, and the L3Cube HingBERT family \cite{nayak2022hingbert} is pretrained on
Hinglish. MC-Hinglish1.0 \cite{laskar2024mchinglish} provides seven cyberbullying types; its authors report an
ensemble accuracy of 0.609. \citet{kumar2025murilxai} apply MuRIL with attribution analysis to six other Hinglish
cyberbullying and hate datasets.

**Sarcasm.** \citet{joshi2017sarcasm} survey sarcasm detection. For Hinglish, \citet{swami2018sarcasm} release
5,250 annotated tweets, \citet{aggarwal2020sarcasm} use bilingual embeddings, and \citet{majumder2026smallwins} compare
LLMs with fine-tuned small models. Multitask learning \cite{caruana1997multitask,ruder2017mtl} has been used to share
sentiment and sarcasm information \cite{majumder2019sarcasmmtl}.

**Multilingual transformers.** BERT \cite{devlin2019bert} and XLM-R \cite{conneau2020xlmr} are standard
multilingual baselines.

**Explainability.** LIME \cite{ribeiro2016lime}, SHAP \cite{lundberg2017shap} and Integrated Gradients
\cite{sundararajan2017ig} are common attribution methods. ERASER \cite{deyoung2020eraser} proposes
comprehensiveness/sufficiency metrics, and \citet{jacovi2020faithfulness} discuss faithfulness.

**Early detection.** eRisk introduced latency-aware evaluation (ERDE; \citealp{losada2016erisk}).
\citet{lopezvizcaino2021early} study early cyberbullying detection on social-network sessions, and
\citet{hosseinmardi2015instagram} annotate Instagram media sessions.

## 3. Research Gap and Contributions

Combining a multilingual encoder, a sarcasm signal, SHAP and early detection is not novel in itself. Our contribution
is an *evidence-controlled* evaluation of these components on Hinglish cyberbullying:
1. a leakage-audited pipeline for MC-Hinglish1.0, including quantification of train–test near-duplicates in the official split;
2. a controlled test of sarcasm transfer (multitask vs. shuffled-label control vs. feature fusion) from a human-annotated corpus whose retrieval-hashtag artefact we identify and remove;
3. faithfulness and stability checks of SHAP explanations;
4. a clearly scoped proxy evaluation of causal early warning, with delay, false-alarm and ERDE metrics.

## 4. Problem Formulation

Given a comment $x$, predict its MC-Hinglish type $y \in \mathcal{Y}$, $|\mathcal{Y}|=7$ (task **MC7**), or the
derived binary label $b = \mathbb{1}[y \neq \texttt{not\_cyberbullying}]$ (task **MC2**). For early detection, given a
conversation $(u_1,\dots,u_n)$ with turn labels $z_t \in \{0,1\}$, emit after each turn a score $p_t = f(u_{\le t})$ and an
alarm at $\hat t = \min\{t: a_t \ge \tau\}$, where $a_t$ is a causal aggregate of $p_{1..t}$.

## 5. Datasets

See `data/DATASET_CARD.md`. MC-Hinglish1.0: 7,400 train and 1,000 test comments; we carve a 741-item
validation set from the training split. Swami et al. sarcasm: 5,250 tweets (504 sarcastic); 70/10/20 split.
CONDA: 35,895 annotated utterances in 10,902 conversations, re-split by match. TweetEval-Offensive test: 860
English tweets.

## 6–11. Method, architecture, setup, metrics

*[Written in full in Version 2 after implementation was finalised.]* Planned: TF-IDF word+char n-grams with LR
and SVM; MuRIL, XLM-R base and HingRoBERTa fine-tuned with identical hyperparameters across 3 seeds; multitask
loss $\mathcal{L} = \mathcal{L}_{\text{CB}} + \lambda \mathcal{L}_{\text{sarc}}$; feature fusion with a frozen,
separately trained sarcasm model; temperature scaling; macro-F1 as the primary metric, with bootstrap CIs and paired tests.

## 12–17. Results

**[PENDING: all results — baseline, transformer, ablation, external tests, SHAP, early detection.]**

Baseline results available at the time of writing (test set, macro-F1): MC7 — majority 0.048, TF-IDF+LR 0.727,
TF-IDF+SVM 0.744; MC2 — majority 0.444, TF-IDF+LR 0.879, TF-IDF+SVM 0.887; sarcasm (hashtags removed) — TF-IDF+LR
0.739, TF-IDF+SVM 0.724 (0.879 with hashtags kept). **[VERIFY: the gap between our TF-IDF SVM (0.744) and the
0.548 reported by \citealp{laskar2024mchinglish} for SVM is probably due to character n-grams; it does not show that
our system is better.]**

## 18. Discussion — [PENDING]

## 19. Limitations and Ethical Considerations

* MC-Hinglish has no reported inter-annotator agreement; its categories mix target-based and language-based definitions.
* The sarcasm corpus is from a different period and set of topics (2017 tweets on cricket, Bollywood and triple talaq) than MC-Hinglish.
* The early-detection evaluation is a proxy (English game chat) and does not validate Hinglish early detection.
* Models can encode dialect, religious and gender biases. They should support human moderators, not replace them.

## 20. Conclusion — [PENDING]

## References — see `references.bib`
