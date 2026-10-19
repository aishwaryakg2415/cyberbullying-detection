# Author review notes

Items the authors must check or decide. The AI assistant that produced this draft could not settle them on its own.

## A. Must verify before any submission

1. **Provenance of `human_labelled_features.csv` / `ai_labelled_features.csv`.** Who collected them, from which
   platform, under what licence, and with what annotation protocol (number of annotators, guidelines, agreement)? Until
   this is documented, keep them out of the paper's main claims (currently: one supplementary table row only).
2. **Literature currency.** Search Google Scholar / Semantic Scholar for papers that cite MC-Hinglish1.0
   (doi:10.13053/CyS-28-3-4989) and for 2025–2026 Hinglish cyberbullying + sarcasm work. The "gap" statements in §3 are
   phrased conservatively but depend on this search.
3. **Characterisations of cited papers.** Descriptions were checked against titles/abstracts. Before submission, confirm
   against the PDFs, especially Van Hee et al. (2018), Salawu et al. (2020), López-Vizcaíno et al. (2021) and Majumder et
   al. (2019).
4. **Swami et al. (2018) collection procedure.** We infer from the data (486/504 sarcastic tweets contain `#sarcasm` /
   `#irony`) that the hashtags were retrieval queries. Confirm with the paper's data-collection section and cite it
   precisely.
5. **MC-Hinglish counts.** The per-class counts in the files differ slightly from the paper's table and the Hugging Face
   card. We report counts measured from the files; consider contacting the dataset authors.

## B. Decisions for the authors

1. **Title.** The brief's title ("SACENet: Sarcasm-Aware Explainable Detection of Cyberbullying in Hinglish Text") was
   changed to a question-style title because the sarcasm components are evaluated, not shown to help (see results).
   You may prefer to keep "SACENet" as the system name in the subtitle.
2. **Venue and format.** No venue was assumed. The LaTeX uses a generic two-column `article` class; switch to the target
   venue's template (ACL, IEEE, Springer LNCS) and re-check page limits.
3. **Whether to keep the early-detection proxy in the main paper or move it to an appendix**, given that it is English
   game chat rather than Hinglish cyberbullying.
4. **Quoting offensive examples.** Figure 16 shows short MC-Hinglish texts verbatim (CC-BY-NC-SA-4.0). Decide whether to
   mask slurs for the venue, and add a content warning at the top of the paper if they are kept.
5. **Author list, affiliations, funding and acknowledgements, ethics statement and data-availability statement.**

## C. Known weaknesses a reviewer will raise

* Hyperparameters are shared and untuned; the token-embedding table is frozen. A reviewer may ask for full fine-tuning
  with a per-model learning-rate search.
* Only three seeds; several comparisons without multiple-comparison correction.
* The sarcasm corpus differs from MC-Hinglish in time period (2017) and topics (cricket, Bollywood, triple talaq).
* No human evaluation of explanation usefulness.
* Early detection is not evaluated on Hinglish conversations.
