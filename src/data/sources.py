"""Dataset source registry (official locations, pinned revisions, licences).

Every URL here was checked on 2026-10-17. Revisions are pinned so that a
re-download yields byte-identical files (verified by SHA-256 in
data/dataset_manifest.json).
"""
from __future__ import annotations

MC_REV = "a8ae581fb5fcea58ae2b7642b86251d464af3e96"
SWAMI_REV = "7c259dc56d7e710cc55da0544bf1132b41d39c1d"
CONDA_REV = "3fb90675f881079c85c1de1f25df1bf8d521194f"
TWEETEVAL_REV = "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"

SOURCES: dict[str, dict] = {
    "mc_hinglish": {
        "name": "MC-Hinglish1.0",
        "citation_key": "laskar2024mchinglish",
        "paper": "Rahman-Laskar, Gupta, Badhani & Pinto-Avendano (2024). Cyberbullying Detection in a "
                 "Multi-classification Codemixed Dataset. Computacion y Sistemas 28(3). "
                 "doi:10.13053/CyS-28-3-4989",
        "official_page": "https://huggingface.co/datasets/sahinurlaskar/MC-Hinglish1.0",
        "also": "https://github.com/sahinurlaskar/MC-Hinglish1.0",
        "license": "CC-BY-NC-SA-4.0",
        "access": "public (no gating)",
        "language": "hi-en",
        "files": {
            "train.csv": f"https://huggingface.co/datasets/sahinurlaskar/MC-Hinglish1.0/resolve/{MC_REV}/Train_data_MC-Hinglish1.0.csv",
            "test.csv": f"https://huggingface.co/datasets/sahinurlaskar/MC-Hinglish1.0/resolve/{MC_REV}/Test_data_MC-Hinglish1.0.csv",
        },
        "revision": MC_REV,
        "role": "primary train/val/test (cyberbullying type, 7 classes; derived binary)",
    },
    "swami_sarcasm": {
        "name": "English-Hindi Code-Mixed Sarcasm Corpus (Swami et al., 2018)",
        "citation_key": "swami2018sarcasm",
        "paper": "Swami, Khandelwal, Singh, Akhtar & Shrivastava (2018). A Corpus of English-Hindi "
                 "Code-Mixed Tweets for Sarcasm Detection. arXiv:1805.11869",
        "official_page": "https://github.com/sahilswami96/SarcasmDetection_CodeMixed",
        "license": "GPL-3.0 (repository licence)",
        "access": "public",
        "language": "hi-en",
        "files": {
            "Sarcasm_tweets.txt": f"https://raw.githubusercontent.com/sahilswami96/SarcasmDetection_CodeMixed/{SWAMI_REV}/Dataset/Sarcasm_tweets.txt",
            "Sarcasm_tweet_truth.txt": f"https://raw.githubusercontent.com/sahilswami96/SarcasmDetection_CodeMixed/{SWAMI_REV}/Dataset/Sarcasm_tweet_truth.txt",
            "Sarcasm_tweets_with_language.txt": f"https://raw.githubusercontent.com/sahilswami96/SarcasmDetection_CodeMixed/{SWAMI_REV}/Dataset/Sarcasm_tweets_with_language.txt",
        },
        "revision": SWAMI_REV,
        "role": "human-annotated sarcasm supervision for the sarcasm component",
    },
    "conda": {
        "name": "CONDA: CONtextual Dual-Annotated in-game toxicity dataset",
        "citation_key": "weld2021conda",
        "paper": "Weld et al. (2021). CONDA: a CONtextual Dual-Annotated dataset for in-game toxicity "
                 "understanding and detection. Findings of ACL-IJCNLP 2021. doi:10.18653/v1/2021.findings-acl.213",
        "official_page": "https://github.com/usydnlp/CONDA",
        "license": "not specified in repository (used for non-commercial research evaluation only; not redistributed)",
        "access": "public",
        "language": "en",
        "files": {
            "CONDA_train.csv": f"https://raw.githubusercontent.com/usydnlp/CONDA/{CONDA_REV}/data/CONDA_train.csv",
            "CONDA_valid.csv": f"https://raw.githubusercontent.com/usydnlp/CONDA/{CONDA_REV}/data/CONDA_valid.csv",
        },
        "revision": CONDA_REV,
        "role": "PROXY sequential dataset for the early-detection mechanism (English game chat toxicity, "
                "not cyberbullying, not Hinglish). Official test file is unannotated and not used.",
    },
    "tweeteval_offensive": {
        "name": "TweetEval - Offensive (OLID / OffensEval-2019 data)",
        "citation_key": "barbieri2020tweeteval",
        "paper": "Barbieri et al. (2020) TweetEval, Findings of EMNLP 2020; data from Zampieri et al. (2019) OLID, NAACL 2019",
        "official_page": "https://huggingface.co/datasets/cardiffnlp/tweet_eval",
        "license": "see OLID terms (TweetEval card lists 'unknown')",
        "access": "public",
        "language": "en",
        "files": {
            "test.parquet": f"https://huggingface.co/datasets/cardiffnlp/tweet_eval/resolve/{TWEETEVAL_REV}/offensive/test-00000-of-00001.parquet",
        },
        "revision": TWEETEVAL_REV,
        "role": "English external stress test only (label = offensive language, NOT cyberbullying)",
    },
    "user_sarcasm_human": {
        "name": "User-supplied Hinglish sarcasm comments (human + AI labels)",
        "citation_key": None,
        "paper": "UNKNOWN - provenance not documented by the project; could not be matched to a public release",
        "official_page": None,
        "license": "UNKNOWN",
        "access": "local file supplied by the project owner (human_labelled_features.csv)",
        "language": "hi-en (+ some Devanagari / Marathi)",
        "files": {"human_labelled_features.csv": None},
        "role": "SUPPLEMENTARY cross-corpus sarcasm test only; excluded from all headline results",
    },
    "user_sarcasm_ai": {
        "name": "User-supplied Hinglish comments with LLM-generated sarcasm labels",
        "citation_key": None,
        "paper": "UNKNOWN",
        "official_page": None,
        "license": "UNKNOWN",
        "access": "local file supplied by the project owner (ai_labelled_features.csv)",
        "language": "hi-en",
        "files": {"ai_labelled_features.csv": None},
        "role": "NOT USED for training or evaluation: labels are machine-generated, not ground truth",
    },
}

# Candidates investigated but not usable without manual steps (see data/README.md).
UNAVAILABLE = {
    "hasoc": "HASOC (FIRE 2019-2021): data is released to registered participants via the organisers' "
             "form at https://hasocfire.github.io/ . Unofficial HF re-uploads exist but have no clear "
             "licence, so they are not used.",
    "trac": "TRAC-1/TRAC-2 aggression data: distributed by the organisers on request "
            "(https://sites.google.com/view/trac2/ ; https://sites.google.com/view/trac1/). Not downloaded.",
    "bohra2018": "Bohra et al. (2018) Hinglish hate speech: repository ships tweet IDs + labels only "
                 "(id_annotated.tsv); text requires X/Twitter API hydration, which is not available.",
    "olid_official": "OLID official zip is hosted on a Google Sites page; the TweetEval redistribution "
                     "(identical OffensEval-2019 test tweets) is used instead.",
}
