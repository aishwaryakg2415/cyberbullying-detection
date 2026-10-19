"""Classical baselines: majority class, TF-IDF + Logistic Regression, TF-IDF + linear SVM.

Features: word 1-2-grams + character 2-5-grams (char_wb), sublinear TF. Char
n-grams are a standard way to absorb transliteration/spelling variation in
romanised Hindi. C is selected on the validation split only.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC


class TfidfFeatures:
    def __init__(self, max_features: int = 200000):
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                                    lowercase=True, token_pattern=r"(?u)\b\w+\b|[^\w\s]", max_features=max_features)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                                    lowercase=True, max_features=max_features)

    def fit_transform(self, texts):
        return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()

    def transform(self, texts):
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()


class MajorityBaseline:
    name = "majority"

    def fit(self, X_train, y_train, X_val=None, y_val=None, n_classes=None):
        self.k = n_classes or int(max(y_train)) + 1
        counts = np.bincount(y_train, minlength=self.k)
        self.cls = int(counts.argmax())
        self.prior = counts / counts.sum()
        self.selected = {"majority_class": self.cls}
        return self

    def predict_proba(self, texts):
        return np.tile(self.prior, (len(texts), 1))

    def predict(self, texts):
        return np.full(len(texts), self.cls)


class _TfidfLinear:
    grid = (0.1, 0.3, 1.0, 3.0, 10.0)

    def _make(self, C):
        raise NotImplementedError

    def fit(self, X_train, y_train, X_val, y_val, n_classes=None, seed: int = 0):
        self.seed = seed
        self.feats = TfidfFeatures()
        Xtr = self.feats.fit_transform(X_train)
        Xva = self.feats.transform(X_val)
        scores = {}
        t0 = time.perf_counter()
        for C in self.grid:
            m = self._make(C).fit(Xtr, y_train)
            scores[C] = f1_score(y_val, m.predict(Xva), average="macro")
        best = max(scores, key=scores.get)
        self.model = self._make(best).fit(Xtr, y_train)
        self.selected = {"C": best, "val_macro_f1_by_C": {str(k): v for k, v in scores.items()},
                         "fit_seconds": time.perf_counter() - t0,
                         "n_features": Xtr.shape[1]}
        return self

    def predict_proba(self, texts):
        return self.model.predict_proba(self.feats.transform(texts))

    def predict(self, texts):
        return self.predict_proba(texts).argmax(1)


class TfidfLogReg(_TfidfLinear):
    name = "tfidf_lr"

    def _make(self, C):
        return LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=self.seed)


class TfidfSVM(_TfidfLinear):
    """LinearSVC with sigmoid (Platt) calibration fitted by 5-fold CV on the training split."""
    name = "tfidf_svm"

    def _make(self, C):
        return CalibratedClassifierCV(
            LinearSVC(C=C, class_weight="balanced", max_iter=20000, random_state=self.seed),
            method="sigmoid", cv=5)


BASELINES = {"majority": MajorityBaseline, "tfidf_lr": TfidfLogReg, "tfidf_svm": TfidfSVM}
