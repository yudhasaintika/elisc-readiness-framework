#!/usr/bin/env python3
"""
SAMPLE SIZE ANALYSIS FOR THE ELISC MACHINE LEARNING LAYER
=========================================================
The readiness label is ordered (Low < Moderate < High). Treating it as a nominal
multiclass target discards that ordering. This script therefore adds an ordinal
classifier built by the Frank & Hall decomposition: for K ordered classes, K-1
binary models estimate P(y > k), and class probabilities follow by differencing.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import worked_example as W        # reuse the validated pipeline

SEED = 20260722
N_GRID = [50, 100, 150, 200, 300, 500, 800]
CV = dict(n_splits=5, n_repeats=5, random_state=SEED)


class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    """Frank & Hall ordinal decomposition around any binary probabilistic base learner."""

    def __init__(self, base=None):
        self.base = base

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = []
        base = self.base if self.base is not None else make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
        for k in self.classes_[:-1]:
            m = clone(base)
            m.fit(X, (np.asarray(y) > k).astype(int))
            self.models_.append(m)
        return self

    def predict_proba(self, X):
        cum = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_])
        cum = np.clip(cum, 1e-6, 1 - 1e-6)
        probs = [1 - cum[:, 0]]
        for i in range(cum.shape[1] - 1):
            probs.append(np.clip(cum[:, i] - cum[:, i + 1], 1e-6, None))
        probs.append(cum[:, -1])
        p = np.column_stack(probs)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def models():
    return {
        "OrdinalLogistic": OrdinalClassifier(),
        "LogisticRegression": make_pipeline(StandardScaler(),
                                            LogisticRegression(max_iter=2000, random_state=SEED)),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED),
    }


def one_sample(n):
    """Generate n units, run the ELISC pipeline, and build the three feature sets."""
    W.N_UNITS = n
    W.rng = np.random.default_rng(SEED)
    df, eta, sub_index = W.generate()
    C, Dk = W.run_pipeline(df)
    y, _ = W.make_labels(eta, sub_index)

    X_ri = Dk[["RI"]].reset_index(drop=True)
    X_dim = Dk[W.DIMS].reset_index(drop=True)
    X_sub = C.reset_index(drop=True)
    X_sub.columns = [f"{a}__{b}" for a, b in X_sub.columns]
    return {"1. RI only": X_ri, "2. dimension scores": X_dim,
            "3. sub-dimension scores": X_sub}, y


if __name__ == "__main__":
    cv = RepeatedStratifiedKFold(**CV)
    rows, deltas = [], []

    for n in N_GRID:
        sets, y = one_sample(n)
        fold_store = {}
        for fs, X in sets.items():
            for name, m in models().items():
                sc = cross_val_score(m, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
                fold_store[(fs, name)] = sc
                se = sc.std(ddof=1) / np.sqrt(len(sc))
                rows.append({"N": n, "feature_set": fs, "model": name,
                             "macro_F1": round(sc.mean(), 4),
                             "ci95_halfwidth": round(1.96 * se, 4)})
        for name in models():
            base = fold_store[("1. RI only", name)]
            for fs in ["2. dimension scores", "3. sub-dimension scores"]:
                d = fold_store[(fs, name)] - base
                se = d.std(ddof=1) / np.sqrt(len(d))
                lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
                deltas.append({"N": n, "model": name, "feature_set": fs,
                               "delta_F1": round(d.mean(), 4),
                               "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                               "excludes_zero": "yes" if lo > 0 else "no"})
        print(f"  N = {n} done")

    res = pd.DataFrame(rows)
    dl = pd.DataFrame(deltas)

    print("\n=== macro-F1 by sample size (best model per feature set) ===")
    best = res.loc[res.groupby(["N", "feature_set"])["macro_F1"].idxmax()]
    print(best.pivot(index="N", columns="feature_set",
                     values="macro_F1").to_string())

    print("\n=== width of the 95% interval (precision of the estimate) ===")
    print(res.groupby("N")["ci95_halfwidth"].mean().round(4).to_string())

    print("\n=== OrdinalLogistic vs the nominal alternatives ===")
    print(res.pivot_table(index=["feature_set", "model"], columns="N",
                          values="macro_F1").round(3).to_string())

    print("\n=== at what N does granularity become reliably detectable? ===")
    for fs in ["2. dimension scores", "3. sub-dimension scores"]:
        sub = dl[dl.feature_set == fs]
        ok = sub[sub.excludes_zero == "yes"].groupby("N").size()
        print(f"\n{fs} — number of models (out of 4) whose gain excludes zero:")
        print(ok.reindex(N_GRID, fill_value=0).to_string())

    res.to_csv("SS_TABLE1_learning_curve.csv", index=False)
    dl.to_csv("SS_TABLE2_paired_deltas.csv", index=False)
    print("\nTables written.")
