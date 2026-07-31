#!/usr/bin/env python3
"""
ELISC WORKED EXAMPLE ON SYNTHETIC DATA
======================================
Implements the ELISC pipeline end to end on a fully specified synthetic dataset,
with membership-function sensitivity analysis and an algorithm ablation study.

CRITICAL DESIGN POINT — breaking the label circularity
------------------------------------------------------
The readiness label y is NOT derived from the readiness index RI. It is generated
from the error-free latent scores through a function g() that is deliberately
structurally different from the linear index of Equation (10): it contains a
culture floor effect, a competency x direction interaction, and noise. The machine
learning layer is therefore tested on a real question — whether the framework's
granular features recover an external readiness signal that the linear index does
not fully capture — rather than re-predicting its own output.
"""

import numpy as np
import pandas as pd
from itertools import product
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

SEED = 20260722
rng = np.random.default_rng(SEED)


class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    """
    For K ordered classes, K-1 binary models estimate P(y > k); class probabilities
    follow by differencing. This exploits the ordering of the readiness label, which
    nominal multiclass learners discard.
    """

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

# ---------------------------------------------------------------- configuration
N_UNITS = 300              # simulated assessment units
N_ITEMS = 3                # indicators per sub-dimension
LIKERT = 5

STRUCTURE = {
    "collaboration": ["interaction", "community_engagement", "crowdsourcing",
                      "co_creation", "knowledge_sharing"],
    "competency":    ["digital_technical", "governance_management",
                      "innovation_ethics", "digital_change"],
    "direction":     ["vision", "value_objectives", "strategy"],
    "culture":       ["character", "digital_culture", "organization_culture"],
}
DIMS = list(STRUCTURE)
BETA = {d: 0.25 for d in DIMS}

MF = {"Low": (0.00, 0.00, 0.45), "Medium": (0.25, 0.50, 0.75), "High": (0.55, 1.00, 1.00)}
OUT = {"Not Ready": 0.08, "Early": 0.30, "Developing": 0.50, "Ready": 0.70, "Advanced": 0.92}
LEVEL = {"Low": 1, "Medium": 2, "High": 3}
CUTS = [(1.4, "Not Ready"), (1.8, "Early"), (2.2, "Developing"), (2.6, "Ready"), (np.inf, "Advanced")]
FUZZY_SPREAD = 0.125

# ---------------------------------------------------------------- data generation
def generate():
    """Simulate a measurement model: latent sub-dimension scores -> Likert indicators."""
    sub_index = [(d, s) for d in DIMS for s in STRUCTURE[d]]
    n_sub = len(sub_index)

    # Dimension-level latent factors, moderately correlated (rho = 0.35)
    rho = 0.35
    cov = np.full((4, 4), rho); np.fill_diagonal(cov, 1.0)
    dim_factor = rng.multivariate_normal(np.zeros(4), cov, size=N_UNITS)

    # Sub-dimension latents load on their dimension factor (0.75) plus unique variance
    eta = np.zeros((N_UNITS, n_sub))
    for j, (d, _) in enumerate(sub_index):
        k = DIMS.index(d)
        eta[:, j] = 0.75 * dim_factor[:, k] + np.sqrt(1 - 0.75 ** 2) * rng.standard_normal(N_UNITS)

    # Observed indicators: loadings U(0.65,0.85), error to target CR ~ 0.8
    rows = []
    for j, (d, s) in enumerate(sub_index):
        for i in range(1, N_ITEMS + 1):
            lam = rng.uniform(0.65, 0.85)
            x = lam * eta[:, j] + np.sqrt(max(1 - lam ** 2, 0.05)) * rng.standard_normal(N_UNITS)
            # map to a 1-5 Likert scale
            resp = np.clip(np.round(3 + x * 1.1), 1, LIKERT)
            rows.append(pd.DataFrame({"unit": np.arange(N_UNITS), "dimension": d,
                                      "sub_dimension": s, "item": f"{s}_{i}",
                                      "response": resp}))
    return pd.concat(rows, ignore_index=True), eta, sub_index


# ---------------------------------------------------------------- fuzzy pipeline
def tfn(v, spread=FUZZY_SPREAD):
    return (max(0.0, v - spread), v, min(1.0, v + spread))


def tri(x, p):
    l, m, u = p
    if x <= l or x >= u: return 0.0
    if x == m: return 1.0
    return (x - l) / (m - l) if x < m else (u - x) / (u - m)


def trap(x, p, plateau=0.10):
    l, m, u = p
    a, b, cc, d = l, max(l, m - plateau), min(u, m + plateau), u
    if x <= a or x >= d: return 0.0
    if b <= x <= cc: return 1.0
    return (x - a) / (b - a) if x < b else (d - x) / (d - cc)


def gauss(x, p):
    l, m, u = p
    sig = max((u - l) / 4.0, 1e-6)
    return float(np.exp(-0.5 * ((x - m) / sig) ** 2))


FAMILIES = {"triangular": tri, "trapezoidal": trap, "gaussian": gauss}


def rule_base(beta):
    rules = []
    for combo in product(["Low", "Medium", "High"], repeat=4):
        ante = dict(zip(DIMS, combo))
        s = sum(beta[d] * LEVEL[ante[d]] for d in DIMS)
        cons = next(t for cut, t in CUTS if s < cut)
        if ante["culture"] == "Low":
            order = list(OUT)
            if order.index(cons) > order.index("Developing"):
                cons = "Developing"
        rules.append((ante, cons))
    return rules


def infer(dim_scores, rules, fam="triangular"):
    fn = FAMILIES[fam]
    mus = {d: {t: fn(v, p) for t, p in MF.items()} for d, v in dim_scores.items()}
    strength = {}
    for ante, cons in rules:
        a = min(mus[d][ante[d]] for d in DIMS)
        if a > 0:
            strength[cons] = max(strength.get(cons, 0.0), a)
    if not strength:
        return "Undetermined", 0.0
    num = sum(a * OUT[t] for t, a in strength.items())
    return max(strength, key=strength.get), num / sum(strength.values())


def run_pipeline(df, beta=BETA, fam="triangular"):
    """Equations (3)-(10): fuzzify -> aggregate -> defuzzify -> C_kj -> D_k -> RI."""
    per_sub = (df.groupby(["unit", "dimension", "sub_dimension"])["response"]
                 .mean().reset_index())
    per_sub["norm"] = (per_sub["response"] - 1) / (LIKERT - 1)
    t = per_sub["norm"].apply(tfn)
    per_sub["C_kj"] = [(a + b + c) / 3 for a, b, c in t]

    C = per_sub.pivot_table(index="unit", columns=["dimension", "sub_dimension"], values="C_kj")
    Dk = pd.DataFrame({d: C[d].mean(axis=1) for d in DIMS})       # equal alpha weights, Eq (6)-(7)
    RI = sum(beta[d] * Dk[d] for d in DIMS)                        # Eq (10)

    rules = rule_base(beta)
    cats, crisps = [], []
    for _, r in Dk.iterrows():
        cat, cr = infer(r.to_dict(), rules, fam)
        cats.append(cat); crisps.append(cr)
    Dk = Dk.assign(RI=RI, fuzzy_category=cats, defuzzified=crisps)
    return C, Dk


# ---------------------------------------------------------------- external label
def make_labels(eta, sub_index):
    """External readiness signal g(): NOT a function of RI.

    Contains (a) a culture floor effect, (b) a competency x direction interaction,
    and (c) unmodelled contextual noise. This is what breaks the circularity.
    """
    z = lambda a: (a - a.mean()) / a.std()
    pos = {s: j for j, (_, s) in enumerate(sub_index)}
    g = lambda s: z(eta[:, pos[s]])

    # Heterogeneous SUB-DIMENSION weights. If g() depended only on dimension means,
    # the aggregate index would capture the signal by construction and the test of
    # granularity would be vacuous. Specific sub-dimensions are therefore weighted
    # unequally, so that granular features can in principle add information.
    R = (0.45 * g("digital_technical") + 0.35 * g("organization_culture")
         + 0.30 * g("knowledge_sharing") + 0.25 * g("strategy")
         + 0.10 * g("crowdsourcing") + 0.10 * g("character"))

    floor = np.where(g("digital_culture") < -0.8, -1.2, 0.0)   # culture floor effect
    inter = 0.35 * g("digital_technical") * g("vision")        # capability x strategic clarity
    noise = 0.45 * rng.standard_normal(eta.shape[0])
    R = R + floor + inter + noise

    q = np.quantile(R, [1 / 3, 2 / 3])
    return np.digitize(R, q), R                              # 0 = Low, 1 = Moderate, 2 = High


# ---------------------------------------------------------------- evaluation
def ablation(X, y, label):
    models = {
        "OrdinalLogistic": OrdinalClassifier(),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=400, random_state=SEED),
        "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=400, random_state=SEED),
        "LogisticRegression": make_pipeline(StandardScaler(),
                                            LogisticRegression(max_iter=2000, random_state=SEED)),
    }
    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(random_state=SEED, verbose=False,
                                                allow_writing_files=False)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    out, folds = [], {}
    for name, m in models.items():
        sc = cross_val_score(m, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        folds[name] = sc
        out.append({"feature_set": label, "model": name,
                    "macro_F1_mean": round(sc.mean(), 4), "macro_F1_sd": round(sc.std(), 4)})
    return pd.DataFrame(out), folds


if __name__ == "__main__":
    print(f"ELISC worked example | seed = {SEED}\n" + "=" * 70)

    df, eta, sub_index = generate()
    print(f"Simulated {N_UNITS} units x {len(sub_index)} sub-dimensions x {N_ITEMS} items "
          f"= {len(df)} responses")

    C, Dk = run_pipeline(df)
    print("\n=== TABLE 1: distribution of readiness classifications (triangular MF) ===")
    dist = Dk["fuzzy_category"].value_counts().rename_axis("category").reset_index(name="n")
    dist["percent"] = (100 * dist["n"] / len(Dk)).round(1)
    print(dist.to_string(index=False))

    print("\n=== TABLE 2: dimension scores and readiness index ===")
    desc = Dk[DIMS + ["RI"]].describe().T[["mean", "std", "min", "max"]].round(4)
    print(desc.to_string())

    # --- sensitivity across membership function families --------------------
    print("\n=== TABLE 3: sensitivity to the membership function family ===")
    base = Dk["fuzzy_category"]
    sens = []
    for fam in FAMILIES:
        _, D2 = run_pipeline(df, fam=fam)
        agree = (D2["fuzzy_category"] == base).mean()
        rho = Dk["defuzzified"].corr(D2["defuzzified"], method="spearman")
        sens.append({"membership_function": fam,
                     "classification_agreement": f"{agree:.1%}",
                     "spearman_rho_vs_triangular": round(rho, 4)})
    # weight perturbation
    skew = {"collaboration": 0.20, "competency": 0.30, "direction": 0.20, "culture": 0.30}
    _, D3 = run_pipeline(df, beta=skew)
    sens.append({"membership_function": "triangular (weights +/-20%)",
                 "classification_agreement": f"{(D3['fuzzy_category'] == base).mean():.1%}",
                 "spearman_rho_vs_triangular": round(Dk["RI"].corr(D3["RI"], method="spearman"), 4)})
    sens_df = pd.DataFrame(sens)
    print(sens_df.to_string(index=False))

    # --- machine learning layer --------------------------------------------
    y, R_true = make_labels(eta, sub_index)
    corr = np.corrcoef(Dk["RI"].values, R_true)[0, 1]
    print(f"\nCorrelation between RI and the external readiness signal: r = {corr:.3f}")
    print("(materially below 1.0 by construction — the label is not a transform of RI)")

    X_full = pd.concat([C.reset_index(drop=True),
                        Dk[DIMS].reset_index(drop=True)], axis=1)
    X_full.columns = [f"{a}__{b}" if isinstance(a, str) and b else str(a)
                      for a, b in (c if isinstance(c, tuple) else (c, "")
                                   for c in X_full.columns)]
    X_ri = Dk[["RI"]].reset_index(drop=True)

    print("\n=== TABLE 4: algorithm ablation across three feature sets ===")
    # Three nested feature sets. Note that D_k is an exact linear function of its own
    # C_kj, so the aggregate and its components are never placed in the same feature
    # set: doing so creates perfect collinearity and scrambles attribution.
    X_dim = Dk[DIMS].reset_index(drop=True)
    X_sub = C.reset_index(drop=True)
    X_sub.columns = [f"{a}__{b}" for a, b in X_sub.columns]

    sets = [("1. RI only", X_ri), ("2. dimension scores", X_dim), ("3. sub-dimension scores", X_sub)]
    tables, foldsets = [], {}
    for lbl, X in sets:
        tb, fl = ablation(X, y, lbl)
        tables.append(tb); foldsets[lbl] = fl
    abl = pd.concat(tables, ignore_index=True)
    print(abl.pivot(index="model", columns="feature_set", values="macro_F1_mean").to_string())

    print("\n=== TABLE 4b: paired within-model comparison vs the RI-only baseline ===")
    print("(same algorithm, same folds; a positive delta means granularity helps that learner)")
    rows = []
    for model in foldsets["1. RI only"]:
        base = foldsets["1. RI only"][model]
        for lbl in ["2. dimension scores", "3. sub-dimension scores"]:
            comp = foldsets[lbl][model]
            d = comp - base
            se = d.std(ddof=1) / np.sqrt(len(d))
            rows.append({"model": model, "feature_set": lbl,
                         "mean_delta_F1": round(d.mean(), 4),
                         "ci95_low": round(d.mean() - 1.96 * se, 4),
                         "ci95_high": round(d.mean() + 1.96 * se, 4)})
    paired = pd.DataFrame(rows)
    print(paired.to_string(index=False))

    best = abl.loc[abl.groupby("feature_set")["macro_F1_mean"].idxmax()]
    print("\nBest attainable performance per feature set:")
    print(best.to_string(index=False))
    print("\nChance level for three balanced classes: macro-F1 = 0.333")

    print("\n=== TABLE 5: feature importance ===")
    # SHAP is computed on the best-performing TREE-BASED model, since tree explainers
    # do not apply to the ordinal decomposition. Where the ordinal formulation is the
    # selected model, its threshold coefficients serve the explanatory purpose instead.
    TREE_MODELS = {"CatBoost", "RandomForest", "ExtraTrees",
                   "GradientBoosting", "HistGradientBoosting"}
    ranked = abl[abl.feature_set == "3. sub-dimension scores"].sort_values(
        "macro_F1_mean", ascending=False)
    top_model = next(r for r in ranked["model"] if r in TREE_MODELS)
    print(f"(computed on the best tree-based sub-dimension model: {top_model})")
    if HAS_CATBOOST and top_model == "CatBoost":
        m = CatBoostClassifier(random_state=SEED, verbose=False,
                               allow_writing_files=False).fit(X_sub, y)
    elif top_model == "RandomForest":
        m = RandomForestClassifier(n_estimators=400, random_state=SEED).fit(X_sub, y)
    elif top_model == "ExtraTrees":
        m = ExtraTreesClassifier(n_estimators=400, random_state=SEED).fit(X_sub, y)
    elif top_model == "GradientBoosting":
        m = GradientBoostingClassifier(random_state=SEED).fit(X_sub, y)
    else:
        m = HistGradientBoostingClassifier(random_state=SEED).fit(X_sub, y)
    if HAS_SHAP:
        sv = shap.TreeExplainer(m).shap_values(X_sub)
        sv = np.asarray(sv) if not isinstance(sv, list) else np.stack(sv, axis=-1)
        imp = np.abs(sv).mean(axis=tuple(a for a in range(sv.ndim) if a != 1))
        method = "mean |SHAP|"
    else:
        raise RuntimeError("SHAP is not installed; install it before generating "
                           "the table reported in the manuscript.")
    top = (pd.DataFrame({"feature": X_sub.columns, "importance": imp})
             .sort_values("importance", ascending=False).round(4))
    print(f"method: {method}")
    print(top.to_string(index=False))
    print("\nSub-dimensions weighted in the generative function g(): digital_technical (0.45), "
          "organization_culture (0.35), knowledge_sharing (0.30), strategy (0.25), "
          "crowdsourcing (0.10), character (0.10); digital_culture enters via the floor effect "
          "and vision via the interaction term.")

    for name, tbl in [("WE_TABLE1_classification", dist), ("WE_TABLE2_scores", desc.reset_index()),
                      ("WE_TABLE3_sensitivity", sens_df), ("WE_TABLE4_ablation", abl),
                      ("WE_TABLE5_importance", top)]:
        tbl.to_csv(f"{name}.csv", index=False)
    print("\nAll tables written to CSV.")
