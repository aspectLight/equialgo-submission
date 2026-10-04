"""Measurements used by the audit notebook: proxy leakage and the decomposition of the committee's gap."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from equialgo.data import LABEL, NUMERIC, POSTAL, PROGRAM, features, group, is_remote
from equialgo.fair_score import MeritModel

FEATURE_SETS = {
    "code_postal_3": [POSTAL],
    "distance_domicile_campus_km": ["distance_domicile_campus_km"],
    "heures_travail_semaine": ["heures_travail_semaine"],
    "revenu_familial_estime": ["revenu_familial_estime"],
    "premiere_generation_universitaire": ["premiere_generation_universitaire"],
    "programme_etudes": [PROGRAM],
    "cote_r_equivalent": ["cote_r_equivalent"],
    "the four named proxies": [POSTAL, "distance_domicile_campus_km", "heures_travail_semaine", "revenu_familial_estime"],
    "everything except region and postal code": [*NUMERIC, PROGRAM],
}


def _design(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    categorical = [c for c in columns if c in (POSTAL, PROGRAM)]
    return pd.get_dummies(frame[columns], columns=categorical).astype(float)


def _normalised_mi(values: pd.Series, target: np.ndarray) -> float:
    binned = values if values.dtype == object or values.nunique() <= 20 else pd.qcut(values, 20, duplicates="drop")
    h_target = mutual_info_score(target, target)
    return float(mutual_info_score(target, binned.astype(str)) / h_target)


def proxy_leakage(frame: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """How well each feature set predicts the region group once the region column is removed.

    AUC is the 5-fold cross-validated ROC-AUC of a gradient-boosting classifier predicting Éloignée
    from the set (0.5: no regional information; 1.0: the set identifies the group exactly).
    NMI is I(feature; group) / H(group) for single features (quantile-binned numerics).
    """

    target = is_remote(frame)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for name, columns in FEATURE_SETS.items():
        proba = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, random_state=seed),
                                  _design(frame, columns), target, cv=folds, method="predict_proba")[:, 1]
        nmi = _normalised_mi(frame[columns[0]], target) if len(columns) == 1 else np.nan
        rows.append({"feature set": name, "AUC predicting the region group": roc_auc_score(target, proba),
                     "normalised mutual information": nmi})
    return pd.DataFrame(rows)


def committee_rule_without_region_control(history: pd.DataFrame) -> pd.Series:
    """The committee's rule fitted without the remote indicator: the proxies absorb the penalty."""

    model = LogisticRegression(C=1e6, max_iter=10_000).fit(features(history), history[LABEL])
    return pd.Series(model.coef_[0], index=features(history).columns)


def decomposition(history: pd.DataFrame, model: MeritModel) -> pd.DataFrame:
    """Split the committee's grant-rate gap into a profile part and a penalty part.

    For each group: the observed grant rate, the rate the fitted committee rule predicts, and the rate
    the same rule predicts with the regional penalty removed (every applicant scored as if from a centre).
    """

    groups = group(history)
    with_penalty = 1 / (1 + np.exp(-model.logit(history, alpha=0.0)))
    without = model.merit_probability(history)
    table = pd.DataFrame({"group": groups, "observed": history[LABEL], "committee rule": with_penalty,
                          "rule without penalty": without})
    return table.groupby("group").mean(numeric_only=True)
