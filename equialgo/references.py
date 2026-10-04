"""Stand-ins for the hidden reference standard, and how much to trust each one.

The hidden standard Y* is unknown. We build several plausible versions of it, each a top-40% allocation
of a merit score fitted on training rows only, and weight them by how well they reproduce the one
number the organisers published about Y*: the production model's equal-opportunity gap of 0.270 on the
4,000 evaluation applicants.

    weight_r ∝ exp(−½ · ((EOD_r(baseline) − 0.270) / 0.03)²)

Scoring a decision rule against a weighted set, rather than against one reference, keeps the comparison
from rewarding a rule merely for being the same model as the reference it is judged against.

The twelve versions differ in what they assume the standard rewards: a constant penalty or one growing
with distance, merit with and without distance, R score alone, a 30% or 50% base rate, a noisy draw, the
committee's own decisions with deserving remote refusals flipped, a model fitted on centre applicants
only, and need measured as R score minus log income.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from equialgo import metrics
from equialgo.data import BASELINE_EOD_GAP, LABEL, features, group, is_remote

RATE = 0.40
WIDTH = 0.03


def _top(score: np.ndarray, reference_score: np.ndarray, rate: float = RATE) -> np.ndarray:
    return (score >= np.quantile(reference_score, 1 - rate)).astype(int)


def _logistic(train: pd.DataFrame, columns: list[str] | None = None, *, log_income: bool = False,
              distance_penalty: bool = False):
    """Committee rule with a remote control (and optionally a remote × distance term); returns a merit scorer."""

    def design(frame: pd.DataFrame) -> pd.DataFrame:
        X = features(frame, with_region=True)
        if log_income:
            X["revenu_familial_estime"] = np.log(frame["revenu_familial_estime"].to_numpy())
        if distance_penalty:
            X["el_dist"] = X["eloignee"] * X["distance_domicile_campus_km"] / 100
        keep = (columns or [c for c in X if c not in ("eloignee", "el_dist")]) + \
            [c for c in ("eloignee", "el_dist") if c in X]
        return X[keep]

    model = LogisticRegression(C=1e6, max_iter=10_000).fit(design(train), train[LABEL])

    def merit(frame: pd.DataFrame) -> np.ndarray:
        X = design(frame).assign(eloignee=0)
        if "el_dist" in X:
            X["el_dist"] = 0.0
        return model.decision_function(X)
    return merit, model


def build(train: pd.DataFrame, target: pd.DataFrame, seed: int = 0) -> dict[str, np.ndarray]:
    """Every candidate Y* for the rows of `target`, fitted on `train`.

    Each reference is a top-40% allocation, and the cutoff always comes from the training rows, so a
    reference never sees the distribution of the rows it labels.
    """

    columns = list(features(train).columns)
    no_distance = [c for c in columns if c != "distance_domicile_campus_km"]
    core = [c for c in no_distance if c not in ("heures_travail_semaine", "revenu_familial_estime")]
    constant, model = _logistic(train)
    merit_target, merit_train = constant(target), constant(train)
    out = {"A constant penalty": _top(merit_target, merit_train)}
    for name, scorer in (("B penalty growing with distance", _logistic(train, distance_penalty=True)[0]),
                         ("C merit without distance", _logistic(train, no_distance)[0]),
                         ("D merit from R, first generation, program", _logistic(train, core)[0]),
                         ("K constant penalty, log income", _logistic(train, log_income=True)[0])):
        out[name] = _top(scorer(target), scorer(train))
    out["E R score only"] = _top(target["cote_r_equivalent"].to_numpy(), train["cote_r_equivalent"].to_numpy())

    cutoff = np.quantile(merit_train, 1 - RATE)
    deserving = 1 / (1 + np.exp(-(merit_target - cutoff)))
    out["F noisy Bernoulli"] = (np.random.default_rng(seed).random(len(target)) < deserving).astype(int)
    out["G base rate 30%"] = _top(merit_target, merit_train, 0.30)
    out["H base rate 50%"] = _top(merit_target, merit_train, 0.50)

    committee = model.decision_function(features(target, with_region=True)[list(model.feature_names_in_)]) > 0
    out["I committee, deserving remote refusals flipped"] = (
        committee | ((is_remote(target) == 1) & (merit_target >= cutoff))).astype(int)

    centre = is_remote(train) == 0
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
    gbm.fit(features(train)[centre], train[LABEL][centre])
    out["J centre-only GBM"] = _top(gbm.predict_proba(features(target))[:, 1], gbm.predict_proba(features(train))[:, 1])

    def need(frame: pd.DataFrame) -> np.ndarray:
        """R score net of family wealth, with the committee's own weight on each (1.384 and 1.78)."""

        return (1.384 * frame["cote_r_equivalent"].to_numpy()
                - 1.78 * np.log(frame["revenu_familial_estime"].to_numpy()))

    out["L need: R minus log income"] = _top(need(target), need(train))
    return out


def weights(history: pd.DataFrame, candidates: pd.DataFrame, baseline_decision: np.ndarray) -> pd.DataFrame:
    """Plausibility of each reference from the production model's gap on the evaluation applicants."""

    rows = []
    for name, label in build(history, candidates).items():
        eod = metrics.equal_opportunity_difference(baseline_decision, label, group(candidates))
        rows.append({"reference": name, "baseline_eod": eod,
                     "likelihood": float(np.exp(-0.5 * ((eod - BASELINE_EOD_GAP) / WIDTH) ** 2))})
    table = pd.DataFrame(rows)
    table["weight"] = table["likelihood"] / table["likelihood"].sum()
    return table
