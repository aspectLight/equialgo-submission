"""A merit score with the committee's regional penalty removed, and allocation under a fixed budget.

The committee's decisions are well described by a logistic rule (cross-validated AUC 0.956):

    logit P(granted) = β·x + γ·eloignee,   γ ≈ −2.2

where x are the application's own characteristics (R score, income, hours worked, distance, first
generation, program) and `eloignee` is 1 for the three remote regions. γ is the regional penalty: at
identical characteristics, a remote applicant's odds are divided by about 9.

Fitting the rule *with* the remote indicator as a control is what keeps the proxies honest: without it,
distance (AUC 0.997 for predicting the region on its own) absorbs the penalty. The merit score is then
β·x alone, which needs neither the region nor the postal code at decision time. `alpha` moves between
the committee (alpha = 0, the full penalty) and the corrected rule (alpha = 1, no penalty); alpha > 1
over-corrects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from equialgo.data import LABEL, features, is_remote


class MeritModel:
    """Logistic model of the committee's decisions with an explicit regional penalty term."""

    def __init__(self, *, drop: tuple[str, ...] = (), log_income: bool = False) -> None:
        self.model = LogisticRegression(C=1e6, max_iter=10_000)
        self.columns: list[str] = []
        self.drop = tuple(drop)
        self.log_income = log_income

    def _features(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = features(frame, with_region=True)
        if self.log_income:
            X["revenu_familial_estime"] = np.log(frame["revenu_familial_estime"].to_numpy())
        return X.drop(columns=list(self.drop))

    def fit(self, history: pd.DataFrame) -> MeritModel:
        X = self._features(history)
        self.columns = list(X.columns)
        self.model.fit(X, history[LABEL])
        return self

    @property
    def coefficients(self) -> pd.Series:
        return pd.Series(self.model.coef_[0], index=self.columns)

    @property
    def penalty(self) -> float:
        """γ, the logit shift the committee applies to remote applicants."""

        return float(self.coefficients["eloignee"])

    def merit_logit(self, frame: pd.DataFrame) -> np.ndarray:
        """β·x + intercept: the committee's rule evaluated as if every applicant were from a centre."""

        return self.model.decision_function(self._features(frame).assign(eloignee=0)[self.columns])

    def logit(self, frame: pd.DataFrame, alpha: float = 1.0) -> np.ndarray:
        """Score with a fraction (1 − alpha) of the regional penalty kept."""

        return self.merit_logit(frame) + (1.0 - alpha) * self.penalty * is_remote(frame)

    def merit_probability(self, frame: pd.DataFrame) -> np.ndarray:
        """P(deserving | x) under the corrected rule: the soft reference label used to evaluate decisions."""

        return 1.0 / (1.0 + np.exp(-self.merit_logit(frame)))


def allocate(score: np.ndarray, rate: float, *, remote: np.ndarray | None = None, offset: float = 0.0) -> np.ndarray:
    """Grant exactly round(rate·n) awards to the highest scores, after adding `offset` to remote applicants.

    A non-zero offset is a group-specific threshold (Hardt et al. 2016; Corbett-Davies et al. 2017): under
    a fixed number of awards it moves grants between the two groups.
    """

    adjusted = np.asarray(score, float).copy()
    if remote is not None:
        adjusted = adjusted + offset * np.asarray(remote, float)
    k = round(float(rate) * len(adjusted))
    order = np.argsort(-adjusted, kind="stable")
    decision = np.zeros(len(adjusted), dtype=int)
    decision[order[:k]] = 1
    return decision
