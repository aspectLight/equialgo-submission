"""The quantities the brief scores, each written out explicitly.

Notation: ŷ ∈ {0,1} is a decision, g the group (Centre or Eloignee), Y a reference label. Y may be hard
(0/1) or soft (a probability P(Y=1 | x)); with a soft label every rate is an expectation, e.g. the
expected number of deserving applicants granted divided by the expected number of deserving applicants.

- selection rate          SR_g  = mean(ŷ | g)
- demographic parity gap  DPD   = |SR_Centre − SR_Eloignee|
- true positive rate      TPR_g = Σ_g ŷ·Y / Σ_g Y            (share of deserving applicants who are granted)
- equal-opportunity gap   EOD   = |TPR_Centre − TPR_Eloignee|  (Hardt et al. 2016; scored against the hidden standard)
- gap closed              (0.270 − EOD) / 0.270, clipped to [0, 1]
- grant rate              GR    = mean(ŷ) on the 4,000 evaluation rows; valid iff 0.36 ≤ GR ≤ 0.44
- agreement               A     = mean(ŷ·Y + (1−ŷ)(1−Y))
- scaled utility          U     = (A − A_rand) / (1 − A_rand), A_rand = GR·p + (1−GR)(1−p), p = mean(Y):
                                  0 for a random draw at the same grant rate, 1 for a perfect allocation
- estimated points        P     = 20·(gap closed) + 15·clip(U, 0, 1), and 0 outside the envelope
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from equialgo.data import BASELINE_EOD_GAP, BUDGET, CENTRE, REMOTE


def selection_rate(decision, groups) -> dict[str, float]:
    decision, groups = np.asarray(decision, float), np.asarray(groups)
    return {g: float(decision[groups == g].mean()) for g in (CENTRE, REMOTE)}


def demographic_parity_difference(decision, groups) -> float:
    rates = selection_rate(decision, groups)
    return abs(rates[CENTRE] - rates[REMOTE])


def true_positive_rate(decision, label, groups) -> dict[str, float]:
    decision, label, groups = np.asarray(decision, float), np.asarray(label, float), np.asarray(groups)
    out = {}
    for g in (CENTRE, REMOTE):
        mask = groups == g
        deserving = label[mask].sum()
        out[g] = float((decision[mask] * label[mask]).sum() / deserving) if deserving > 0 else float("nan")
    return out


def equal_opportunity_difference(decision, label, groups) -> float:
    rates = true_positive_rate(decision, label, groups)
    return abs(rates[CENTRE] - rates[REMOTE])


def gap_closed(eod: float, baseline: float = BASELINE_EOD_GAP) -> float:
    return float(np.clip((baseline - eod) / baseline, 0.0, 1.0))


def grant_rate(decision) -> float:
    return float(np.asarray(decision, float).mean())


def within_budget(decision, budget: tuple[float, float] = BUDGET) -> bool:
    return budget[0] <= grant_rate(decision) <= budget[1]


def agreement(decision, label) -> float:
    decision, label = np.asarray(decision, float), np.asarray(label, float)
    return float(np.mean(decision * label + (1 - decision) * (1 - label)))


def scaled_utility(decision, label) -> float:
    q, p = grant_rate(decision), float(np.asarray(label, float).mean())
    random_agreement = q * p + (1 - q) * (1 - p)
    return (agreement(decision, label) - random_agreement) / (1 - random_agreement)


@dataclass(frozen=True)
class Report:
    grant_rate: float
    within_budget: bool
    sr_centre: float
    sr_eloignee: float
    dpd: float
    tpr_centre: float
    tpr_eloignee: float
    eod: float
    gap_closed: float
    agreement: float
    utility: float

    def as_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def report(decision, label, groups) -> Report:
    """Every scored quantity for one set of decisions against one reference label."""

    sr = selection_rate(decision, groups)
    tpr = true_positive_rate(decision, label, groups)
    eod = abs(tpr[CENTRE] - tpr[REMOTE])
    return Report(
        grant_rate=grant_rate(decision), within_budget=within_budget(decision),
        sr_centre=sr[CENTRE], sr_eloignee=sr[REMOTE], dpd=abs(sr[CENTRE] - sr[REMOTE]),
        tpr_centre=tpr[CENTRE], tpr_eloignee=tpr[REMOTE], eod=eod, gap_closed=gap_closed(eod),
        agreement=agreement(decision, label), utility=scaled_utility(decision, label),
    )


def estimated_points(decision, label, groups) -> float:
    """The 35 automated points these decisions would score if `label` were the hidden standard."""

    scored = report(decision, label, groups)
    if not scored.within_budget:
        return 0.0
    return 20 * scored.gap_closed + 15 * float(np.clip(scored.utility, 0.0, 1.0))
