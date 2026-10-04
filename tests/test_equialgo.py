"""The scoring and allocation functions on hand-checked cases. Pure functions, no fitting: fast."""

import numpy as np
import pandas as pd
import pytest

from equialgo import metrics
from equialgo.data import CENTRE, PROGRAMS, REMOTE, features
from equialgo.fair_score import allocate

GROUPS = np.array([CENTRE] * 4 + [REMOTE] * 4)


def test_selection_rate_and_demographic_parity():
    decision = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    assert metrics.selection_rate(decision, GROUPS) == {CENTRE: 0.75, REMOTE: 0.25}
    assert metrics.demographic_parity_difference(decision, GROUPS) == pytest.approx(0.5)


def test_equal_opportunity_difference():
    label = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    decision = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    assert metrics.true_positive_rate(decision, label, GROUPS) == {CENTRE: 1.0, REMOTE: 0.5}
    assert metrics.equal_opportunity_difference(decision, label, GROUPS) == 0.5


def test_true_positive_rate_is_nan_when_a_group_has_no_deserving_applicant():
    label = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    rates = metrics.true_positive_rate(np.ones(8, int), label, GROUPS)
    assert rates[CENTRE] == 1.0
    assert np.isnan(rates[REMOTE])


def test_gap_closed_is_clipped():
    assert metrics.gap_closed(0.0) == 1.0
    assert metrics.gap_closed(0.135) == pytest.approx(0.5)
    assert metrics.gap_closed(0.4) == 0.0


def test_scaled_utility_is_one_for_perfect_and_zero_for_independent():
    label = np.array([1, 0] * 50)
    assert metrics.scaled_utility(label, label) == pytest.approx(1.0)
    assert metrics.scaled_utility(np.array([1, 1, 0, 0] * 25), label) == pytest.approx(0.0)


def test_soft_labels_give_expected_rates():
    """A reference label may be a probability; every rate is then an expectation."""

    label = np.array([1.0, 0.0, 1.0, 0.0] * 2)
    half = np.full(8, 0.5)
    assert metrics.true_positive_rate(np.ones(8, int), half, GROUPS) == {CENTRE: 1.0, REMOTE: 1.0}
    assert metrics.agreement(np.ones(8, int), label) == pytest.approx(0.5)


def test_estimated_points_are_zero_outside_the_envelope():
    label = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    assert metrics.estimated_points(np.ones(8, int), label, GROUPS) == 0.0
    assert metrics.estimated_points(np.array([1, 0, 0, 0, 1, 0, 0, 0]), label, GROUPS) == 0.0  # 25% < 36%


def test_estimated_points_are_full_marks_for_a_fair_and_perfect_allocation():
    """Twenty applicants, ten per group, four deserving in each; both allocations grant 40%."""

    groups = np.array([CENTRE] * 10 + [REMOTE] * 10)
    label = np.array(([1] * 4 + [0] * 6) * 2)

    fair = label.copy()  # every deserving applicant granted, in both groups
    assert metrics.grant_rate(fair) == pytest.approx(0.40)
    assert metrics.estimated_points(fair, label, groups) == pytest.approx(35.0)

    # The same 40%, but the four remote awards go to applicants the standard calls undeserving.
    unfair = np.array([1] * 4 + [0] * 6 + [0] * 4 + [1] * 4 + [0] * 2)
    assert metrics.grant_rate(unfair) == pytest.approx(0.40)
    assert metrics.equal_opportunity_difference(unfair, label, groups) == pytest.approx(1.0)
    assert metrics.estimated_points(unfair, label, groups) == pytest.approx(2.5)


def test_allocate_grants_exactly_the_rate_and_offset_moves_grants():
    score = np.arange(10, dtype=float)
    remote = np.array([0] * 5 + [1] * 5)
    assert allocate(score, 0.4).sum() == 4
    assert allocate(score, 0.4, remote=remote, offset=-10).tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]


def test_allocate_is_a_fixed_number_of_awards_whatever_the_offset():
    score = np.arange(100, dtype=float)
    remote = np.array([0, 1] * 50)
    for offset in (-5.0, 0.0, 5.0):
        assert allocate(score, 0.41, remote=remote, offset=offset).sum() == 41


def _frame(program: str) -> pd.DataFrame:
    return pd.DataFrame({
        "cote_r_equivalent": [28.0], "revenu_familial_estime": [50_000.0],
        "heures_travail_semaine": [10.0], "distance_domicile_campus_km": [12.0],
        "premiere_generation_universitaire": [0], "programme_etudes": [program],
        "region_administrative": ["Montreal"],
    })


def test_features_rejects_an_unknown_program():
    """A program outside the five would otherwise become an all-zero dummy row and be scored silently."""

    with pytest.raises(ValueError, match="programs outside"):
        features(_frame("Droit"))


def test_features_scales_income_and_drops_one_program_dummy():
    out = features(_frame(PROGRAMS[0]), with_region=True)
    assert out["revenu_familial_estime"].iloc[0] == pytest.approx(5.0)
    assert sum(c.startswith("prog_") for c in out.columns) == len(PROGRAMS) - 1
    assert out["eloignee"].iloc[0] == 0
