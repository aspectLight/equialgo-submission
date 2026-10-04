"""Loading the ÉquiAlgo files and defining the two regional groups."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

ID = "id_candidat"
LABEL = "decision_octroi"
REGION = "region_administrative"
POSTAL = "code_postal_3"
PROGRAM = "programme_etudes"
NUMERIC = [
    "cote_r_equivalent",
    "revenu_familial_estime",
    "heures_travail_semaine",
    "distance_domicile_campus_km",
    "premiere_generation_universitaire",
]
PROGRAMS = ("Arts et lettres", "Genie", "Sante", "Sciences", "Sciences sociales")

CENTRE_REGIONS = ("Montreal", "Capitale-Nationale")
REMOTE_REGIONS = ("Bas-Saint-Laurent", "Cote-Nord", "Gaspesie-Iles-de-la-Madeleine")
CENTRE, REMOTE = "Centre", "Eloignee"

BUDGET = (0.36, 0.44)
BASELINE_EOD_GAP = 0.270


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (historical applications with the committee decision, the 4,000 applicants to score)."""

    history = pd.read_csv(DATA / "donnees_demandes.csv")
    candidates = pd.read_csv(DATA / "candidats_evaluation.csv")
    for frame in (history, candidates):
        unknown = set(frame[REGION]) - set(CENTRE_REGIONS) - set(REMOTE_REGIONS)
        if unknown:
            raise ValueError(f"regions outside both groups: {sorted(unknown)}")
    return history, candidates


def group(frame: pd.DataFrame) -> np.ndarray:
    """Centre (Montréal, Capitale-Nationale) or Eloignee (Bas-Saint-Laurent, Côte-Nord, Gaspésie)."""

    return np.where(frame[REGION].isin(REMOTE_REGIONS), REMOTE, CENTRE)


def is_remote(frame: pd.DataFrame) -> np.ndarray:
    return (group(frame) == REMOTE).astype(int)


def features(frame: pd.DataFrame, *, with_region: bool = False) -> pd.DataFrame:
    """Model inputs: numeric columns, income in 10 k$, program dummies; optionally the remote indicator.

    Region and postal code are never inputs of the merit score; `with_region` adds the remote indicator
    as a control variable only, so that proxies cannot absorb the committee's regional penalty.
    """

    unknown = set(frame[PROGRAM]) - set(PROGRAMS)
    if unknown:
        raise ValueError(f"programs outside {PROGRAMS}: {sorted(unknown)}")

    out = frame[NUMERIC].astype(float).copy()
    out["revenu_familial_estime"] = out["revenu_familial_estime"] / 1e4
    programs = pd.Categorical(frame[PROGRAM], categories=sorted(PROGRAMS))
    dummies = pd.get_dummies(programs, prefix="prog", drop_first=True).astype(float)
    dummies.index = out.index
    out = pd.concat([out, dummies], axis=1)
    if with_region:
        out["eloignee"] = is_remote(frame)
    return out
