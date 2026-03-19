"""
Genproducerer Stage 3 (V4) analysen fra VM2022_messi_mbappe.ipynb:
samme filtre, merges, features, vægte, score og KMeans (k=2) som i notebooken.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Samme konstanter som i notebooken
POSITIONS_KEEP = ["FW", "MF"]
NOTEBOOK_MIN_MINUTES = 270

FEATURE_COLS_V4 = [
    "nonpen_goals_per90",
    "pen_goals_per90",
    "pens_won_per90",
    "assists_per90",
    "xg_assist_per90",
    "npxg_net_per90",
    "games_played",
    "sca_shots_per90",
    "sca_dribbles_per90",
    "sca_fouled_per90",
    "pass_completion_factor_per90",
    "passes_received_per90",
    "progressive_passes_received_per90",
    "weighted_dribbles_completed_per90",
    "miscontrols_per90",
    "dispossessed_per90",
    "touches_mid_3rd_per90",
    "touches_att_3rd_per90",
    "touches_att_pen_area_per90",
    "progressive_passes_per90",
    "passes_into_final_third_per90",
    "passes_into_penalty_area_per90",
    "assisted_shots_per90",
    "shots_on_target_per90",
    "shots_not_on_target_per90",
    "aerials_won_per90",
    "ball_recoveries_per90",
    "tackles_won_per90",
    "interceptions_per90",
    "blocked_passes_per90",
    "fouled_per90",
    "fouls_per90",
    "cards_yellow_per90",
    "cards_red_per90",
    "dribbled_past_per90",
    "offsides_per90",
    "errors_per90",
    "aerials_lost_per90",
    "pens_missed_per90",
]

WEIGHTS_V4 = np.array(
    [
        5.0,
        1.5,
        2.0,
        2.0,
        1.8,
        2.0,
        0.6,
        0.3,
        0.6,
        0.3,
        0.03,
        0.05,
        0.75,
        0.8,
        -0.6,
        -0.9,
        0.01,
        0.1,
        0.15,
        0.2,
        0.15,
        0.2,
        0.4,
        0.8,
        0.5,
        0.25,
        0.25,
        0.35,
        0.35,
        0.20,
        0.5,
        -1.0,
        -1.2,
        -7.0,
        -0.35,
        -0.25,
        -3.0,
        -0.20,
        -3.5,
    ],
    dtype=float,
)

CLUSTER_K_V4 = 2
KMEANS_RANDOM_STATE = 42


def _base_dir(data_dir: Path | None) -> Path:
    return Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent


def load_players_off_v4(
    data_dir: Path | None = None,
    min_minutes: int = NOTEBOOK_MIN_MINUTES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returnerer (players_off, contributions) hvor contributions er feature × vægt pr. række
    (samme som weighted_contributions i notebooken).
    """
    root = _base_dir(data_dir)

    df = pd.read_csv(root / "player_stats.csv")
    positions_keep = POSITIONS_KEEP
    mask_pos = df["position"].isin(positions_keep)
    mask_min = df["minutes"] >= min_minutes
    players_off = df[mask_pos & mask_min].copy()

    gca = pd.read_csv(root / "player_gca.csv")
    passing = pd.read_csv(root / "player_passing.csv")
    misc = pd.read_csv(root / "player_misc.csv")
    shooting = pd.read_csv(root / "player_shooting.csv")
    possession = pd.read_csv(root / "player_possession.csv")
    defense = pd.read_csv(root / "player_defense.csv")

    gca_small = gca[
        ["player", "team", "sca_shots", "sca_dribbles", "sca_fouled"]
    ].copy()
    passing_small = passing[
        [
            "player",
            "team",
            "passes_pct",
            "passes_completed",
            "passes_into_final_third",
            "passes_into_penalty_area",
            "assisted_shots",
            "progressive_passes",
        ]
    ].copy()
    misc_small = misc[
        [
            "player",
            "team",
            "pens_won",
            "aerials_won",
            "ball_recoveries",
            "interceptions",
            "tackles_won",
            "fouled",
            "fouls",
            "offsides",
            "aerials_lost",
        ]
    ].copy()
    shooting_small = shooting[["player", "team", "npxg_net", "shots", "shots_on_target"]].copy()
    possession_small = possession[
        [
            "player",
            "team",
            "touches",
            "passes_received",
            "progressive_passes_received",
            "touches_mid_3rd",
            "touches_att_3rd",
            "touches_att_pen_area",
            "dribbles_completed",
            "dribbles_completed_pct",
            "miscontrols",
            "dispossessed",
        ]
    ].copy()
    defense_small = defense[["player", "team", "dribbled_past", "blocked_passes", "errors"]].copy()

    players_off = players_off.merge(gca_small, on=["player", "team"], how="left")
    players_off = players_off.merge(passing_small, on=["player", "team"], how="left")
    players_off = players_off.merge(misc_small, on=["player", "team"], how="left")
    players_off = players_off.merge(shooting_small, on=["player", "team"], how="left")
    players_off = players_off.merge(possession_small, on=["player", "team"], how="left")
    players_off = players_off.merge(defense_small, on=["player", "team"], how="left")

    m90 = players_off["minutes_90s"]
    players_off["games_played"] = players_off["minutes"] / 90.0
    players_off["nonpen_goals_per90"] = players_off["goals_pens"] / m90
    players_off["pen_goals_per90"] = players_off["pens_made"] / m90
    players_off["pens_won_per90"] = players_off["pens_won"] / m90
    players_off["npxg_net_per90"] = players_off["npxg_net"] / m90

    players_off["pass_completion_factor_per90"] = (
        players_off["passes_completed"] * (players_off["passes_pct"] / 100.0)
    ) / m90

    for col in ["sca_shots", "sca_dribbles", "sca_fouled"]:
        players_off[f"{col}_per90"] = players_off[col] / m90

    players_off["touches_per90"] = players_off["touches"] / m90
    players_off["passes_received_per90"] = players_off["passes_received"] / m90
    players_off["progressive_passes_received_per90"] = (
        players_off["progressive_passes_received"] / m90
    )

    players_off["dribbles_completed_pct_dec"] = players_off["dribbles_completed_pct"] / 100.0
    players_off["weighted_dribbles_completed"] = (
        players_off["dribbles_completed"] * players_off["dribbles_completed_pct_dec"]
    )
    players_off["weighted_dribbles_completed_per90"] = (
        players_off["weighted_dribbles_completed"] / m90
    )

    players_off["miscontrols_per90"] = players_off["miscontrols"] / m90
    players_off["dispossessed_per90"] = players_off["dispossessed"] / m90

    for col in [
        "touches_mid_3rd",
        "touches_att_3rd",
        "touches_att_pen_area",
        "progressive_passes",
        "passes_into_final_third",
        "passes_into_penalty_area",
        "assisted_shots",
        "shots_on_target",
    ]:
        players_off[f"{col}_per90"] = players_off[col] / m90

    players_off["shots_not_on_target"] = players_off["shots"] - players_off["shots_on_target"]
    players_off["shots_not_on_target_per90"] = players_off["shots_not_on_target"] / m90

    for col in [
        "aerials_won",
        "ball_recoveries",
        "tackles_won",
        "interceptions",
        "blocked_passes",
        "fouled",
    ]:
        players_off[f"{col}_per90"] = players_off[col] / m90

    for col in [
        "fouls",
        "cards_yellow",
        "cards_red",
        "dribbled_past",
        "offsides",
        "errors",
        "aerials_lost",
    ]:
        players_off[f"{col}_per90"] = players_off[col] / m90

    players_off["pens_missed_per90"] = (players_off["pens_att"] - players_off["pens_made"]) / m90

    feat_v4 = players_off[FEATURE_COLS_V4].copy().fillna(0.0)
    players_off["offensive_score_v4"] = feat_v4.to_numpy() @ WEIGHTS_V4

    contributions = feat_v4.mul(WEIGHTS_V4, axis=1)
    contributions.index = players_off.index

    scaler_v4 = StandardScaler()
    X_v4 = scaler_v4.fit_transform(feat_v4)
    kmeans_v4 = KMeans(n_clusters=CLUSTER_K_V4, random_state=KMEANS_RANDOM_STATE, n_init=10)
    players_off["cluster_v4"] = kmeans_v4.fit_predict(X_v4)

    return players_off, contributions


def player_rank(score_series: pd.Series, idx: int) -> int:
    return int(score_series.rank(ascending=False).loc[idx])


def find_player_row(players_off: pd.DataFrame, name: str) -> int | None:
    mask = players_off["player"].str.lower() == name.strip().lower()
    if mask.sum() == 0:
        return None
    return players_off[mask].index[0]
