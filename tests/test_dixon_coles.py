import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from src.dixon_coles import fit, outcome_probs, predict, score_matrix, tau, time_weights

XI = np.log(2) / 365
TRUE_ATTACK = {"A": 0.4, "B": 0.2, "C": 0.0, "D": -0.2, "E": -0.4}
TRUE_DEFENCE = {"A": -0.3, "B": -0.1, "C": 0.0, "D": 0.1, "E": 0.3}
TRUE_HOME_ADVANTAGE = 0.25


def simulate_matches(seed: int = 0, rounds: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(rounds):
        for home in TRUE_ATTACK:
            for away in TRUE_ATTACK:
                if home == away:
                    continue
                lam = np.exp(TRUE_ATTACK[home] + TRUE_DEFENCE[away] + TRUE_HOME_ADVANTAGE)
                mu = np.exp(TRUE_ATTACK[away] + TRUE_DEFENCE[home])
                rows.append((pd.Timestamp("2020-01-01") + pd.Timedelta(days=r), home, away,
                             rng.poisson(lam), rng.poisson(mu)))
    return pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])


def test_tau_hand_values():
    # λ=1.5, μ=1.0, ρ=-0.1
    assert tau(0, 0, 1.5, 1.0, -0.1) == pytest.approx(1.15)  # 1 - 1.5*1.0*(-0.1)
    assert tau(0, 1, 1.5, 1.0, -0.1) == pytest.approx(0.85)  # 1 + 1.5*(-0.1)
    assert tau(1, 0, 1.5, 1.0, -0.1) == pytest.approx(0.90)  # 1 + 1.0*(-0.1)
    assert tau(1, 1, 1.5, 1.0, -0.1) == pytest.approx(1.10)  # 1 - (-0.1)
    assert tau(2, 0, 1.5, 1.0, -0.1) == pytest.approx(1.0)


def test_score_matrix_sums_to_one():
    assert score_matrix(1.5, 1.0, -0.1).sum() == pytest.approx(1.0)


def test_rho_zero_is_independent_poisson():
    goals = np.arange(11)
    expected = np.outer(poisson.pmf(goals, 1.5), poisson.pmf(goals, 1.0))
    expected /= expected.sum()
    np.testing.assert_allclose(score_matrix(1.5, 1.0, 0.0), expected)


def test_negative_rho_shifts_mass_to_draws_on_low_scores():
    base = score_matrix(1.5, 1.0, 0.0)
    corrected = score_matrix(1.5, 1.0, -0.1)
    assert corrected[0, 0] > base[0, 0]
    assert corrected[1, 1] > base[1, 1]
    assert corrected[1, 0] < base[1, 0]
    assert corrected[0, 1] < base[0, 1]


def test_outcome_probs_hand_values():
    # lignes = buts domicile, colonnes = buts extérieur
    matrix = np.array([[0.1, 0.2, 0.0],
                       [0.3, 0.1, 0.0],
                       [0.2, 0.1, 0.0]])
    home, draw, away = outcome_probs(matrix)
    assert home == pytest.approx(0.6)  # (1,0) + (2,0) + (2,1) = 0.3 + 0.2 + 0.1
    assert draw == pytest.approx(0.2)  # (0,0) + (1,1) + (2,2) = 0.1 + 0.1 + 0.0
    assert away == pytest.approx(0.2)  # (0,1) + (0,2) + (1,2) = 0.2 + 0.0 + 0.0


def test_stronger_away_team_is_favourite():
    # Exemple Le Havre (dom) - PSG (ext) : λ ≈ 0.64, μ ≈ 2.01
    home, draw, away = outcome_probs(score_matrix(0.64, 2.01, -0.1))
    assert home + draw + away == pytest.approx(1.0)
    assert away > draw > home


def test_time_weights_hand_values():
    dates = pd.Series(pd.to_datetime(["2024-01-01", "2023-01-01"]))
    weights = time_weights(dates, pd.Timestamp("2024-01-01"), XI)
    assert weights[0] == pytest.approx(1.0)  # même jour
    assert weights[1] == pytest.approx(0.5)  # 365 jours = une demi-vie


def test_fit_recovers_simulated_parameters():
    params = fit(simulate_matches(), pd.Timestamp("2021-01-01"), xi=0.0)
    teams = list(TRUE_ATTACK)
    assert sorted(teams, key=params["attack"].get) == sorted(teams, key=TRUE_ATTACK.get)
    assert sorted(teams, key=params["defence"].get) == sorted(teams, key=TRUE_DEFENCE.get)
    assert params["home_advantage"] == pytest.approx(TRUE_HOME_ADVANTAGE, abs=0.1)
    assert params["rho"] == pytest.approx(0.0, abs=0.1)  # données simulées sans corrélation


def test_fit_attacks_are_centered():
    params = fit(simulate_matches(), pd.Timestamp("2021-01-01"), xi=XI)
    assert np.mean(list(params["attack"].values())) == pytest.approx(0.0, abs=1e-9)


def test_fit_ignores_matches_from_ref_date_onwards():
    matches = simulate_matches()
    ref_date = pd.Timestamp("2020-02-15")
    # Un match le jour même et un après, avec des scores absurdes : ils ne doivent rien changer.
    leak = pd.DataFrame({
        "Date": [ref_date, ref_date + pd.Timedelta(days=1)],
        "HomeTeam": ["A", "E"], "AwayTeam": ["E", "A"], "FTHG": [9, 9], "FTAG": [0, 0],
    })
    clean = fit(matches[matches["Date"] < ref_date], ref_date, XI)
    with_future = fit(pd.concat([matches, leak], ignore_index=True), ref_date, XI)
    assert with_future["home_advantage"] == pytest.approx(clean["home_advantage"])
    assert with_future["attack"] == pytest.approx(clean["attack"])


PARAMS = {
    "attack": {"A": 0.5, "B": -0.5},
    "defence": {"A": -0.1, "B": 0.3},
    "home_advantage": 0.25,
    "rho": -0.1,
}


def test_predict_hand_values():
    # λ = exp(0.5 + 0.3 + 0.25), μ = exp(-0.5 + -0.1)
    expected = outcome_probs(score_matrix(np.exp(1.05), np.exp(-0.6), -0.1))
    assert predict(PARAMS, "A", "B") == pytest.approx(expected)
    assert sum(predict(PARAMS, "A", "B")) == pytest.approx(1.0)


def test_predict_unknown_team_gets_average_parameters():
    # Défense moyenne = (-0.1 + 0.3) / 2 = 0.1 ; attaque moyenne = 0.
    explicit = {
        "attack": {**PARAMS["attack"], "Promu": 0.0},
        "defence": {**PARAMS["defence"], "Promu": 0.1},
        "home_advantage": 0.25,
        "rho": -0.1,
    }
    assert predict(PARAMS, "A", "Promu") == pytest.approx(predict(explicit, "A", "Promu"))
    assert predict(PARAMS, "Promu", "B") == pytest.approx(predict(explicit, "Promu", "B"))
