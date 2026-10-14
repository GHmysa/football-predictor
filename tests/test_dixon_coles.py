import numpy as np
import pytest
from scipy.stats import poisson

from src.dixon_coles import outcome_probs, score_matrix, tau


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
