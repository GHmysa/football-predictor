import numpy as np
import pytest

from src.metrics import brier, devig, log_loss, rps

FORECAST = np.array([[0.5, 0.3, 0.2]])
HOME, DRAW, AWAY = np.array([0]), np.array([1]), np.array([2])


def test_devig_hand_values():
    # 1/7 + 1/4.75 + 1/1.42 = 0.142857 + 0.210526 + 0.704225 = 1.057608
    probs = devig(np.array([[7.0, 4.75, 1.42]]))
    np.testing.assert_allclose(probs, [[0.13508, 0.19906, 0.66587]], atol=1e-5)


def test_devig_rows_sum_to_one():
    odds = np.array([[7.0, 4.75, 1.42], [2.1, 3.4, 3.6]])
    np.testing.assert_allclose(devig(odds).sum(axis=1), [1.0, 1.0])


def test_log_loss_hand_values():
    probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    # (-ln 0.5 - ln 0.7) / 2 = (0.693147 + 0.356675) / 2
    assert log_loss(probs, np.array([0, 2])) == pytest.approx(0.524911, abs=1e-6)


def test_log_loss_zero_probability_on_real_outcome_is_infinite():
    with np.errstate(divide="ignore"):
        assert log_loss(np.array([[0.0, 0.5, 0.5]]), HOME) == np.inf


def test_brier_hand_values():
    assert brier(FORECAST, HOME) == pytest.approx(0.38)  # 0.5² + 0.3² + 0.2² = 0.25 + 0.09 + 0.04
    assert brier(FORECAST, AWAY) == pytest.approx(0.98)  # 0.5² + 0.3² + 0.8² = 0.25 + 0.09 + 0.64


def test_rps_hand_values():
    assert rps(FORECAST, HOME) == pytest.approx(0.145)  # ((0.5-1)² + (0.8-1)²) / 2
    assert rps(FORECAST, DRAW) == pytest.approx(0.145)  # ((0.5-0)² + (0.8-1)²) / 2
    assert rps(FORECAST, AWAY) == pytest.approx(0.445)  # ((0.5-0)² + (0.8-0)²) / 2


def test_rps_penalises_distant_errors_unlike_brier():
    certain_home = np.array([[1.0, 0.0, 0.0]])
    assert brier(certain_home, DRAW) == pytest.approx(brier(certain_home, AWAY))  # 2 dans les deux cas
    assert rps(certain_home, DRAW) == pytest.approx(0.5)
    assert rps(certain_home, AWAY) == pytest.approx(1.0)


def test_perfect_forecast_scores_zero():
    perfect = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    outcomes = np.array([0, 2])
    assert log_loss(perfect, outcomes) == pytest.approx(0.0)
    assert brier(perfect, outcomes) == pytest.approx(0.0)
    assert rps(perfect, outcomes) == pytest.approx(0.0)
