import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

logger = logging.getLogger(__name__)

# Garde τ > 0 pour des λ et μ réalistes (τ(0,0) = 1 - λμρ, τ(0,1) = 1 + λρ).
RHO_BOUNDS = (-0.2, 0.2)


def tau(x, y, lam, mu, rho):
    """Facteur de correction de Dixon-Coles ; x = buts domicile, y = buts extérieur."""
    return np.select(
        [(x == 0) & (y == 0), (x == 0) & (y == 1), (x == 1) & (y == 0), (x == 1) & (y == 1)],
        [1 - lam * mu * rho, 1 + lam * rho, 1 + mu * rho, 1 - rho],
        default=1.0,
    )


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = 10) -> np.ndarray:
    """Probabilités de chaque score : lignes = buts domicile, colonnes = buts extérieur."""
    goals = np.arange(max_goals + 1)
    x, y = np.meshgrid(goals, goals, indexing="ij")
    matrix = poisson.pmf(x, lam) * poisson.pmf(y, mu) * tau(x, y, lam, mu, rho)
    # On tronque à max_goals et τ modifie les masses : on renormalise pour sommer à 1.
    return matrix / matrix.sum()


def outcome_probs(matrix: np.ndarray) -> tuple[float, float, float]:
    """Probabilités (victoire domicile, nul, victoire extérieur)."""
    home = np.tril(matrix, -1).sum()
    draw = np.trace(matrix)
    away = np.triu(matrix, 1).sum()
    return home, draw, away


def time_weights(dates: pd.Series, ref_date: pd.Timestamp, xi: float) -> np.ndarray:
    days = (ref_date - dates).dt.days.to_numpy()
    return np.exp(-xi * days)


def neg_log_likelihood(params, home_idx, away_idx, home_goals, away_goals, weights, n_teams):
    attack = params[:n_teams]
    # Contrainte d'identifiabilité : on recentre ici plutôt que de la passer à scipy.
    attack = attack - attack.mean()
    defence = params[n_teams : 2 * n_teams]
    home_advantage, rho = params[2 * n_teams], params[2 * n_teams + 1]
    lam = np.exp(attack[home_idx] + defence[away_idx] + home_advantage)
    mu = np.exp(attack[away_idx] + defence[home_idx])
    correction = tau(home_goals, away_goals, lam, mu, rho)
    # Plancher : l'optimiseur peut tester des valeurs où τ <= 0 et log donnerait NaN.
    log_p = (
        poisson.logpmf(home_goals, lam)
        + poisson.logpmf(away_goals, mu)
        + np.log(np.maximum(correction, 1e-10))
    )
    return -np.sum(weights * log_p)


def fit(matches: pd.DataFrame, ref_date: pd.Timestamp, xi: float) -> dict:
    """Ajuste le modèle sur les matchs strictement antérieurs à ref_date."""
    train = matches[matches["Date"] < ref_date]
    teams = sorted(set(train["HomeTeam"]) | set(train["AwayTeam"]))
    index = {team: i for i, team in enumerate(teams)}
    n = len(teams)
    args = (
        train["HomeTeam"].map(index).to_numpy(),
        train["AwayTeam"].map(index).to_numpy(),
        train["FTHG"].to_numpy(),
        train["FTAG"].to_numpy(),
        time_weights(train["Date"], ref_date, xi),
        n,
    )
    bounds = [(None, None)] * (2 * n + 1) + [RHO_BOUNDS]
    result = minimize(neg_log_likelihood, np.zeros(2 * n + 2), args=args, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        logger.warning("Fit non convergé (ref_date=%s) : %s", ref_date, result.message)
    attack = result.x[:n]
    return {
        "attack": dict(zip(teams, attack - attack.mean())),
        "defence": dict(zip(teams, result.x[n : 2 * n])),
        "home_advantage": float(result.x[2 * n]),
        "rho": float(result.x[2 * n + 1]),
    }


def predict(params: dict, home_team: str, away_team: str) -> tuple[float, float, float]:
    attack, defence = params["attack"], params["defence"]
    # Équipe sans historique (promue) : paramètres moyens. Les attaques sont centrées sur 0,
    # mais pas les défenses (leur moyenne porte le niveau de buts de la ligue).
    mean_defence = np.mean(list(defence.values()))
    lam = np.exp(
        attack.get(home_team, 0.0) + defence.get(away_team, mean_defence) + params["home_advantage"]
    )
    mu = np.exp(attack.get(away_team, 0.0) + defence.get(home_team, mean_defence))
    return outcome_probs(score_matrix(lam, mu, params["rho"]))
