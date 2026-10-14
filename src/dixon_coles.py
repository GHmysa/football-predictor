import numpy as np
from scipy.stats import poisson


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
