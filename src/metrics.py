import numpy as np

# Convention : probs de forme (n, 3) dans l'ordre domicile, nul, extérieur ;
# outcomes = entiers 0 (domicile), 1 (nul), 2 (extérieur). L'ordre compte pour le RPS.


def devig(odds: np.ndarray) -> np.ndarray:
    """Retire la marge du bookmaker (méthode proportionnelle) : cotes (n, 3) -> probas (n, 3)."""
    inverse = 1 / odds
    return inverse / inverse.sum(axis=1, keepdims=True)


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    # Pas d'écrêtage : une proba nulle sur l'issue réelle doit donner inf, pas être masquée.
    return float(-np.mean(np.log(probs[np.arange(len(probs)), outcomes])))


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    observed = np.eye(probs.shape[1])[outcomes]
    return float(np.mean(np.sum((probs - observed) ** 2, axis=1)))


def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Ranked probability score : compare les probas cumulées, donc punit plus une erreur éloignée."""
    observed = np.eye(probs.shape[1])[outcomes]
    # La dernière cumulée vaut toujours 1 des deux côtés : on l'ignore.
    gap = np.cumsum(probs, axis=1)[:, :-1] - np.cumsum(observed, axis=1)[:, :-1]
    return float(np.mean(np.sum(gap**2, axis=1)) / (probs.shape[1] - 1))
