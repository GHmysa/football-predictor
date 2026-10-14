# SPEC — Prédiction de résultats Ligue 1 (Dixon-Coles + backtest vs marché)

Projet personnel, sprint de 2 jours. Objectif : avoir quelque chose de **propre, reproductible et que je sais expliquer ligne par ligne**.

## Question posée

Un Dixon-Coles simple, réentraîné au fil des journées, prédit-il les résultats 1X2 de Ligue 1 mieux qu'un baseline naïf ? Et quel est l'écart avec les cotes de clôture du marché ?

On s'attend à ce que le marché fasse mieux. Le livrable est la **mesure rigoureuse de l'écart**, pas une victoire sur le marché.

## Hors scope (sauf temps restant)

- xG (pas dans football-data.co.uk, à ma connaissance : piste d'amélioration à documenter)
- Ligue 2, autres championnats
- Deep learning, XGBoost
- Dashboard, frontend, cron, déploiement cloud

## Données

- Source : football-data.co.uk, fichiers CSV par saison de Ligue 1 (code `F1`).
- Format d'URL probable : `https://www.football-data.co.uk/mmz4281/<SSAA>/F1.csv` (ex. `2425`). **À vérifier à la main.**
- Saisons visées : ~2017-18 à la saison en cours.
- **À vérifier en ouvrant les CSV** : noms exacts des colonnes (résultat, buts, cotes de clôture), format des dates, noms d'équipes stables d'une saison à l'autre, saison en cours incomplète.
- Cotes : utiliser les cotes de **clôture** si présentes, sinon documenter le choix. Probabilités implicites = inverses des cotes normalisées pour que la somme fasse 1 (retrait de la marge, méthode proportionnelle).
- Les CSV bruts ne sont pas commités (`data/raw/` dans `.gitignore`) ; un script les télécharge.

## Modèle

Dixon-Coles, écrit à la main avec `scipy.optimize` :

- Buts domicile ~ Poisson(λ), λ = exp(attaque_dom + défense_ext + avantage_domicile) ; symétrique pour l'extérieur (convention de signe à fixer et documenter).
- Correction τ de Dixon-Coles sur les scores faibles (0-0, 1-0, 0-1, 1-1), paramètre ρ.
- Pondération temporelle des matchs : poids = exp(−ξ · jours_depuis_le_match). ξ fixé au départ (valeur à justifier), éventuellement ajusté à la fin.
- Contrainte d'identifiabilité (ex. moyenne des attaques = 0).
- Sortie : matrice de scores → probabilités 1X2.

## Évaluation

Backtest **walk-forward** :

- À chaque pas (semaine ou mois, selon le temps de calcul), refit sur les matchs **strictement antérieurs** puis prédiction des matchs du pas suivant.
- Période de chauffe : ignorer les premières journées d'une période pour laisser le modèle s'initialiser (durée à décider et à documenter).
- Équipes promues sans historique : règle simple et documentée (ex. paramètres moyens).

Métriques (sur les mêmes matchs pour tous) :

- Log loss
- RPS (ranked probability score, adapté à l'ordre V/N/D)
- Brier

Comparées pour : (1) fréquences historiques V/N/D de la fenêtre d'entraînement, (2) Dixon-Coles, (3) cotes de clôture du marché.

Calibration : courbe de fiabilité par classe (au minimum victoire domicile).

Incertitude : intervalle de confiance par bootstrap sur la différence de log loss modèle vs marché.

## Règles anti-fuite (non négociables)

- Aucun match n'est utilisé pour sa propre prédiction ni pour une prédiction antérieure.
- Les cotes ne servent **que** de référence, jamais d'entrée du modèle.
- Tri chronologique avant tout calcul. Aucun shuffle.

## Plan

### Jour 1 — Le cœur

| Étape | Livrable | Critère de validation |
|---|---|---|
| J1.1 | `src/data.py` : téléchargement et nettoyage | Je regarde moi-même le DataFrame ; noms d'équipes cohérents ; nb de matchs/saison plausible (~380) |
| J1.2 | `src/dixon_coles.py` : fit + prédiction 1X2 | Probabilités somment à 1 ; paramètres plausibles (avantage domicile > 0) ; test sur un fit simple |
| J1.3 | `src/metrics.py` : log loss, RPS, Brier, de-vig | Tests sur petits exemples calculés à la main |
| J1.4 | `src/backtest.py` : walk-forward | Aucune fuite (test dédié) ; tableau de résultats modèle / fréquences / marché |

### Jour 2 — Le propre

| Étape | Livrable | Critère de validation |
|---|---|---|
| J2.1 | Calibration + bootstrap | Figure sauvegardée ; IC calculé |
| J2.2 | README avec résultats, limites, commande de reproduction | Un tiers peut tout relancer depuis un clone propre |
| J2.3 | `src/api.py` : FastAPI, une route `/predict` | Réponse 1X2 pour deux équipes données |
| J2.4 | Log des prédictions (CSV ou SQLite) + `retrain.py` lançable à la main | Chaque appel API est journalisé |
| J2.5 | Dockerfile (seulement si le reste est fini) | `docker run` répond sur `/predict` |

## Définition de « fini »

- Le backtest tourne avec une seule commande.
- Les résultats sont dans le README, **y compris s'ils sont défavorables**.
- Je peux expliquer : la vraisemblance, τ, la pondération, le walk-forward, pourquoi pas de fuite, ce que mesurent log loss et RPS.
- Ce qui n'est pas fait (xG, réentraînement automatique, etc.) est écrit comme tel dans le README.
