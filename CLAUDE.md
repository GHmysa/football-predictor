# CLAUDE.md — football-predictor

Projet personnel de prédiction de résultats de Ligue 1 (Dixon-Coles + backtest walk-forward contre les cotes). Specs dans `SPEC.md` : c'est la référence.

## Ma façon de travailler (important)

- Je suis étudiant ingénieur et je dois **maîtriser chaque ligne** : je dois pouvoir l'expliquer en entretien.
- **Un fichier à la fois.** Avant de l'écrire, dis en 3-5 lignes ce que tu vas faire et pourquoi. Attends mon accord avant de passer au suivant.
- Après chaque étape de `SPEC.md`, résume ce qui a été fait et comment je peux le vérifier moi-même.
- Pas de sur-ingénierie : pas de classes ou d'abstractions sans besoin concret, pas de dépendances inutiles. Si tu hésites entre simple et élégant, choisis simple.
- Reste dans le scope de `SPEC.md`. Si tu penses qu'il faut sortir du scope, propose-le, ne le fais pas.
- Ne jamais inventer : si tu ne sais pas (colonnes d'un CSV, comportement d'une lib), dis-le et vérifie sur les vraies données.
- Challenge mes choix si je me trompe.

## Stack

Python 3.12, pandas, numpy, scipy, matplotlib, pytest, FastAPI (jour 2). Dépendances épinglées dans `requirements.txt`.

## Style de code

- Pas de commentaires qui expliquent le QUOI, seulement le POURQUOI quand ce n'est pas évident.
- Pas de code mort, pas d'imports inutilisés.
- Fonctions courtes, typées quand ça aide, noms explicites.

## ML — règles anti-fuite (non négociables)

- Trier par date avant tout calcul.
- Au pas t, entraîner uniquement sur les matchs strictement antérieurs aux matchs prédits.
- Les cotes ne servent que de référence d'évaluation, jamais d'entrée du modèle.
- Aucun shuffle, aucune validation croisée mélangée.
- Chaque fonction d'évaluation doit avoir un test qui vérifie l'absence de fuite.

## Git

- Commits conventionnels : `feat:`, `fix:`, `refactor:`, `chore:`, `docs:`, `test:`.
- Impératif, sujet ≤ 72 caractères.
- **Pas de ligne `Co-Authored-By` ni de mention de Claude dans les messages de commit ou les descriptions de PR.**
- Je commite moi-même, ou je valide chaque commit avant qu'il soit fait. Propose moi des timings et messages de commit.  
- Ne jamais commiter `data/raw/`, les secrets, ni `.env`.

## À ne pas faire

- Ne pas ajouter de xG, de modèle supplémentaire, de dashboard ou d'automatisation sans que je le demande.
- Ne pas avaler les exceptions silencieusement : logguer avec contexte.
- Ne pas affirmer qu'un résultat est bon sans l'avoir comparé aux baselines.
