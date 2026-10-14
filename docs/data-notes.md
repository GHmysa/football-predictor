# Notes sur les données

Source : [football-data.co.uk](https://football-data.co.uk), un CSV par saison de Ligue 1 (code `F1`), de 2017-18 à 2026-27 (en cours). Chargés par `src/data.py`.

## URL

`https://football-data.co.uk/mmz4281/<SSAA>/F1.csv` (ex. `2425`).

Avec le préfixe `www.`, le site répond par une redirection 302 vers l'adresse sans `www`. `curl` sans `-L` ne la suit pas et écrit un fichier **vide** sans afficher d'erreur. On utilise donc directement l'URL sans `www`.

## Pièges du format

| Piège | Détail | Traitement dans `data.py` |
|---|---|---|
| BOM | Voir section dédiée ci-dessous. | `encoding="utf-8-sig"` |
| Format de date | `jj/mm/aaaa`, sauf 2017-18 en `jj/mm/aa`. | Format choisi selon la longueur de la chaîne, jamais deviné par pandas (risque d'inverser jour et mois). |
| Colonnes | Le nombre de colonnes varie de 61 à 131 selon la saison. `Time` est absente en 2017-18 et 2018-19, donc `HomeTeam` est en colonne 3 au lieu de 4. | Sélection par **nom** de colonne, jamais par position. |
| Cotes absentes | Pas de `PSC*` en 2026-27, `PSC*` remplies à moitié en 2025-26. | Colonnes complétées par `NaN` avec un avertissement logué. Les matchs sont gardés (le modèle s'entraîne dessus). |
| Saison tronquée | 2019-20 : 279 matchs (arrêtée par le Covid). 2026-27 : en cours. | Aucun traitement, c'est normal. |

Nombre de matchs : 380 par saison jusqu'en 2022-23 (20 clubs), 306 ensuite (18 clubs). Total chargé : 3 151.

Les noms d'équipes sont stables d'une saison à l'autre (30 noms distincts, pas de variante du type `PSG` / `Paris SG`).

### Le BOM

Le BOM (*byte order mark*) est un caractère invisible (`U+FEFF`, octets `EF BB BF`) que certains outils placent au tout début d'un fichier UTF-8. Lu avec l'encodage `utf-8` classique, il reste collé au premier mot : la première colonne s'appelle alors `﻿Div` et non `Div`.

Comment il a été repéré :

1. En listant les colonnes du CSV de 2024-25 (`head -1 ... | tr ',' '\n' | cat -n`), le premier nom sortait avec un caractère invisible devant `Div` (`﻿Div`), alors que les autres étaient normaux.
2. La commande `file data/raw/F1_2425.csv` a confirmé : `CSV Unicode text, UTF-8 (with BOM) text`.

Conséquence si on l'ignore : `df["Div"]` lève une `KeyError`, et `usecols=[...]` échoue avec « columns expected but not found ». Avec `utf-8-sig`, pandas retire le BOM à la lecture. (`Div` n'est pas utilisée ensuite, mais le problème toucherait n'importe quelle première colonne.)

## Cotes : ouverture, clôture, marge

Une cote de `1.42` signifie qu'une mise de 1 € rapporte 1,42 € au total en cas de gain. La probabilité implicite est `1 / cote` (ici ≈ 70 %). Chaque bookmaker donne trois cotes 1X2 : `H` (victoire domicile), `D` (nul), `A` (victoire extérieur).

- **Ouverture** (colonnes sans `C` : `PSH`, `B365H`…) : cote publiée plusieurs jours avant le match.
- **Clôture** (colonnes avec `C` : `PSCH`, `B365CH`…) : cote juste avant le coup d'envoi. Elle intègre compositions, blessures et mises du marché. C'est la référence la plus exigeante, et celle du SPEC.

### La marge (*overround*)

La somme des `1/cote` sur H, D, A dépasse toujours 1. L'excédent est la commission du bookmaker. Exemple, Le Havre – PSG, cotes d'ouverture Bet365 (`B365H/D/A`) `7 / 4.75 / 1.42` :

`1/7 + 1/4.75 + 1/1.42 = 0.143 + 0.211 + 0.704 = 1.058` → marge de 5,8 %.

Pour obtenir de vraies probabilités, on **normalise** (méthode proportionnelle du SPEC) : chaque `1/cote` est divisé par la somme, pour que les trois totalisent 1. Ce sera dans `metrics.py` (J1.3).

### Pourquoi Pinnacle

Marge moyenne de clôture mesurée sur 2019-20 à 2024-25 :

| Bookmaker | Marge moyenne |
|---|---|
| **Pinnacle (`PSC`)** | **2,7 %** |
| Bwin (`BWC`) | 5,5 % |
| Bet365 (`B365C`) | 5,7 % |
| William Hill (`WHC`) | 6,5 % |
| Moyenne des books (`AvgC`) | 4,8 % |

(`1XBC` et `BFC` n'existent que sur une saison, `MaxC` est la meilleure cote par issue chez des bookmakers différents, avec une marge de −0,6 % : ce n'est pas une prévision utilisable.)

Pinnacle prend deux fois moins de commission que les bookmakers grand public : sa normalisation déforme moins les probabilités. **Ce que les données montrent** : qu'il est moins cher. **Ce qu'elles ne montrent pas** : qu'il est plus précis. C'est la réputation du marché Pinnacle dans la littérature ; on ne l'a pas mesuré ici. Pinnacle reste un proxy public, pas « la vérité ».

Couverture de `PSC*` : complète de 2017-18 à 2024-25 (2 735 matchs), 153 / 306 en 2025-26, aucune en 2026-27. Les cotes ne servent qu'à l'évaluation, jamais à l'entraînement : le modèle utilise les 10 saisons, seule la comparaison au marché se limite aux matchs avec cote, les mêmes pour les trois méthodes.

## Exploration (3 151 matchs)

- Fréquences globales : victoire domicile 43,4 %, nul 25,3 %, victoire extérieur 31,3 % (baseline n°1).
- Buts moyens : 1,51 à domicile, 1,23 à l'extérieur. Variances : 1,66 et 1,33, soit une légère sur-dispersion par rapport à Poisson.
- La distribution des buts est proche d'un Poisson de même moyenne (0 but à domicile : 23,6 % observé, 22,0 % Poisson).
- Victoire domicile par saison : de 37,4 % (2020-21) à 48,0 % (2019-20). La baisse de 2020-21 coïncide avec les huis clos du Covid, ce qui est une explication plausible **non vérifiée** ici.
- Scores faibles, observé / indépendance de deux Poisson de moyenne globale : 0-0 → 1,08 ; 1-0 → 0,97 ; 0-1 → 0,99 ; 1-1 → 0,99. Écart faible, mais ce test porte sur la ligue entière et non sur des équipes données : il ne permet pas de conclure que la correction τ de Dixon-Coles est inutile. À mesurer avec ρ au J1.2.
