# ÉquiAlgo — audit et correction du biais régional

Réponse au défi ÉquiAlgo (IVADO / CodeML 2026). Le modèle en production accorde une bourse à 48,4 % des
candidat·es des grands centres et à 27,3 % de celles et ceux des régions éloignées. L'écart de 21 points
n'est pas expliqué par les dossiers : le comité historique retire environ 2,2 en logit aux demandes des
trois régions éloignées, ce qui divise leurs chances par neuf à dossier identique. La correction retire
ce terme et classe les demandes sur la cote R et les heures travaillées, sans région ni code postal.

Résultat mesuré : l'écart d'égalité des chances passe de 0,283 à 0,045 contre le même étalon, avec un
taux d'octroi de 0,410 (0,409 dans les grands centres, 0,411 dans les régions éloignées).

## Livrables

| Fichier | Contenu |
|---|---|
| `predictions.csv` | décisions pour les 4 000 candidat·es d'évaluation, taux d'octroi 0,410 |
| `audit_rapport.ipynb` | l'audit : mesure de l'écart, règle du comité, décomposition, proxys, métrique retenue, surveillance, gouvernance |
| `model_corrige.py` | l'atténuation : balayage des trois familles de contraintes, front de Pareto, écriture de `predictions.csv` |
| `presentation.pdf` | le support du pitch de cinq minutes, douze pages en 16:9 (source : `presentation.html`) |

## Reproduire

```bash
python -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook audit_rapport.ipynb   # l'audit, de bout en bout
python model_corrige.py                # le balayage, les figures, predictions.csv
python -m pytest -q tests              # les fonctions de mesure sur des cas vérifiés à la main
```

`presentation.pdf` se régénère à partir de `presentation.html`, qui embarque ses polices et ne
dépend d'aucun réseau :

```bash
chrome --headless=new --no-pdf-header-footer --print-to-pdf=presentation.pdf presentation.html
```

Python 3.10 ou plus récent. `model_corrige.py` prend quelques minutes : il valide 31 règles de décision
sur 5 plis croisés, dont sept réentraînements sous contrainte avec `ExponentiatedGradient`.

## Organisation

| Dossier | Contenu |
|---|---|
| `equialgo/` | la bibliothèque : chargement, métriques, score de mérite, étalons de substitution, mesures d'audit |
| `data/` | les deux fichiers fournis par les organisateurs |
| `figures/` | les fronts de Pareto écrits par `model_corrige.py` |
| `results/` | les mesures écrites par `model_corrige.py` (plis croisés, ensemble d'évaluation, points estimés) |
| `tests/` | les tests des fonctions de mesure |

Dans `equialgo/` :

- `data.py` — chargement des deux fichiers, définition des deux groupes régionaux, matrice des variables
  explicatives. La région et le code postal ne sont jamais des entrées du score de mérite.
- `metrics.py` — chaque quantité notée, écrite explicitement : taux de sélection, parité démographique,
  taux de vrais positifs, écart d'égalité des chances, part de l'écart refermée, utilité mise à l'échelle,
  points estimés.
- `fair_score.py` — la règle du comité ajustée avec un indicateur « région éloignée » comme variable de
  contrôle, puis évaluée avec ce coefficient mis à zéro ; l'allocation sous enveloppe fixe.
- `references.py` — douze versions plausibles de l'étalon caché, et le poids de chacune, mesuré par sa
  capacité à reproduire le seul chiffre publié à son sujet (l'écart de 0,270 du modèle de base).
- `audit.py` — la fuite d'information régionale par variable, et la décomposition de l'écart du comité
  entre une part « dossier » et une part « pénalité ».

## Choix de la métrique

L'égalité des chances (parité des taux de vrais positifs) est la contrainte retenue ; la parité
démographique est mesurée et rapportée, mais pas imposée. Les deux ne peuvent pas tenir ensemble quand
les profils des deux groupes diffèrent, et c'est bien le cas ici : la cote R moyenne vaut 27,3 en région
éloignée contre 28,0 dans les grands centres. Imposer la parité démographique obligerait à accorder le
même taux de bourses malgré cette différence ; l'égalité des chances demande seulement qu'une candidate
méritante ait la même chance d'être financée, où qu'elle habite. Le carnet d'audit développe l'argument,
y compris le rejet de l'égalité des chances généralisée (*equalized odds*).

`decision_octroi` n'est jamais la cible. C'est la trace de ce que le comité a fait, et c'est le comité
qui est audité : une contrainte d'équité calculée sur cette colonne est déjà satisfaite par le comité
lui-même, ce qui explique pourquoi `ThresholdOptimizer` et `ExponentiatedGradient` plafonnent autour de
0,20 dans le front de Pareto. Les règles sont donc évaluées contre douze étalons de substitution
(`equialgo/references.py`), pondérés par leur plausibilité.

## Données

Toutes les données sont synthétiques et l'institution est fictive, comme l'indiquent les consignes du
défi. `data/` reproduit les deux fichiers fournis, sans modification.
