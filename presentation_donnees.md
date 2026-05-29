# Présentation des données — Projet MLOps Home Credit

---

## Contexte

Le dataset provient de la compétition Kaggle **Home Credit Default Risk**.  
**Objectif** : prédire si un client va rembourser son crédit ou faire défaut de paiement.

---

## Vue d'ensemble des fichiers disponibles

| Fichier | Lignes | Colonnes | Rôle |
|---------|--------|----------|------|
| `application_train.csv` | 307 511 | 122 | **Fichier principal** — demandes de crédit avec la cible (TARGET) |
| `application_test.csv` | 48 744 | 121 | Demandes sans TARGET — pour la soumission Kaggle |
| `bureau.csv` | 1 716 428 | 17 | Historique des crédits passés (sources externes — Crédit Bureau) |
| `bureau_balance.csv` | 27 299 925 | 3 | Soldes mensuels des crédits bureau |
| `previous_application.csv` | 1 670 214 | 37 | Demandes précédentes chez Home Credit |
| `POS_CASH_balance.csv` | 10 001 358 | 8 | Soldes mensuels des crédits POS/Cash |
| `installments_payments.csv` | 13 605 401 | 8 | Historique des paiements par mensualité |
| `credit_card_balance.csv` | 3 840 312 | 23 | Soldes mensuels des cartes de crédit |
| `HomeCredit_columns_description.csv` | — | — | Dictionnaire des variables |
| `sample_submission.csv` | 48 744 | 2 | Format attendu pour la soumission |

---

## Fichier central : `application_train.csv`

C'est la table de référence autour de laquelle tout s'articule.

- **307 511 demandeurs de crédit** (clients uniques)
- **122 variables** décrivant chaque client au moment de sa demande
  - 106 variables numériques
  - 16 variables catégorielles
- **Clé de jointure** : `SK_ID_CURR` (identifiant unique par client)

### Variable cible (TARGET)

| Valeur | Signification | Nombre | % |
|--------|--------------|--------|---|
| `0` | Remboursement normal | 282 686 | **91.9%** |
| `1` | Défaut de paiement | 24 825 | **8.1%** |

> Le dataset est **fortement déséquilibré** : seulement 1 client sur 12 est en défaut.  
> Ce déséquilibre devra être pris en compte dans la modélisation.

### Principaux groupes de variables disponibles

| Groupe | Exemples de variables |
|--------|-----------------------|
| **Infos personnelles** | `CODE_GENDER`, `CNT_CHILDREN`, `NAME_FAMILY_STATUS`, `DAYS_BIRTH` |
| **Situation financière** | `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY` |
| **Emploi** | `NAME_INCOME_TYPE`, `OCCUPATION_TYPE`, `DAYS_EMPLOYED` |
| **Logement** | `NAME_HOUSING_TYPE`, `FLAG_OWN_REALTY`, `FLAG_OWN_CAR` |
| **Documents fournis** | `FLAG_DOCUMENT_2` à `FLAG_DOCUMENT_21` |
| **Scores externes** | `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` |
| **Géographie** | `REGION_RATING_CLIENT`, `REGION_POPULATION_RELATIVE` |

---

## Les tables secondaires (données historiques)

Ces tables enrichissent le profil de chaque client avec son **historique financier**.  
Elles se rattachent à `application_train` via `SK_ID_CURR` ou `SK_ID_PREV`.

```
application_train / application_test
         │ SK_ID_CURR
         ├─── bureau.csv
         │         │ SK_ID_BUREAU
         │         └─── bureau_balance.csv
         │
         ├─── previous_application.csv
         │         │ SK_ID_PREV
         │         ├─── POS_CASH_balance.csv
         │         ├─── installments_payments.csv
         │         └─── credit_card_balance.csv
```

### `bureau.csv` — Historique crédit externe
Crédits passés et en cours déclarés au **Crédit Bureau** (autres établissements).  
→ Renseigne sur le comportement de remboursement hors Home Credit.

### `bureau_balance.csv` — Soldes mensuels bureau
Statut mois par mois de chaque crédit bureau (`C`=clos, `X`=inconnu, `0`–`5`=retard).  
→ Permet de reconstituer la trajectoire de défaut dans le temps.

### `previous_application.csv` — Demandes précédentes Home Credit
Toutes les demandes antérieures du client chez Home Credit (accordées ou refusées).  
→ Révèle le profil de risque historique interne.

### `POS_CASH_balance.csv` — Soldes POS/Cash mensuels
Suivi mois par mois des crédits à la consommation et crédits cash précédents.  
→ Informe sur les retards de paiement passés (`SK_DPD`).

### `installments_payments.csv` — Paiements par mensualité
Détail de chaque mensualité : montant prévu vs montant payé, date prévue vs date réelle.  
→ Permet de calculer des indicateurs de ponctualité de paiement.

### `credit_card_balance.csv` — Soldes carte de crédit mensuels
Utilisation mensuelle des cartes de crédit précédentes (solde, limite, retraits).  
→ Révèle les habitudes d'utilisation du crédit revolving.

---

## Ce que nous allons utiliser dans ce projet

### Phase d'exploration et modélisation initiale

Dans un premier temps, nous travaillons principalement sur **`application_train.csv`** :
- C'est le seul fichier contenant la variable `TARGET`
- Il est suffisamment riche (122 variables) pour construire un premier modèle
- Il permet de valider toute la chaîne MLOps sans complexité de jointure

### Feature engineering avancé (phases suivantes)

Les tables secondaires seront **agrégées et jointes** pour enrichir les features :

| Table secondaire | Features extraites (exemples) |
|-----------------|-------------------------------|
| `bureau.csv` | Nombre de crédits actifs, total des dettes en cours |
| `bureau_balance.csv` | Nombre de mois en retard, ratio de statuts défavorables |
| `previous_application.csv` | Taux d'acceptation passé, montant moyen demandé |
| `installments_payments.csv` | Retard moyen de paiement, ratio payé/dû |
| `credit_card_balance.csv` | Taux d'utilisation moyen de la carte |
| `POS_CASH_balance.csv` | Nombre de mensualités restantes moyennes |

### Ce que nous n'utilisons pas directement
- `application_test.csv` : réservé à la prédiction finale (pas de TARGET)
- `sample_submission.csv` : format de soumission Kaggle uniquement

---

## Résumé

> **~57 millions de lignes** de données historiques viennent enrichir  
> **307 511 clients** pour prédire leur risque de défaut de crédit.  
> Le projet démarre sur `application_train.csv` et intégrera progressivement  
> les tables secondaires via du feature engineering par agrégation.
