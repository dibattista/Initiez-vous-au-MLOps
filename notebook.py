# # Étape 1 — Exploration et préparation des données
# 
# Pipeline MLOps de scoring de risque de crédit — dataset Home Credit.  
# Ce notebook couvre les phases 1 à 4 : exploration, nettoyage, feature engineering et préparation pour le modèle.

# ## Phase 1 — Découverte des données
# 
# ---
# 
# ### 1.1 Chargement des fichiers
# 
# **Objectif** : Charger tous les fichiers CSV et en faire un premier aperçu.

import pandas as pd
pd.set_option('display.max_columns', None)


DATA_PATH = '/home/bdb/Bureau/P6/Initiez-vous-au-MLOps/data/'

# #### application_train.csv

application_train = pd.read_csv(DATA_PATH + 'application_train.csv')
print(f'Shape: {application_train.shape}')
application_train.info()

application_train

# #### application_test.csv

application_test = pd.read_csv(DATA_PATH + 'application_test.csv')
print(f'Shape: {application_test.shape}')
application_test.info()

# #### bureau.csv

bureau = pd.read_csv(DATA_PATH + 'bureau.csv')
print(f'Shape: {bureau.shape}')
bureau.info()

# #### bureau_balance.csv

bureau_balance = pd.read_csv(DATA_PATH + 'bureau_balance.csv')
print(f'Shape: {bureau_balance.shape}')
bureau_balance.info()

# #### credit_card_balance.csv

credit_card_balance = pd.read_csv(DATA_PATH + 'credit_card_balance.csv')
print(f'Shape: {credit_card_balance.shape}')
credit_card_balance.info()

# #### installments_payments.csv

installments_payments = pd.read_csv(DATA_PATH + 'installments_payments.csv')
print(f'Shape: {installments_payments.shape}')
installments_payments.info()

# #### POS_CASH_balance.csv

pos_cash_balance = pd.read_csv(DATA_PATH + 'POS_CASH_balance.csv')
print(f'Shape: {pos_cash_balance.shape}')
pos_cash_balance.info()

# #### previous_application.csv

previous_application = pd.read_csv(DATA_PATH + 'previous_application.csv')
print(f'Shape: {previous_application.shape}')
previous_application.info()

# #### HomeCredit_columns_description.csv

columns_description = pd.read_csv(DATA_PATH + 'HomeCredit_columns_description.csv', encoding='latin-1')
print(f'Shape: {columns_description.shape}')
columns_description.info()

# Voir la description complète de TARGET dans le dictionnaire
print(columns_description[columns_description['Row'] == 'TARGET']['Description'].values[0])

# #### sample_submission.csv

sample_submission = pd.read_csv(DATA_PATH + 'sample_submission.csv')
print(f'Shape: {sample_submission.shape}')
sample_submission.info()

application_train.head()

# Toutes les descriptions des colonnes de application_train
app_desc = columns_description[
    columns_description['Table'] == 'application_{train|test}.csv'
][['Row', 'Description']]

print(f"Nombre de colonnes décrites : {len(app_desc)}")
display(app_desc)

# ### Aperçu rapide — Distribution de TARGET
# 
# Premier regard sur l'équilibre de la variable cible. L'analyse visuelle complète, avec graphique et implications, est dans la section **1.3**.

# Distribution de la variable cible TARGET
print("Distribution TARGET :")
print(application_train['TARGET'].value_counts())
print(f"\nDéséquilibre : {application_train['TARGET'].mean():.1%} de mauvais clients")

# ### Aperçu rapide — Vue d'ensemble des valeurs manquantes
# 
# Visualisation globale des données manquantes sur `application_train`. L'analyse détaillée colonne par colonne est dans la section **1.4**.

import missingno as msno
import matplotlib.pyplot as plt

# Vue d'ensemble visuelle des valeurs manquantes
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

plt.subplot(1, 2, 1)
msno.bar(application_train, ax=plt.gca(), color='steelblue')
plt.title('% de valeurs présentes par colonne')

plt.subplot(1, 2, 2)
msno.matrix(application_train, ax=plt.gca())
plt.title('Patterns de valeurs manquantes')

plt.tight_layout()
plt.show()

# ---
# 
# ## Résumé — 1.1 Chargement des fichiers
# 
# | Fichier | Lignes | Colonnes | Rôle |
# |---|---|---|---|
# | **application_train** | 307 511 | 122 | Données d'entraînement avec TARGET |
# | **application_test** | 48 744 | 121 | Données de soumission Kaggle (sans TARGET) |
# | **bureau** | 1 716 428 | 17 | Historique crédits bureau |
# | **bureau_balance** | 27 299 925 | 3 | Soldes mensuels crédits bureau |
# | **credit_card_balance** | 3 840 312 | 23 | Soldes cartes de crédit |
# | **installments_payments** | 13 605 401 | 8 | Paiements d'échéances |
# | **POS_CASH_balance** | 10 001 358 | 8 | Soldes POS et cash |
# | **previous_application** | 1 670 214 | 37 | Demandes précédentes de crédit |
# | **sample_submission** | 48 744 | 2 | Format de soumission Kaggle |
# 
# **Décisions clés :**
# - Seuls `application_train` et `application_test` servent respectivement à l'entraînement et à la prédiction.
# - Les tables secondaires seront agrégées par `SK_ID_CURR` en **Phase 3**.
# - `DAYS_EMPLOYED == 365 243` est une valeur sentinelle ("sans emploi") — à remplacer en **Phase 2**.

# ---
# 
# ## Phase 1 — 1.2 Forme et types des données
# 
# ### Objectif
# Comparer `application_train` et `application_test` pour comprendre leurs différences structurelles, et expliquer pourquoi `application_test` **ne peut pas** être utilisé pour l'entraînement d'un modèle.

# ### Étape 1 — Dimensions de chaque dataset
# 
# On regarde combien de lignes (clients) et de colonnes (variables) contient chaque fichier.

# Dimensions de chaque dataset
print("=" * 50)
print("DIMENSIONS DES DATASETS")
print("=" * 50)
print(f"application_train : {application_train.shape[0]:>7} lignes × {application_train.shape[1]} colonnes")
print(f"application_test  : {application_test.shape[0]:>7} lignes × {application_test.shape[1]} colonnes")
print("=" * 50)

# ### Étape 2 — Types de colonnes (int, float, object)
# 
# `.dtypes.value_counts()` compte combien de colonnes appartiennent à chaque type : numérique entier (`int64`), numérique décimal (`float64`), ou texte (`object`).

# Types de colonnes par dataset
print("=" * 50)
print("TYPES DE COLONNES — application_train")
print("=" * 50)
print(application_train.dtypes.value_counts().to_string())

print("\n" + "=" * 50)
print("TYPES DE COLONNES — application_test")
print("=" * 50)
print(application_test.dtypes.value_counts().to_string())

# ### Étape 3 — Comparaison des colonnes : train vs test
# 
# On cherche les colonnes présentes dans un dataset mais absentes de l'autre.
# Un décalage révèle une information que l'un possède et l'autre non.

# Colonnes présentes dans train mais absentes dans test
cols_train_only = set(application_train.columns) - set(application_test.columns)

# Colonnes présentes dans test mais absentes dans train
cols_test_only = set(application_test.columns) - set(application_train.columns)

print("=" * 50)
print("COLONNES DANS TRAIN mais PAS dans TEST")
print("=" * 50)
print(cols_train_only if cols_train_only else "  Aucune")

print("\n" + "=" * 50)
print("COLONNES DANS TEST mais PAS dans TRAIN")
print("=" * 50)
print(cols_test_only if cols_test_only else "  Aucune")

# ### Étape 4 — Vérification explicite de TARGET dans chaque dataset
# 
# `TARGET` est la variable que le modèle doit apprendre à prédire.
# Sans elle, impossible d'entraîner : on ne saurait pas quoi prédire !

# Vérification de la présence de TARGET dans chaque dataset
def check_target(df, name):
    if 'TARGET' in df.columns:
        n_vals = df['TARGET'].notna().sum()
        print(f"  ✓ TARGET présente dans {name} — {n_vals} valeurs non-nulles")
    else:
        print(f"  ✗ TARGET ABSENTE dans {name}")

print("=" * 50)
print("PRÉSENCE DE LA VARIABLE TARGET")
print("=" * 50)
check_target(application_train, "application_train")
check_target(application_test,  "application_test")
print("=" * 50)

# ---
# 
# ## Résumé — 1.2 Forme et types des données
# 
# | Caractéristique | `application_train` | `application_test` |
# |---|---|---|
# | Lignes (clients) | 307 511 | 48 744 |
# | Colonnes | 122 | 121 |
# | float64 | 65 | 65 |
# | int64 | 41 | 40 |
# | object (texte) | 16 | 16 |
# | Colonne TARGET | **OUI** ✓ | **NON** ✗ |
# 
# ### Pourquoi application_test est inutilisable pour l'entraînement ?
# 
# `application_test` est un fichier de **compétition Kaggle** : il contient les données de clients dont on veut prédire le risque, mais **sans la réponse** (`TARGET` absente).
# 
# Un algorithme de machine learning supervisé apprend en observant des exemples **avec** leur étiquette correcte (`TARGET = 0` ou `1`). Sans cette étiquette, le modèle n'a rien à apprendre.
# 
# **Conséquence pratique :**
# - `application_train` → sert à **entraîner et évaluer** le modèle (on connaît le résultat réel)
# - `application_test` → sert uniquement à **produire des prédictions** à soumettre sur Kaggle (on ne connaît pas le résultat réel)
# 
# > La seule différence structurelle entre les deux fichiers est l'absence de `TARGET` dans le test : **1 colonne manquante**, mais c'est précisément celle qui rend l'entraînement possible.

# ---
# 
# ## Phase 1 — 1.3 La variable cible (TARGET)
# 
# ### Objectif
# Comprendre la **distribution de TARGET** : combien de bons clients (0) vs mauvais clients (1), et ce que ce déséquilibre implique pour le modèle.

# ### Étape 1 — Distribution de TARGET en chiffres
# 
# On compte le nombre de clients par classe et on calcule les pourcentages.

# Distribution de TARGET : effectifs et pourcentages
counts = application_train['TARGET'].value_counts().sort_index()
pcts   = application_train['TARGET'].value_counts(normalize=True).sort_index() * 100

print("=" * 50)
print("DISTRIBUTION DE LA VARIABLE TARGET")
print("=" * 50)
print(f"  0 — Bon client    : {counts[0]:>7}  ({pcts[0]:.1f} %)")
print(f"  1 — Mauvais client: {counts[1]:>7}  ({pcts[1]:.1f} %)")
print("-" * 50)
print(f"  Total             : {counts.sum():>7}  (100.0 %)")
print("=" * 50)
print(f"\n  Déséquilibre : 1 mauvais client pour {counts[0] // counts[1]:.0f} bons clients")

# ### Étape 2 — Visualisation : graphique en barres
# 
# Un graphique permet de rendre le déséquilibre immédiatement visible à l'œil.

import matplotlib.pyplot as plt
import os

fig, ax = plt.subplots(figsize=(7, 5))

labels = ['0 — Bon client', '1 — Mauvais client']
colors = ['#2ecc71', '#e74c3c']  # vert, rouge

bars = ax.bar(labels, counts.values, color=colors, edgecolor='white', width=0.5)

# Valeur exacte + pourcentage au-dessus de chaque barre
for bar, count, pct in zip(bars, counts.values, pcts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 3000,
        f"{count:,}\n({pct:.1f} %)",
        ha='center', va='bottom', fontsize=11, fontweight='bold'
    )

ax.set_title("Distribution de la variable TARGET", fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel("Nombre de clients", fontsize=11)
ax.set_ylim(0, counts.max() * 1.18)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()

# Sauvegarde dans graphiques/
GRAPH_PATH = os.path.join(os.path.dirname('/home/bdb/Bureau/P6/Initiez-vous-au-MLOps/notebook.ipynb'), 'graphiques')
plt.savefig(os.path.join(GRAPH_PATH, 'target_distribution.png'), dpi=150, bbox_inches='tight')
print(f"Graphique sauvegardé : graphiques/target_distribution.png")

plt.show()

# ---
# 
# ## Résumé — 1.3 La variable cible (TARGET)
# 
# | Classe | Signification | Effectif | Proportion |
# |---|---|---|---|
# | **0** | Bon client (remboursement OK) | 282 686 | 91,9 % |
# | **1** | Mauvais client (difficultés de paiement) | 24 825 | 8,1 % |
# 
# **Ratio de déséquilibre :** environ 1 mauvais client pour 11 bons clients.
# 
# **Implication pour la modélisation :** ce fort déséquilibre (8 % vs 92 %) signifie qu'un modèle naïf qui prédit toujours 0 aurait déjà 91,9 % de précision — il faudra utiliser des techniques comme le rééchantillonnage (SMOTE, under-sampling) ou ajuster les poids des classes pour que le modèle apprenne réellement à détecter les mauvais payeurs.

# ---
# 
# ## Phase 1 — 1.4 Valeurs manquantes
# 
# ### Objectif
# Identifier les colonnes incomplètes dans `application_train` et visualiser leur taux de remplissage — en filtrant sur les colonnes concernées pour que le graphique reste lisible.

# ### Étape 1 — Identifier les colonnes avec des valeurs manquantes
# 
# On filtre sur les colonnes qui ont au moins 1 NaN, puis on affiche combien sont concernées.

# Colonnes avec au moins 1 valeur manquante
mask_nan = application_train.isnull().sum() > 0
cols_manquantes = application_train.columns[mask_nan].tolist()

total_cols = application_train.shape[1]
print("=" * 50)
print("VALEURS MANQUANTES — application_train")
print("=" * 50)
print(f"  Colonnes avec NaN : {len(cols_manquantes)} sur {total_cols}")
print(f"  Colonnes complètes: {total_cols - len(cols_manquantes)} sur {total_cols}")
print("=" * 50)

# ### Étape 2 — Tableau des taux de valeurs manquantes
# 
# Classement du plus incomplet au plus complet, avec le nombre brut et le pourcentage.

# Tableau classé par taux de NaN décroissant
n_rows = application_train.shape[0]
nan_counts = application_train[cols_manquantes].isnull().sum()

tableau_nan = pd.DataFrame({
    'nb_manquants': nan_counts,
    'pct_manquants': (nan_counts / n_rows * 100).round(1)
}).sort_values('pct_manquants', ascending=False)

print(f"{'Colonne':<45} {'NaN':>7} {'%':>7}")
print("-" * 62)
for col, row in tableau_nan.iterrows():
    print(f"{col:<45} {int(row['nb_manquants']):>7,} {row['pct_manquants']:>6.1f}%")

# ### Étape 3 — Visualisation missingno par groupes de 10 colonnes
# 
# 67 colonnes dans un seul graphique missingno = illisible. Solution : on découpe `cols_manquantes` en **groupes de 10** et on génère un graphique par groupe. Chaque graphique est propre, les noms de colonnes restent lisibles.

import missingno as msno
import matplotlib.pyplot as plt
import math

# Colonnes triées du plus incomplet au plus complet
cols_tries = tableau_nan.index.tolist()

TAILLE_GROUPE = 10
nb_groupes = math.ceil(len(cols_tries) / TAILLE_GROUPE)

for i in range(nb_groupes):
    groupe = cols_tries[i * TAILLE_GROUPE : (i + 1) * TAILLE_GROUPE]
    fig, ax = plt.subplots(figsize=(12, 5))
    msno.bar(application_train[groupe], ax=ax, color='steelblue', fontsize=10)
    ax.set_title(f"Valeurs manquantes — groupe {i+1}/{nb_groupes}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    nom_fichier = f"graphiques/manquantes_groupe_{i+1}.png"
    plt.savefig(nom_fichier, dpi=150, bbox_inches='tight')
    print(f"Sauvegardé : {nom_fichier}")
    plt.show()

print(f"\n{'='*50}\n{nb_groupes} graphiques générés pour {len(cols_tries)} colonnes\n{'='*50}")

# ### Étape 4 — Top 10 des colonnes les plus incomplètes
# 
# Récapitulatif chiffré des colonnes les plus problématiques.

# Top 10 colonnes les plus incomplètes
top10 = tableau_nan.head(10)

print("=" * 55)
print("TOP 10 — COLONNES LES PLUS INCOMPLÈTES")
print("=" * 55)
print(f"  {'Rang':<5} {'Colonne':<40} {'% NaN':>6}")
print("-" * 55)
for rang, (col, row) in enumerate(top10.iterrows(), start=1):
    print(f"  {rang:<5} {col:<40} {row['pct_manquants']:>5.1f}%")
print("=" * 55)

# Répartition par seuil de NaN
sup_40   = (tableau_nan['pct_manquants'] > 40).sum()
entre    = ((tableau_nan['pct_manquants'] >= 10) & (tableau_nan['pct_manquants'] <= 40)).sum()
inf_10   = (tableau_nan['pct_manquants'] < 10).sum()

print("=" * 50)
print("RÉPARTITION PAR SEUIL DE NaN")
print("=" * 50)
print(f"  > 40 % de NaN  : {sup_40:>3} colonnes  → candidats à la suppression")
print(f"  10 % à 40 %    : {entre:>3} colonnes  → imputation à envisager")
print(f"  < 10 % de NaN  : {inf_10:>3} colonnes  → imputation simple")
print("=" * 50)

# ---
# 
# ## Résumé — 1.4 Valeurs manquantes
# 
# **67 colonnes sur 122** ont au moins une valeur manquante.
# 
# **Répartition par seuil :**
# 
# | Seuil | Nb colonnes | Stratégie Phase 2 |
# |---|---|---|
# | **> 40 % de NaN** | 49 | Suppression — trop peu de données pour imputer |
# | **10 % à 40 %** | 8 | Imputation (médiane pour numériques, mode pour catégorielles) |
# | **< 10 % de NaN** | 10 | Imputation simple — impact faible |
# 
# > Les colonnes à fort taux de NaN appartiennent principalement au groupe des caractéristiques **logement** (`COMMONAREA_*`, `NONLIVINGAPARTMENTS_*`, etc.) — l'information était probablement non collectée pour une grande partie des clients.

# ---
# 
# ## Phase 1 — 1.5 Distributions des variables principales
# 
# ### Objectif
# Visualiser la forme des variables clés avant tout nettoyage : détecter les asymétries, valeurs aberrantes et déséquilibres qui guideront les choix de la Phase 2.

# ### Étape 1 — Variables numériques (grille 2×2)
# 
# On affiche les histogrammes de 4 variables numériques clés.
# > **Note sur `DAYS_EMPLOYED`** : certaines valeurs valent 365 243 — c'est une valeur sentinelle du dataset Home Credit pour coder "sans emploi". On les filtre avant de tracer.

import matplotlib.pyplot as plt
import numpy as np

# Préparation des séries (conversions + filtre aberrant)
age_ans          = application_train['DAYS_BIRTH'].abs() / 365
anciennete_ans   = application_train.loc[
    application_train['DAYS_EMPLOYED'] != 365243, 'DAYS_EMPLOYED'
].abs() / 365
n_aberrants      = (application_train['DAYS_EMPLOYED'] == 365243).sum()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Distributions des variables numériques clés", fontsize=15, fontweight='bold', y=1.01)

# — AMT_CREDIT
axes[0, 0].hist(application_train['AMT_CREDIT'].dropna(), bins=60, color='steelblue', edgecolor='white')
axes[0, 0].set_title("Montant du crédit (AMT_CREDIT)")
axes[0, 0].set_xlabel("Montant (€)")
axes[0, 0].set_ylabel("Nombre de clients")
axes[0, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))

# — AMT_INCOME_TOTAL (zoom : on écarte le top 1% pour ne pas écraser le graphique)
revenu_p99 = application_train['AMT_INCOME_TOTAL'].quantile(0.99)
axes[0, 1].hist(application_train.loc[application_train['AMT_INCOME_TOTAL'] <= revenu_p99, 'AMT_INCOME_TOTAL'],
                bins=60, color='mediumseagreen', edgecolor='white')
axes[0, 1].set_title("Revenus annuels (AMT_INCOME_TOTAL) — zoom 99e percentile")
axes[0, 1].set_xlabel("Revenu (€)")
axes[0, 1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))

# — ÂGE
axes[1, 0].hist(age_ans, bins=50, color='coral', edgecolor='white')
axes[1, 0].set_title("Âge des clients (DAYS_BIRTH → années)")
axes[1, 0].set_xlabel("Âge (années)")
axes[1, 0].set_ylabel("Nombre de clients")

# — ANCIENNETÉ EMPLOI
axes[1, 1].hist(anciennete_ans, bins=50, color='mediumpurple', edgecolor='white')
axes[1, 1].set_title(f"Ancienneté emploi — {n_aberrants:,} valeurs 365 243 exclues")
axes[1, 1].set_xlabel("Années d'ancienneté")

for ax in axes.flat:
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('graphiques/distributions_numeriques.png', dpi=150, bbox_inches='tight')
print("Sauvegardé : graphiques/distributions_numeriques.png")
plt.show()

# ### Étape 2 — Variables catégorielles (grille 2×2)
# 
# On visualise 4 variables qualitatives avec leur décompte de modalités.
# La 4e variable choisie est `NAME_FAMILY_STATUS` (situation familiale), très liée au profil de risque.

vars_cat = {
    'NAME_CONTRACT_TYPE' : "Type de contrat",
    'CODE_GENDER'        : "Genre",
    'NAME_EDUCATION_TYPE': "Niveau d'éducation",
    'NAME_FAMILY_STATUS' : "Situation familiale",
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Distributions des variables catégorielles clés", fontsize=15, fontweight='bold', y=1.01)

for ax, (col, titre) in zip(axes.flat, vars_cat.items()):
    counts = application_train[col].value_counts()
    bars = ax.bar(range(len(counts)), counts.values, color='steelblue', edgecolor='white')
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=30, ha='right', fontsize=9)
    ax.set_title(titre)
    ax.set_ylabel("Nombre de clients")
    # Pourcentage au-dessus de chaque barre
    total = counts.sum()
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.005,
                f"{val/total:.1%}", ha='center', va='bottom', fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('graphiques/distributions_categorielles.png', dpi=150, bbox_inches='tight')
print("Sauvegardé : graphiques/distributions_categorielles.png")
plt.show()

# ---
# 
# ## Résumé — 1.5 Distributions des variables principales
# 
# ### Variables numériques
# 
# | Variable | Observation clé | Anomalie repérée |
# |---|---|---|
# | **AMT_CREDIT** | Distribution asymétrique à droite — la majorité des crédits sont petits, quelques très grands crédits tirent la queue vers la droite | Pas de valeur aberrante technique, mais des outliers extrêmes à surveiller |
# | **AMT_INCOME_TOTAL** | Forte asymétrie droite — quelques revenus très élevés (top 1 % filtré pour le graphique) écrasent la majorité | Outliers à plusieurs millions → à plafonner ou logger en Phase 2 |
# | **DAYS_BIRTH (âge)** | Distribution relativement uniforme entre 20 et 70 ans, légère sur-représentation des 30–50 ans | Aucune valeur aberrante |
# | **DAYS_EMPLOYED (ancienneté)** | Concentrée sur 0–15 ans, queue droite modérée | **365 243** présent pour ~55 000 clients — valeur sentinelle "sans emploi" à remplacer par NaN en Phase 2 |
# 
# ### Variables catégorielles
# 
# | Variable | Observation clé |
# |---|---|
# | **NAME_CONTRACT_TYPE** | Très déséquilibré : les crédits Cash dominent largement face aux Revolving |
# | **CODE_GENDER** | Majorité de femmes (~65 %) — la variable XNA (indéfini) devra être traitée |
# | **NAME_EDUCATION_TYPE** | Le niveau "Secondary / secondary special" est de loin le plus représenté |
# | **NAME_FAMILY_STATUS** | "Married" domine, suivi de "Single/not married" — information utile pour le scoring |

# #### Vérification — Identification des clients avec DAYS_EMPLOYED = 365 243
# 
# Avant de passer en Phase 2, on confirme l'origine de la valeur sentinelle et on crée le flag d'anomalie qui sera conservé comme feature.

# Qui sont les clients avec DAYS_EMPLOYED = 365243 ?
mask = application_train['DAYS_EMPLOYED'] == 365243

print(application_train[mask]['NAME_INCOME_TYPE'].value_counts())

# Étape 1 : créer le flag AVANT de remplacer
application_train['FLAG_EMPLOYED_ANOMALY'] = (
    application_train['DAYS_EMPLOYED'] == 365243
).astype(int)

# Étape 2 : remplacer par NaN
application_train['DAYS_EMPLOYED'] = application_train['DAYS_EMPLOYED'].replace(
    365243, np.nan
)

# ---
# 
# ## Phase 2 — Nettoyage des données
# 
# ---
# 
# ### 2.1 Gestion des valeurs manquantes
# 
# **Objectif** : Supprimer les colonnes trop vides (> 40 % de NaN), traiter les cas particuliers (`OWN_CAR_AGE`, `DAYS_EMPLOYED`), imputer le reste.

# Phase 2 — 2.1 Suppression des colonnes avec trop de valeurs manquantes
df_train = application_train.copy()

# Taux de NaN par colonne
nan_rate = df_train.isnull().mean()

# Colonnes avec plus de 40 % de valeurs manquantes
cols_to_drop = nan_rate[nan_rate > 0.40].index.tolist()

print(f"Colonnes avec > 40 % de NaN : {len(cols_to_drop)}")
print(cols_to_drop)

# Suppression
df_train.drop(columns=cols_to_drop, inplace=True)

print(f"\nNouvelle forme : {df_train.shape}")

# --- Récupération et imputation de OWN_CAR_AGE ---

# 1. Récupérer OWN_CAR_AGE depuis le dataframe original et l'ajouter à df_train
df_train['OWN_CAR_AGE'] = application_train['OWN_CAR_AGE']

# 2. NaN dans OWN_CAR_AGE selon FLAG_OWN_CAR
print("NaN dans OWN_CAR_AGE par valeur de FLAG_OWN_CAR :")
print(df_train.groupby('FLAG_OWN_CAR')['OWN_CAR_AGE'].apply(lambda x: x.isna().sum()).rename('nb_NaN'))

# 3. Imputation
#    - Pas de voiture (N) → 0
df_train.loc[df_train['FLAG_OWN_CAR'] == 'N', 'OWN_CAR_AGE'] = 0

#    - A une voiture (Y) mais âge inconnu → médiane des propriétaires
median_car_age = df_train.loc[df_train['FLAG_OWN_CAR'] == 'Y', 'OWN_CAR_AGE'].median()
df_train.loc[
    (df_train['FLAG_OWN_CAR'] == 'Y') & (df_train['OWN_CAR_AGE'].isna()),
    'OWN_CAR_AGE'
] = median_car_age
print(f"\nMédiane utilisée pour les propriétaires sans âge connu : {median_car_age} ans")

# 4. Vérification : plus aucun NaN
remaining_nan = df_train['OWN_CAR_AGE'].isna().sum()
print(f"\nNaN restants dans OWN_CAR_AGE : {remaining_nan}")
assert remaining_nan == 0, "Il reste des NaN dans OWN_CAR_AGE !"

# 5. Confirmation du shape : doit afficher 75 colonnes
print(f"\ndf_train.shape : {df_train.shape}  ← 75 colonnes attendues")

# --- Traitement de DAYS_EMPLOYED dans df_train ---

# 1. FLAG_EMPLOYED_ANOMALY (créé sur application_train avant la copie → déjà dans df_train)
#    On affiche le nombre d'anomalies détectées
n_anomalies = df_train['FLAG_EMPLOYED_ANOMALY'].sum()
print(f"Nombre de clients avec FLAG_EMPLOYED_ANOMALY == 1 : {n_anomalies:,}")

# 2. Remplacement défensif : s'assurer qu'aucun 365243 ne subsiste dans df_train
df_train['DAYS_EMPLOYED'] = df_train['DAYS_EMPLOYED'].replace(365243, np.nan)

# 3. Imputation des NaN avec la médiane (hors anomalies, donc hors NaN)
median_days_employed = df_train['DAYS_EMPLOYED'].median()
df_train['DAYS_EMPLOYED'] = df_train['DAYS_EMPLOYED'].fillna(median_days_employed)
print(f"Médiane utilisée pour l'imputation : {median_days_employed:.0f} jours")

# 4. Vérification : aucun NaN ne doit subsister
nan_remaining = df_train['DAYS_EMPLOYED'].isna().sum()
print(f"NaN restants dans DAYS_EMPLOYED : {nan_remaining}")
assert nan_remaining == 0, "Il reste des NaN dans DAYS_EMPLOYED !"

# 5. Statistiques pour confirmer que 365243 a disparu
print("\nStatistiques de DAYS_EMPLOYED après imputation :")
print(df_train['DAYS_EMPLOYED'].describe())

# Phase 2 — 2.1 Imputation des valeurs manquantes restantes dans df_train

# --- Étape 1 : état des lieux ---
nan_par_col = df_train.isnull().sum()
cols_nan = nan_par_col[nan_par_col > 0]

print("=" * 55)
print(f"COLONNES AVEC DES NaN RESTANTS : {len(cols_nan)}")
print("=" * 55)
for col, n in cols_nan.items():
    print(f"  {col:<45} {n:>6,}")
print("=" * 55)

# --- Étape 2 : imputation des colonnes numériques avec la médiane ---
num_cols_nan = df_train[cols_nan.index].select_dtypes(include='number').columns.tolist()
for col in num_cols_nan:
    mediane = df_train[col].median()
    df_train[col] = df_train[col].fillna(mediane)

print(f"\nColonnes numériques imputées avec la médiane ({len(num_cols_nan)}) :")
print(num_cols_nan)

# --- Étape 3 : imputation des colonnes catégorielles avec le mode ---
cat_cols_nan = df_train[cols_nan.index].select_dtypes(include='object').columns.tolist()
for col in cat_cols_nan:
    mode = df_train[col].mode()[0]
    df_train[col] = df_train[col].fillna(mode)

print(f"\nColonnes catégorielles imputées avec le mode ({len(cat_cols_nan)}) :")
print(cat_cols_nan)

# --- Étape 4 : vérification — aucun NaN ne doit subsister ---
total_nan = df_train.isnull().sum().sum()
print(f"\n{'=' * 55}")
print(f"NaN restants dans df_train (total) : {total_nan}")
assert total_nan == 0, f"Il reste {total_nan} NaN dans df_train !"
print("Aucun NaN restant — imputation complète.")
print("=" * 55)

# --- Étape 5 : confirmation du shape ---
print(f"\ndf_train.shape : {df_train.shape}")

# ---
# 
# ## Résumé — 2.1 Gestion des valeurs manquantes
# 
# | Étape | Action | Résultat |
# |---|---|---|
# | **Suppression colonnes > 40 % NaN** | 49 colonnes supprimées | 122 → 74 colonnes, puis 75 après réintégration de OWN_CAR_AGE |
# | **OWN_CAR_AGE** | 0 si pas de voiture ; médiane = 9 ans si propriétaire sans âge connu | 202 929 NaN résolus |
# | **DAYS_EMPLOYED** | Valeur sentinelle 365 243 → `FLAG_EMPLOYED_ANOMALY` créé + imputation médiane | 55 374 anomalies traitées |
# | **Colonnes numériques (16)** | Imputation par médiane | `AMT_ANNUITY`, `EXT_SOURCE_2/3`, `AMT_REQ_CREDIT_BUREAU_*`, … |
# | **Colonnes catégorielles (2)** | Imputation par mode | `NAME_TYPE_SUITE`, `OCCUPATION_TYPE` |
# 
# **Résultat final :** `df_train` — **307 511 lignes × 75 colonnes**, **0 NaN restant**.

# ---
# 
# ### 2.2 Traitement des valeurs aberrantes
# 
# **Objectif** : Identifier les distributions très asymétriques (`AMT_INCOME_TOTAL`, `AMT_CREDIT`) et les normaliser par transformation logarithmique pour limiter l'influence des valeurs extrêmes sur le modèle.

import numpy as np
import matplotlib.pyplot as plt

# --- Étape 1 : statistiques avant transformation ---
print("=" * 55)
print("STATISTIQUES — AVANT TRANSFORMATION LOG")
print("=" * 55)
print(df_train[['AMT_INCOME_TOTAL', 'AMT_CREDIT']].describe().round(0).to_string())

# --- Étape 2 : colonnes log ---
df_train['AMT_INCOME_TOTAL_LOG'] = np.log1p(df_train['AMT_INCOME_TOTAL'])
df_train['AMT_CREDIT_LOG']       = np.log1p(df_train['AMT_CREDIT'])

# --- Étape 3 : 4 histogrammes côte à côte ---
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Transformation log — AMT_INCOME_TOTAL et AMT_CREDIT",
             fontsize=14, fontweight='bold')

axes[0].hist(df_train['AMT_INCOME_TOTAL'], bins=80, color='steelblue', edgecolor='white')
axes[0].set_title("AMT_INCOME_TOTAL (brut)")
axes[0].set_xlabel("Revenu (€)")
axes[0].set_ylabel("Nombre de clients")
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))

axes[1].hist(df_train['AMT_INCOME_TOTAL_LOG'], bins=80, color='steelblue', edgecolor='white')
axes[1].set_title("AMT_INCOME_TOTAL_LOG (log1p)")
axes[1].set_xlabel("log1p(Revenu)")
axes[1].set_ylabel("Nombre de clients")

axes[2].hist(df_train['AMT_CREDIT'], bins=80, color='coral', edgecolor='white')
axes[2].set_title("AMT_CREDIT (brut)")
axes[2].set_xlabel("Montant crédit (€)")
axes[2].set_ylabel("Nombre de clients")
axes[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))

axes[3].hist(df_train['AMT_CREDIT_LOG'], bins=80, color='coral', edgecolor='white')
axes[3].set_title("AMT_CREDIT_LOG (log1p)")
axes[3].set_xlabel("log1p(Crédit)")
axes[3].set_ylabel("Nombre de clients")

for ax in axes:
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('graphiques/log_transform.png', dpi=150, bbox_inches='tight')
print("\nGraphique sauvegardé : graphiques/log_transform.png")
plt.show()

# --- Étape 4 : confirmation du shape ---
print(f"\ndf_train.shape : {df_train.shape}  ← 77 colonnes attendues (+2 colonnes log)")

# ---
# 
# ## Résumé — 2.2 Traitement des valeurs aberrantes
# 
# | Variable | Problème observé | Traitement appliqué | Nouvelle colonne |
# |---|---|---|---|
# | **AMT_INCOME_TOTAL** | Forte asymétrie droite — quelques revenus à plusieurs millions écrasent la majorité | Transformation `log1p` | `AMT_INCOME_TOTAL_LOG` |
# | **AMT_CREDIT** | Asymétrie droite modérée — queue longue vers les gros crédits | Transformation `log1p` | `AMT_CREDIT_LOG` |
# 
# **Pourquoi `log1p` ?** Elle compresse les grandes valeurs, rapproche la distribution d'une gaussienne et est robuste aux zéros (`log(0)` indéfini, `log1p(0) = 0`).
# 
# **Résultat :** `df_train` passe de 75 à **77 colonnes** — les colonnes brutes sont conservées pour traçabilité.

# ---
# 
# ### 2.3 Encodage des variables catégorielles
# 
# **Objectif** : Convertir toutes les colonnes `object` de `df_train` en valeurs numériques exploitables par le modèle — via Label Encoding (2 modalités) ou One-Hot Encoding (> 2 modalités).

# --- Étape 1 : État des lieux — colonnes object et leurs modalités ---
obj_cols = df_train.select_dtypes(include='object').columns.tolist()

print("=" * 55)
print(f"COLONNES DE TYPE object : {len(obj_cols)}")
print("=" * 55)

for col in obj_cols:
    print(f"\n▶ {col}  ({df_train[col].nunique()} modalités)")
    print(df_train[col].value_counts().to_string())

# --- Étape 2 : Traitement de CODE_GENDER — remplacement de XNA par le mode ---
n_xna = (df_train['CODE_GENDER'] == 'XNA').sum()
mode_gender = df_train.loc[df_train['CODE_GENDER'] != 'XNA', 'CODE_GENDER'].mode()[0]

df_train['CODE_GENDER'] = df_train['CODE_GENDER'].replace('XNA', mode_gender)

print(f"Valeurs XNA remplacées : {n_xna}")
print(f"Mode utilisé           : {mode_gender}")
print("\nDistribution CODE_GENDER après remplacement :")
print(df_train['CODE_GENDER'].value_counts())

# --- Étape 3 : Label Encoding — colonnes object avec exactement 2 modalités ---
from sklearn.preprocessing import LabelEncoder

obj_cols = df_train.select_dtypes(include='object').columns.tolist()
label_encoded_cols = [col for col in obj_cols if df_train[col].nunique() == 2]

le = LabelEncoder()
for col in label_encoded_cols:
    df_train[col] = le.fit_transform(df_train[col])

print(f"Colonnes encodées en Label Encoding ({len(label_encoded_cols)}) :")
for col in label_encoded_cols:
    print(f"  {col}")

# --- Étape 4 : One-Hot Encoding — colonnes object restantes (> 2 modalités) ---
obj_cols_remaining = df_train.select_dtypes(include='object').columns.tolist()

n_cols_before = df_train.shape[1]
df_train = pd.get_dummies(df_train, columns=obj_cols_remaining, drop_first=False, dtype=int)
n_cols_after = df_train.shape[1]

nouvelles_cols = n_cols_after - n_cols_before + len(obj_cols_remaining)
print(f"Colonnes OHE appliqué sur ({len(obj_cols_remaining)}) : {obj_cols_remaining}")
print(f"\nNombre de nouvelles colonnes créées : {nouvelles_cols}")
print(f"Shape après OHE : {df_train.shape}")

# --- Étape 5 : Vérification — aucune colonne object ne doit subsister ---
obj_restantes = df_train.select_dtypes(include='object').columns.tolist()

print("=" * 55)
print("VÉRIFICATION — COLONNES object RESTANTES")
print("=" * 55)
if len(obj_restantes) == 0:
    print("  Aucune colonne de type object — encodage complet.")
else:
    print(f"  ATTENTION : {len(obj_restantes)} colonne(s) object restante(s) !")
    print(obj_restantes)

print(f"\ndf_train.shape : {df_train.shape}")
