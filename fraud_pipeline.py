"""
=============================================================================
    PIPELINE DE DÉTECTION DE FRAUDE FINANCIÈRE (AML)
    Dataset : Synthetic Financial Datasets For Fraud Detection (Kaggle)
=============================================================================
Ce script implémente un pipeline complet de Machine Learning pour
détecter les transactions frauduleuses. Il est divisé en 4 étapes :
    1. Analyse Exploratoire (EDA)
    2. Nettoyage & Préparation des données
    3. Modélisation (Régression Logistique + XGBoost)
    4. Évaluation des modèles
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek

sns.set_style("whitegrid")


# =============================================================================
# CHARGEMENT + ÉCHANTILLONNAGE
# =============================================================================
# On prend un échantillon de 50 000 lignes dès le départ pour que
# tout le pipeline (EDA incluse) tourne sur un volume raisonnable.
# On garde TOUTES les fraudes + on complète avec des normales.
CHEMIN_FICHIER = "data.csv"
TAILLE_ECHANTILLON = 50_000

print("LANCEMENT DU PIPELINE DE DÉTECTION DE FRAUDE\n")
print(f"Chargement du fichier : {CHEMIN_FICHIER}")
df_complet = pd.read_csv(CHEMIN_FICHIER)
print(f"Dataset complet : {df_complet.shape[0]:,} lignes × {df_complet.shape[1]} colonnes")

# Séparer fraudes et normales
df_fraude = df_complet[df_complet["isFraud"] == 1]
df_normal = df_complet[df_complet["isFraud"] == 0]

# Garder toutes les fraudes + compléter avec des normales jusqu'à 50 000
n_normales = TAILLE_ECHANTILLON - len(df_fraude)
df_normal_sample = df_normal.sample(n=n_normales, random_state=42)

df = pd.concat([df_fraude, df_normal_sample], ignore_index=True).sample(frac=1, random_state=42)

print(f"\nÉchantillon retenu : {len(df):,} lignes")
print(f"  Fraudes  : {len(df_fraude):,}")
print(f"  Normales : {n_normales:,}")
print(f"  Ratio fraude : {df['isFraud'].mean():.2%}\n")


# =============================================================================
# ÉTAPE 1 : ANALYSE EXPLORATOIRE (EDA)
# =============================================================================
print("=" * 70)
print("  ÉTAPE 1 : ANALYSE EXPLORATOIRE (EDA)")
print("=" * 70)

print("\nDimensions :", df.shape)
print(df.head())
print("\nValeurs manquantes :", df.isnull().sum().sum())

# Répartition de isFraud
print("\nRépartition de isFraud :")
print(df["isFraud"].value_counts())

fig, ax = plt.subplots()
df["isFraud"].value_counts().plot(kind="bar", color=["#2ecc71", "#e74c3c"], edgecolor="black", ax=ax)
ax.set_title("Répartition des classes (isFraud)", fontweight="bold")
ax.set_xlabel("0 = Normal, 1 = Fraude")
ax.set_ylabel("Nombre de transactions")
plt.tight_layout()
plt.savefig("eda_repartition_classes.png", dpi=150)
plt.show()

# Taux de fraude par type
fraude_par_type = df.groupby("type")["isFraud"].mean().sort_values(ascending=False)

fig, ax = plt.subplots()
fraude_par_type.plot(kind="bar", color="#e67e22", edgecolor="black", ax=ax)
ax.set_title("Proportion de fraudes par type de transaction", fontweight="bold")
ax.set_ylabel("Taux de fraude")
for i, v in enumerate(fraude_par_type):
    ax.text(i, v + 0.002, f"{v:.2%}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("eda_fraude_par_type.png", dpi=150)
plt.show()

# Distribution des montants
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df[df["isFraud"] == 0]["amount"].plot(kind="hist", bins=50, ax=axes[0], color="#3498db", edgecolor="black", log=True)
axes[0].set_title("Montants – Normal", fontweight="bold")
df[df["isFraud"] == 1]["amount"].plot(kind="hist", bins=50, ax=axes[1], color="#e74c3c", edgecolor="black", log=True)
axes[1].set_title("Montants – Fraude", fontweight="bold")
plt.tight_layout()
plt.savefig("eda_distribution_montants.png", dpi=150)
plt.show()

print("Analyse exploratoire terminée.\n")


# =============================================================================
# ÉTAPE 2 : NETTOYAGE ET PRÉPARATION DES DONNÉES
# =============================================================================
print("=" * 70)
print("  ÉTAPE 2 : NETTOYAGE ET PRÉPARATION DES DONNÉES")
print("=" * 70)

# Suppression des colonnes inutiles
df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud", "step"], inplace=True)
print(f"\nColonnes restantes : {list(df.columns)}")

# One-Hot Encoding de 'type'
df = pd.get_dummies(df, columns=["type"], drop_first=True)

# Séparation X / y
X = df.drop(columns=["isFraud"])
y = df["isFraud"]

# Split train/test 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train : {X_train.shape[0]}  |  Test : {X_test.shape[0]}")

# Rééquilibrage avec SMOTETomek (SMOTE + nettoyage Tomek Links)
print(f"\nAvant SMOTETomek : {y_train.value_counts().to_dict()}")
X_train, y_train = SMOTETomek(random_state=42).fit_resample(X_train, y_train)
print(f"Après SMOTETomek : {y_train.value_counts().to_dict()}")

print("Préparation terminée.\n")


# =============================================================================
# ÉTAPE 3 : MODÉLISATION
# =============================================================================
print("=" * 70)
print("  ÉTAPE 3 : MODÉLISATION")
print("=" * 70)

# Modèle 1 : Régression Logistique (baseline)
print("\nEntraînement : Régression Logistique...")
lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr.fit(X_train, y_train)
print("OK")

# Modèle 2 : XGBoost
print("\nEntraînement : XGBoost...")
xgb = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric="logloss", n_jobs=-1,
)
xgb.fit(X_train, y_train)
print("OK")

modeles = {"Régression Logistique": lr, "XGBoost": xgb}


# =============================================================================
# ÉTAPE 4 : ÉVALUATION
# =============================================================================
print("\n" + "=" * 70)
print("  ÉTAPE 4 : ÉVALUATION DES MODÈLES")
print("=" * 70)

resultats = []

for nom, modele in modeles.items():
    y_pred = modele.predict(X_test)

    precision = precision_score(y_test, y_pred)
    rappel = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n--- {nom} ---")
    print(f"Précision : {precision:.4f}  |  Rappel : {rappel:.4f}  |  F1 : {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraude"]))

    resultats.append({"Modèle": nom, "Précision": f"{precision:.4f}", "Rappel": f"{rappel:.4f}", "F1-Score": f"{f1:.4f}"})

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues" if "Logistique" in nom else "Oranges",
                xticklabels=["Normal", "Fraude"], yticklabels=["Normal", "Fraude"], ax=ax)
    ax.set_title(f"Matrice de Confusion – {nom}", fontweight="bold")
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Réalité")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{nom.replace(' ', '_').lower()}.png", dpi=150)
    plt.show()

# Tableau comparatif
print("\n" + "=" * 70)
print("TABLEAU COMPARATIF")
print("=" * 70)
print(pd.DataFrame(resultats).to_string(index=False))

print("\nINTERPRÉTATION :")
print("  Le Rappel (Recall) est la métrique clé en détection de fraude.")
print("  Faux Négatif = fraude non détectée = perte financière.")
print("  Faux Positif = fausse alerte = désagrément client.\n")

print("PIPELINE TERMINÉ AVEC SUCCÈS !")
