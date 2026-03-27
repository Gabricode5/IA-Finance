import polars as pl

# 1. Chargement optimisé
df = pl.scan_csv("data.csv")

# 2. Nettoyage et Feature Engineering
df_clean = (
    df
    # Supprimer les colonnes de texte non structurées
    .drop(["nameOrig", "nameDest", "isFlaggedFraud"]) 
    
    # Créer des variables logiques pour le modèle
    .with_columns([
        # Encodage du type
        pl.col("type").cast(pl.Categorical).to_physical(),
        
        # Détection d'anomalies de solde
        (pl.col("newbalanceOrig") + pl.col("amount") != pl.col("oldbalanceOrg")).alias("balance_error_orig"),
    ])
    
    # Supprimer les lignes avec des valeurs manquantes (s'il y en a)
    .drop_nulls()
)

# 3. Exécution et sauvegarde
dataset_final = df_clean.collect()
print(dataset_final.head())

# Sauvegarde en format Parquet (plus léger et rapide que CSV pour l'IA)
dataset_final.write_parquet("data_cleaned.parquet")