from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from models import TreeNode, ScalingStat

OUTPUT_DIR = Path("outputs")
REPORT_PATH = OUTPUT_DIR / "fraud_report.md"
TREE_PATH = OUTPUT_DIR / "fraud_tree.json"
LOGISTIC_PATH = OUTPUT_DIR / "fraud_logistic.json"
BAYES_PATH = OUTPUT_DIR / "fraud_gaussian_nb.json"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
CONFUSION_CSV_PATH = OUTPUT_DIR / "confusion_matrix.csv"
CLASS_SVG_PATH = OUTPUT_DIR / "class_distribution.svg"
TYPE_SVG_PATH = OUTPUT_DIR / "fraud_by_type.svg"
CONFUSION_SVG_PATH = OUTPUT_DIR / "confusion_matrix.svg"
TYPE_RATE_SVG_PATH = OUTPUT_DIR / "fraud_rate_by_type.svg"
STEP_RATE_SVG_PATH = OUTPUT_DIR / "fraud_rate_by_period.svg"
AMOUNT_RATE_SVG_PATH = OUTPUT_DIR / "fraud_rate_by_amount_bucket.svg"
SIGNAL_RATE_SVG_PATH = OUTPUT_DIR / "fraud_signal_comparison.svg"

CLEANED_PATH = OUTPUT_DIR / "data_fraud_cleaned.parquet"
REDUCED_PATH = OUTPUT_DIR / "data_fraud_reduced_50000.parquet"
SCALED_REDUCED_PATH = OUTPUT_DIR / "data_fraud_reduced_50000_scaled.parquet"


def build_svg_bar_chart(
    items: list[tuple[str, float]],
    path: Path,
    title: str,
    color: str,
    value_format: str = "int",
) -> None:
    width = 900
    height = 420
    left = 100
    bottom = 70
    top = 70
    bar_gap = 40
    bar_width = max(60, int((width - left - 80 - bar_gap * (len(items) - 1)) / max(len(items), 1)))
    chart_height = height - top - bottom
    max_value = max(value for _, value in items) if items else 1.0
    bars = []
    labels = []
    values = []
    x = left
    for label, value in items:
        bar_height = 0 if max_value == 0 else (value / max_value) * chart_height
        y = height - bottom - bar_height
        bars.append(
            f'<rect x="{x}" y="{y:.2f}" width="{bar_width}" height="{bar_height:.2f}" fill="{color}" rx="6" />'
        )
        labels.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{height - 30}" font-size="15" text-anchor="middle">{label}</text>'
        )
        text_value = f"{int(value):,}".replace(",", " ") if value_format == "int" else f"{value:.2%}"
        values.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{y - 10:.2f}" font-size="14" text-anchor="middle">{text_value}</text>'
        )
        x += bar_width + bar_gap

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="#fcfbf7"/>
<text x="{width / 2}" y="38" font-size="24" text-anchor="middle" font-weight="bold">{title}</text>
<line x1="{left}" y1="{height - bottom}" x2="{width - 30}" y2="{height - bottom}" stroke="#444" stroke-width="2"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#444" stroke-width="2"/>
{''.join(bars)}
{''.join(labels)}
{''.join(values)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def build_svg_grouped_rate_chart(
    groups: list[str],
    normal_values: list[float],
    fraud_values: list[float],
    path: Path,
    title: str,
    normal_label: str = "Normal",
    fraud_label: str = "Fraude",
) -> None:
    width = max(1000, 160 + len(groups) * 120)
    height = 500
    left = 90
    bottom = 100
    top = 70
    chart_height = height - top - bottom
    group_width = 90
    bar_width = 28
    gap = 12
    max_value = max(normal_values + fraud_values) if groups else 1.0
    elements: list[str] = []
    x = left

    for group, normal, fraud in zip(groups, normal_values, fraud_values):
        n_height = (normal / max_value) * chart_height if max_value else 0.0
        f_height = (fraud / max_value) * chart_height if max_value else 0.0
        n_y = height - bottom - n_height
        f_y = height - bottom - f_height
        elements.append(
            f'<rect x="{x}" y="{n_y:.2f}" width="{bar_width}" height="{n_height:.2f}" fill="#5b8c5a" rx="5" />'
        )
        elements.append(
            f'<rect x="{x + bar_width + gap}" y="{f_y:.2f}" width="{bar_width}" height="{f_height:.2f}" fill="#c8553d" rx="5" />'
        )
        elements.append(
            f'<text x="{x + group_width / 2}" y="{height - 40}" font-size="12" text-anchor="middle">{group}</text>'
        )
        elements.append(
            f'<text x="{x + bar_width / 2}" y="{n_y - 10:.2f}" font-size="11" text-anchor="middle">{normal:.1%}</text>'
        )
        elements.append(
            f'<text x="{x + bar_width + gap + bar_width / 2}" y="{f_y - 10:.2f}" font-size="11" text-anchor="middle">{fraud:.1%}</text>'
        )
        x += group_width + 24

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="#fcfbf7"/>
<text x="{width / 2}" y="38" font-size="24" text-anchor="middle" font-weight="bold">{title}</text>
<line x1="{left}" y1="{height - bottom}" x2="{width - 30}" y2="{height - bottom}" stroke="#444" stroke-width="2"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#444" stroke-width="2"/>
<rect x="{width - 240}" y="22" width="18" height="18" fill="#5b8c5a" rx="4" />
<text x="{width - 216}" y="36" font-size="13">{normal_label}</text>
<rect x="{width - 120}" y="22" width="18" height="18" fill="#c8553d" rx="4" />
<text x="{width - 96}" y="36" font-size="13">{fraud_label}</text>
{''.join(elements)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def build_svg_rate_chart(
    items: list[tuple[str, float]],
    path: Path,
    title: str,
    color: str,
) -> None:
    build_svg_bar_chart(items, path, title, color, value_format="percent")


def build_svg_confusion(confusion: dict[str, int], path: Path) -> None:
    width = 620
    height = 500
    x0 = 170
    y0 = 120
    cell = 130
    cells = [
        ("TN", confusion["tn"], x0, y0, "#d9ead3"),
        ("FP", confusion["fp"], x0 + cell, y0, "#f4cccc"),
        ("FN", confusion["fn"], x0, y0 + cell, "#fce5cd"),
        ("TP", confusion["tp"], x0 + cell, y0 + cell, "#cfe2f3"),
    ]
    cell_svg = []
    for label, value, x, y, fill in cells:
        cell_svg.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#444" stroke-width="2"/>')
        cell_svg.append(f'<text x="{x + cell / 2}" y="{y + 48}" font-size="22" text-anchor="middle" font-weight="bold">{label}</text>')
        cell_svg.append(
            f'<text x="{x + cell / 2}" y="{y + 84}" font-size="18" text-anchor="middle">{value:,}</text>'.replace(",", " ")
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="#fcfbf7"/>
<text x="{width / 2}" y="46" font-size="24" text-anchor="middle" font-weight="bold">Matrice de confusion</text>
<text x="{x0 + cell}" y="92" font-size="18" text-anchor="middle">Pred 0</text>
<text x="{x0 + 2 * cell}" y="92" font-size="18" text-anchor="middle">Pred 1</text>
<text x="82" y="{y0 + 72}" font-size="18" text-anchor="middle">Vrai 0</text>
<text x="82" y="{y0 + 202}" font-size="18" text-anchor="middle">Vrai 1</text>
{''.join(cell_svg)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def perform_analysis(df: pl.DataFrame, reduced_df: pl.DataFrame) -> dict[str, Any]:
    class_rows = df.group_by("isFraud").len().sort("isFraud").to_dicts()
    reduced_class_rows = reduced_df.group_by("isFraud").len().sort("isFraud").to_dicts()
    class_counts = {int(row["isFraud"]): int(row["len"]) for row in reduced_class_rows}

    fraud_by_type_rows = (
        reduced_df.group_by(["type", "isFraud"])
        .len()
        .sort(["type", "isFraud"])
        .to_dicts()
    )
    fraud_by_type: list[dict[str, Any]] = []
    by_type: dict[str, dict[str, int]] = {}
    for row in fraud_by_type_rows:
        current = by_type.setdefault(row["type"], {"type": row["type"], "normal": 0, "fraud": 0})
        if int(row["isFraud"]) == 1:
            current["fraud"] = int(row["len"])
        else:
            current["normal"] = int(row["len"])
    fraud_by_type = list(by_type.values())

    fraud_rate_by_type_rows = (
        df.group_by("type")
        .agg(
            [
                pl.len().alias("transactions"),
                pl.col("isFraud").sum().alias("frauds"),
                pl.col("isFraud").mean().alias("fraud_rate"),
            ]
        )
        .sort("fraud_rate", descending=True)
        .to_dicts()
    )

    fraud_rate_by_period_rows = (
        df.with_columns((((pl.col("step") - 1) // 24) + 1).alias("period"))
        .group_by("period")
        .agg(
            [
                pl.len().alias("transactions"),
                pl.col("isFraud").sum().alias("frauds"),
                pl.col("isFraud").mean().alias("fraud_rate"),
            ]
        )
        .sort("period")
        .to_dicts()
    )

    amount_bucket_expr = (
        pl.when(pl.col("amount") < 1_000)
        .then(pl.lit("<1k"))
        .when(pl.col("amount") < 10_000)
        .then(pl.lit("1k-10k"))
        .when(pl.col("amount") < 100_000)
        .then(pl.lit("10k-100k"))
        .when(pl.col("amount") < 1_000_000)
        .then(pl.lit("100k-1M"))
        .when(pl.col("amount") < 10_000_000)
        .then(pl.lit("1M-10M"))
        .otherwise(pl.lit("10M+"))
        .alias("amount_bucket")
    )
    amount_bucket_order = {
        "<1k": 0,
        "1k-10k": 1,
        "10k-100k": 2,
        "100k-1M": 3,
        "1M-10M": 4,
        "10M+": 5,
    }
    fraud_rate_by_amount_rows = (
        df.with_columns(amount_bucket_expr)
        .group_by("amount_bucket")
        .agg(
            [
                pl.len().alias("transactions"),
                pl.col("isFraud").sum().alias("frauds"),
                pl.col("isFraud").mean().alias("fraud_rate"),
            ]
        )
        .with_columns(pl.col("amount_bucket").replace(amount_bucket_order).alias("order"))
        .sort("order")
        .drop("order")
        .to_dicts()
    )

    signal_features = [
        ("emptied_origin", "Orig videe"),
        ("dest_zero", "Dest solde 0"),
        ("dest_untouched", "Dest intacte"),
        ("orig_zero", "Orig solde 0"),
        ("balance_error_dest", "Erreur dest"),
        ("balance_error_orig", "Erreur orig"),
        ("flagged_fraud", "Flag systeme"),
    ]
    signal_comparison_rows: list[dict[str, Any]] = []
    for feature, label in signal_features:
        rates = (
            df.group_by("isFraud")
            .agg(pl.col(feature).mean().alias("rate"))
            .sort("isFraud")
            .to_dicts()
        )
        normal_rate = next(float(row["rate"]) for row in rates if int(row["isFraud"]) == 0)
        fraud_rate = next(float(row["rate"]) for row in rates if int(row["isFraud"]) == 1)
        signal_comparison_rows.append(
            {"label": label, "normal_rate": normal_rate, "fraud_rate": fraud_rate}
        )

    # Generate plots
    build_svg_bar_chart(
        [(str(cls), count) for cls, count in class_counts.items()],
        CLASS_SVG_PATH,
        "Distribution des classes",
        "#5b8c5a",
    )

    build_svg_grouped_rate_chart(
        [row["type"] for row in fraud_by_type],
        [row["normal"] for row in fraud_by_type],
        [row["fraud"] for row in fraud_by_type],
        TYPE_SVG_PATH,
        "Fraudes par type de transaction",
    )

    build_svg_rate_chart(
        [(row["type"], row["fraud_rate"]) for row in fraud_rate_by_type_rows],
        TYPE_RATE_SVG_PATH,
        "Taux de fraude par type",
        "#c8553d",
    )

    build_svg_rate_chart(
        [(f"Jour {row['period']}", row["fraud_rate"]) for row in fraud_rate_by_period_rows],
        STEP_RATE_SVG_PATH,
        "Taux de fraude par periode",
        "#c8553d",
    )

    build_svg_rate_chart(
        [(row["amount_bucket"], row["fraud_rate"]) for row in fraud_rate_by_amount_rows],
        AMOUNT_RATE_SVG_PATH,
        "Taux de fraude par tranche de montant",
        "#c8553d",
    )

    build_svg_grouped_rate_chart(
        [row["label"] for row in signal_comparison_rows],
        [row["normal_rate"] for row in signal_comparison_rows],
        [row["fraud_rate"] for row in signal_comparison_rows],
        SIGNAL_RATE_SVG_PATH,
        "Signaux de fraude",
    )

    return {
        "class_counts": class_counts,
        "fraud_by_type": fraud_by_type,
        "fraud_rate_by_type": fraud_rate_by_type_rows,
        "fraud_rate_by_period": fraud_rate_by_period_rows,
        "fraud_rate_by_amount": fraud_rate_by_amount_rows,
        "signal_comparison": signal_comparison_rows,
    }


def render_tree(node: TreeNode, depth: int = 0) -> list[str]:
    indent = "  " * depth
    if node.is_leaf:
        return [
            f"{indent}- Feuille: taux_fraude={node.positive_rate:.4f}, echantillons={node.samples}, fraudes={node.positives}"
        ]

    assert node.feature is not None
    assert node.threshold is not None
    assert node.left is not None
    assert node.right is not None
    lines = [
        f"{indent}- Si `{node.feature} <= {node.threshold:.4f}` (n={node.samples}, fraude={node.positives})"
    ]
    lines.extend(render_tree(node.left, depth + 1))
    lines.append(f"{indent}- Sinon")
    lines.extend(render_tree(node.right, depth + 1))
    return lines


def write_report(
    df: pl.DataFrame,
    reduced_df: pl.DataFrame,
    class_counts: dict[int, int],
    fraud_by_type: list[dict[str, Any]],
    tree_validation_confusion: dict[str, int],
    tree_validation_scores: dict[str, float],
    tree_test_confusion: dict[str, int],
    tree_test_scores: dict[str, float],
    tree_threshold: float,
    tree: TreeNode,
    logistic_validation_confusion: dict[str, int],
    logistic_validation_scores: dict[str, float],
    logistic_test_confusion: dict[str, int],
    logistic_test_scores: dict[str, float],
    logistic_threshold: float,
    logistic_model: LogisticModel,
    bayes_validation_scores: dict[str, float],
    bayes_test_confusion: dict[str, int],
    bayes_test_scores: dict[str, float],
    bayes_threshold: float,
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    train_frauds: int,
    validation_frauds: int,
    test_frauds: int,
    split_train_max_step: int,
    split_validation_max_step: int,
    scaling_stats: dict[str, ScalingStat],
) -> None:
    fraud_rate = class_counts[1] / (class_counts[0] + class_counts[1])
    report = f"""# Rapport de detection de fraude

## Resume

- Lignes analysees: {df.height:,}
- Lignes retenues apres reduction: {reduced_df.height:,}
- Variables apres nettoyage / enrichissement: {df.width}
- Fraudes: {class_counts[1]:,}
- Taux de fraude: {fraud_rate:.4%}
- Seuil arbre de decision: {tree_threshold:.2f}
- Seuil regression logistique: {logistic_threshold:.2f}
- Seuil Naive Bayes gaussien: {bayes_threshold:.2f}

## Nettoyage

- Aucune valeur manquante detectee dans le dataset source.
- Colonnes textuelles identifiantes retirees du modele: `nameOrig`, `nameDest`.
- Variables derivees ajoutees: indicateurs de type, soldes nuls, compte origine vide apres transaction, incoherences de solde, ratios et logarithmes.
- Variables interdites au modele pour eviter la fuite d\'information: `newbalanceOrig`, `newbalanceDest`, `isFlaggedFraud` et toutes les variables derivees apres transaction.
- Variables numeriques du modele centrees-reduites apres clipping `q01-q99` calcule sur le train uniquement.
- Jeu nettoye exporte dans: `{CLEANED_PATH}`
- Jeu reduit pour l\'entrainement exporte dans: `{REDUCED_PATH}`
- Jeu reduit centre-reduit exporte dans: `{SCALED_REDUCED_PATH}`

## Protocole anti fuite

- Entrainement: `step <= {split_train_max_step}` avec {train_rows:,} lignes et {train_frauds:,} fraudes.
- Validation: `{split_train_max_step} < step <= {split_validation_max_step}` avec {validation_rows:,} lignes et {validation_frauds:,} fraudes.
- Test: `step > {split_validation_max_step}` avec {test_rows:,} lignes et {test_frauds:,} fraudes.
- Le modele n\'utilise que des variables disponibles avant ou au debut de la transaction.

## Schema

```mermaid
flowchart LR
    A[data.csv] --> B[Nettoyage]
    B --> C[data_fraud_cleaned.parquet]
    C --> D[Feature engineering]
    D --> E[Train stratife]
    D --> F[Validation]
    D --> G[Test]
    E --> H[Arbre de decision]
    F --> I[Choix du seuil]
    H --> J[Predictions]
    I --> J
    J --> K[Matrice de confusion]
    J --> L[Metriques]
```

## Comparaison des modeles

| Modele | Accuracy test | Precision test | Recall test | F1 test | Balanced accuracy test |
|---|---:|---:|---:|---:|---:|
| Arbre de decision | {tree_test_scores["accuracy"]:.4%} | {tree_test_scores["precision"]:.4%} | {tree_test_scores["recall"]:.4%} | {tree_test_scores["f1"]:.4%} | {tree_test_scores["balanced_accuracy"]:.4%} |
| Regression logistique | {logistic_test_scores["accuracy"]:.4%} | {logistic_test_scores["precision"]:.4%} | {logistic_test_scores["recall"]:.4%} | {logistic_test_scores["f1"]:.4%} | {logistic_test_scores["balanced_accuracy"]:.4%} |
| Naive Bayes gaussien | {bayes_test_scores["accuracy"]:.4%} | {bayes_test_scores["precision"]:.4%} | {bayes_test_scores["recall"]:.4%} | {bayes_test_scores["f1"]:.4%} | {bayes_test_scores["balanced_accuracy"]:.4%} |

## Metriques validation arbre

| Metrique | Valeur |
|---|---:|
| Accuracy | {tree_validation_scores["accuracy"]:.4%} |
| Precision | {tree_validation_scores["precision"]:.4%} |
| Recall | {tree_validation_scores["recall"]:.4%} |
| Specificity | {tree_validation_scores["specificity"]:.4%} |
| F1 | {tree_validation_scores["f1"]:.4%} |
| Balanced accuracy | {tree_validation_scores["balanced_accuracy"]:.4%} |

## Matrice de confusion test arbre

| Metrique | Valeur |
|---|---:|
| TN | {tree_test_confusion["tn"]:,} |
| FP | {tree_test_confusion["fp"]:,} |
| FN | {tree_test_confusion["fn"]:,} |
| TP | {tree_test_confusion["tp"]:,} |

## Metriques test arbre

| Metrique | Valeur |
|---|---:|
| Accuracy | {tree_test_scores["accuracy"]:.4%} |
| Precision | {tree_test_scores["precision"]:.4%} |
| Recall | {tree_test_scores["recall"]:.4%} |
| Specificity | {tree_test_scores["specificity"]:.4%} |
| F1 | {tree_test_scores["f1"]:.4%} |
| Balanced accuracy | {tree_test_scores["balanced_accuracy"]:.4%} |

## Metriques validation regression logistique

| Metrique | Valeur |
|---|---:|
| Accuracy | {logistic_validation_scores["accuracy"]:.4%} |
| Precision | {logistic_validation_scores["precision"]:.4%} |
| Recall | {logistic_validation_scores["recall"]:.4%} |
| Specificity | {logistic_validation_scores["specificity"]:.4%} |
| F1 | {logistic_validation_scores["f1"]:.4%} |
| Balanced accuracy | {logistic_validation_scores["balanced_accuracy"]:.4%} |

## Matrice de confusion test regression logistique

| Metrique | Valeur |
|---|---:|
| TN | {logistic_test_confusion["tn"]:,} |
| FP | {logistic_test_confusion["fp"]:,} |
| FN | {logistic_test_confusion["fn"]:,} |
| TP | {logistic_test_confusion["tp"]:,} |

## Metriques test regression logistique

| Metrique | Valeur |
|---|---:|
| Accuracy | {logistic_test_scores["accuracy"]:.4%} |
| Precision | {logistic_test_scores["precision"]:.4%} |
| Recall | {logistic_test_scores["recall"]:.4%} |
| Specificity | {logistic_test_scores["specificity"]:.4%} |
| F1 | {logistic_test_scores["f1"]:.4%} |
| Balanced accuracy | {logistic_test_scores["balanced_accuracy"]:.4%} |

## Matrice de confusion test Naive Bayes gaussien

| Metrique | Valeur |
|---|---:|
| TN | {bayes_test_confusion["tn"]:,} |
| FP | {bayes_test_confusion["fp"]:,} |
| FN | {bayes_test_confusion["fn"]:,} |
| TP | {bayes_test_confusion["tp"]:,} |

## Metriques test Naive Bayes gaussien

| Metrique | Valeur |
|---|---:|
| Accuracy | {bayes_test_scores["accuracy"]:.4%} |
| Precision | {bayes_test_scores["precision"]:.4%} |
| Recall | {bayes_test_scores["recall"]:.4%} |
| Specificity | {bayes_test_scores["specificity"]:.4%} |
| F1 | {bayes_test_scores["f1"]:.4%} |
| Balanced accuracy | {bayes_test_scores["balanced_accuracy"]:.4%} |

## Fraude par type

| Type | Normal | Fraude |
|---|---:|---:|
{chr(10).join(f'| {row["type"]} | {row["normal"]:,} | {row["fraud"]:,} |' for row in fraud_by_type)}

## Structure du modele

{chr(10).join(render_tree(tree))}

## Coefficients regression logistique

| Variable | Poids |
|---|---:|
{chr(10).join(f'| {feature} | {weight:.6f} |' for feature, weight in zip(logistic_model.features, logistic_model.weights))}
| bias | {logistic_model.bias:.6f} |

## Centrage reduction

| Variable | Clip bas | Clip haut | Moyenne train | Ecart-type train |
|---|---:|---:|---:|---:|
{chr(10).join(f'| {stat.feature} | {stat.clip_low:.4f} | {stat.clip_high:.4f} | {stat.mean:.4f} | {stat.std:.4f} |' for stat in scaling_stats.values())}

## Fichiers generes

- `{REPORT_PATH}`
- `{TREE_PATH}`
- `{LOGISTIC_PATH}`
- `{BAYES_PATH}`
- `{METRICS_PATH}`
- `{CONFUSION_CSV_PATH}`
- `{CLASS_SVG_PATH}`
- `{TYPE_SVG_PATH}`
- `{CONFUSION_SVG_PATH}`
- `{TYPE_RATE_SVG_PATH}`
- `{STEP_RATE_SVG_PATH}`
- `{AMOUNT_RATE_SVG_PATH}`
- `{SIGNAL_RATE_SVG_PATH}`
"""
    REPORT_PATH.write_text(report.replace(",", " "), encoding="utf-8")