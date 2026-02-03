from __future__ import annotations

import os
import json
from datetime import datetime, timezone

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sqlalchemy import text

from shared.db import get_session
from shared.logging import setup_logger

logger = setup_logger("agent_v2_compare")

OUT_DIR = "agents/v2_compare/artifacts"

FEATURES = [
    "length_km",
    "mean_ele",
    "mean_slope",
    "mean_green",
    "frac_forest",
    "frac_water",
    "frac_road",
    "frac_building",
]


def load_dataset() -> pd.DataFrame:
    sess = get_session()
    q = """
    SELECT
      a.track_id,
      a.length_km,
      a.mean_ele,
      a.mean_slope,
      a.mean_green,
      a.frac_forest,
      a.frac_water,
      a.frac_road,
      a.frac_building,
      l.label
    FROM (
      SELECT
        dp.track_id,
        COUNT(*) * 0.02 AS length_km,
        AVG(dp.ele) AS mean_ele,
        AVG(dp.slope_deg) AS mean_slope,
        AVG(dp.green_index) AS mean_green,
        AVG(dp.frac_vegetation) AS frac_forest,
        AVG(dp.frac_water) AS frac_water,
        AVG(dp.frac_road) AS frac_road,
        AVG(dp.frac_building) AS frac_building
      FROM dataset_points dp
      GROUP BY dp.track_id
    ) a
    JOIN track_labels l ON l.track_id = a.track_id
    """
    return pd.read_sql(text(q), sess.bind)


def train_rf(X_train, y_train, X_test, y_test) -> tuple[dict, object, pd.Series]:
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    res = {
        "model": "RandomForest",
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }

    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return res, model, importances


def train_logreg(X_train, y_train, X_test, y_test) -> tuple[dict, object]:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    res = {
        "model": "LogisticRegression",
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }
    return res, pipe


def save_plots(
    cmp_df: pd.DataFrame,
    rf_importances: pd.Series | None,
    y_test: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.DataFrame | None,
) -> dict:
    """
    Рисует PNG-графики в OUT_DIR.
    Без seaborn, только matplotlib (дефолтные цвета).
    """
    import matplotlib.pyplot as plt

    artifacts: dict = {}

    # 1) Сравнение моделей
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = range(len(cmp_df))
    ax.bar([i - 0.15 for i in x], cmp_df["accuracy"], width=0.3, label="accuracy")
    ax.bar([i + 0.15 for i in x], cmp_df["macro_f1"], width=0.3, label="macro_f1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cmp_df["model"], rotation=20, ha="right")
    ax.set_title("Model comparison")
    ax.legend()
    p = os.path.join(OUT_DIR, "model_comparison.png")
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    artifacts["model_comparison_png"] = "model_comparison.png"

    # 2) Feature importance (RF)
    if rf_importances is not None and len(rf_importances) > 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(rf_importances.index.tolist(), rf_importances.values.tolist())
        ax.set_title("RandomForest feature importance")
        ax.set_xticklabels(rf_importances.index.tolist(), rotation=25, ha="right")
        p = os.path.join(OUT_DIR, "feature_importance_rf.png")
        fig.tight_layout()
        fig.savefig(p, dpi=160)
        plt.close(fig)
        artifacts["feature_importance_rf_png"] = "feature_importance_rf.png"

    # 3) Confusion matrix (если классов >= 2)
    if y_test.nunique() >= 2:
        labels = sorted(pd.unique(pd.concat([y_test, y_pred])))
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(cm)
        ax.set_title("Confusion matrix")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax)
        p = os.path.join(OUT_DIR, "confusion_matrix.png")
        fig.tight_layout()
        fig.savefig(p, dpi=160)
        plt.close(fig)
        artifacts["confusion_matrix_png"] = "confusion_matrix.png"

    # 4) ROC-AUC (если есть вероятности и классов >= 2)
    if y_proba is not None and y_test.nunique() >= 2:
        classes = sorted(y_proba.columns.tolist())
        y_true_bin = label_binarize(y_test, classes=classes)

        try:
            auc_macro = roc_auc_score(y_true_bin, y_proba.values, average="macro", multi_class="ovr")
        except Exception:
            auc_macro = None

        # График: просто значения вероятностей/auc текстом (ROC-кривые на 5 треках будут выглядеть странно,
        # но файл всё равно полезен для демонстрации)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.02, 0.8, f"ROC-AUC macro (ovr): {auc_macro}", fontsize=12)
        ax.text(0.02, 0.6, "Note: small sample size, ROC is unstable.", fontsize=10)
        p = os.path.join(OUT_DIR, "roc_auc.png")
        fig.tight_layout()
        fig.savefig(p, dpi=160)
        plt.close(fig)
        artifacts["roc_auc_png"] = "roc_auc.png"

    return artifacts


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_dataset()
    logger.info(f"dataset loaded: {df.shape}")

    if df.shape[0] < 3:
        logger.warning("Too few samples for comparison")
        return

    X = df[FEATURES].copy()
    y = df["label"].astype(str)

    X = X.fillna(X.median(numeric_only=True))

    n_classes = int(y.nunique())
    logger.info(f"unique classes: {n_classes}")

    stratify = y if n_classes >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=stratify
    )

    results: list[dict] = []

    rf_res, rf_model, rf_importances = train_rf(X_train, y_train, X_test, y_test)
    results.append(rf_res)

    lr_model = None
    lr_res = None
    if n_classes >= 2:
        lr_res, lr_model = train_logreg(X_train, y_train, X_test, y_test)
        results.append(lr_res)
    else:
        logger.warning(
            "Only one class present in labels. "
            "Skipping Logistic Regression; evaluating RandomForest only."
        )

    cmp_df = pd.DataFrame(results)
    cmp_df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False, encoding="utf-8")

    # RF importances csv
    imp_path = None
    if rf_importances is not None and len(rf_importances) > 0:
        imp_df = rf_importances.reset_index()
        imp_df.columns = ["feature", "importance"]
        imp_path = os.path.join(OUT_DIR, "feature_importance_rf.csv")
        imp_df.to_csv(imp_path, index=False, encoding="utf-8")

    # Predictions for plots
    y_pred_rf = pd.Series(rf_model.predict(X_test), index=X_test.index)

    # Proba (если классов >= 2)
    y_proba = None
    if n_classes >= 2 and hasattr(rf_model, "predict_proba"):
        proba = rf_model.predict_proba(X_test)
        classes = rf_model.classes_.tolist()
        y_proba = pd.DataFrame(proba, columns=classes)

    plot_artifacts = save_plots(
        cmp_df=cmp_df,
        rf_importances=rf_importances,
        y_test=y_test.reset_index(drop=True),
        y_pred=y_pred_rf.reset_index(drop=True),
        y_proba=y_proba,
    )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(df.shape[0]),
        "features": FEATURES,
        "unique_classes": n_classes,
        "models": [{"name": r["model"], "accuracy": r["accuracy"], "macro_f1": r["macro_f1"]} for r in results],
        "artifacts": {
            "model_comparison_csv": "model_comparison.csv",
            "feature_importance_rf_csv": "feature_importance_rf.csv" if imp_path else None,
            **plot_artifacts,
        },
        "chosen_model": "RandomForest",
        "reason": (
            "Базовая модель для классификации. "
            "Часть моделей пропускается автоматически, если в данных недостаточно классов."
        ),
        "notes": (
            "На малом числе треков метрики нестабильны. "
            "Для стабильной оценки требуется больше разнообразных меток маршрутов."
        ),
    }

    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("comparison artifacts saved")


if __name__ == "__main__":
    main()
