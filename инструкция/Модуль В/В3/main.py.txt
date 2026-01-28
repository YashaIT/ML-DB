from __future__ import annotations

import os
import json
from datetime import datetime, timezone

import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from sqlalchemy import text

from shared.db import get_session
from shared.logging import setup_logger

logger = setup_logger("agent_v3_evaluate")

OUT_DIR = "agents/v3_evaluate/artifacts"

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


def save_confusion_png(y_true: pd.Series, y_pred: pd.Series, out_path: str) -> None:
    import matplotlib.pyplot as plt

    labels = sorted(pd.unique(pd.concat([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    ax.set_title("Confusion matrix")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_dataset()
    logger.info(f"dataset loaded: {df.shape}")

    df.to_csv(os.path.join(OUT_DIR, "dataset_tracks.csv"), index=False, encoding="utf-8")

    if df.shape[0] < 3:
        meta = {
            "status": "too_few_samples",
            "rows": int(df.shape[0]),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(os.path.join(OUT_DIR, "evaluation.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return

    X = df[FEATURES].copy().fillna(df[FEATURES].median(numeric_only=True))
    y = df["label"].astype(str)
    n_classes = int(y.nunique())

    meta: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(df.shape[0]),
        "unique_classes": n_classes,
        "features": FEATURES,
        "cv": None,
        "holdout": None,
        "notes": None,
    }

    if n_classes < 2:
        meta["status"] = "single_class"
        meta["notes"] = (
            "В наборе данных присутствует только один класс. "
            "CV и confusion matrix для классификации неинформативны. "
            "Рекомендуется увеличить число треков и разнообразие меток."
        )
        with open(os.path.join(OUT_DIR, "evaluation.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.warning("single-class dataset: evaluation limited")
        return

    # модель для оценки
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )

    # --- CV (StratifiedKFold)
    k = min(3, df.shape[0])  # на малых данных не делаем много фолдов
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    cv_rows = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

        model.fit(Xtr, ytr)
        pred = model.predict(Xte)

        cv_rows.append(
            {
                "fold": fold,
                "accuracy": float(accuracy_score(yte, pred)),
                "macro_f1": float(f1_score(yte, pred, average="macro")),
            }
        )

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(os.path.join(OUT_DIR, "cv_metrics.csv"), index=False, encoding="utf-8")

    meta["cv"] = {
        "folds": k,
        "accuracy_mean": float(cv_df["accuracy"].mean()),
        "macro_f1_mean": float(cv_df["macro_f1"].mean()),
    }

    # --- Holdout + confusion matrix
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    y_pred = pd.Series(model.predict(X_test))

    holdout = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
    }
    meta["holdout"] = holdout

    cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    save_confusion_png(y_test.reset_index(drop=True), y_pred.reset_index(drop=True), cm_path)

    with open(os.path.join(OUT_DIR, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("evaluation artifacts saved")


if __name__ == "__main__":
    main()
