from __future__ import annotations

import argparse
import os
import json
from datetime import datetime, timezone

import pandas as pd
import joblib

from shared.logging import setup_logger

from agents.a3_preprocess.services.io import load_dataset_points, FEATURE_COLS
from agents.a3_preprocess.services.vif import compute_vif
from agents.a3_preprocess.services.feature_select import rf_importance, mi_importance, pick_top_features
from agents.a3_preprocess.services.pipeline import build_preprocess_pipeline

logger = setup_logger("agent_a3_preprocess")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, default="land_type", help="target column for feature selection")
    ap.add_argument("--top_k", type=int, default=8, help="how many features keep after selection")
    ap.add_argument("--balance", type=str, default="none", choices=["none", "smote", "adasyn"])
    ap.add_argument("--method", type=str, default="rf", choices=["rf", "mi"])
    ap.add_argument("--out_dir", type=str, default="agents/a3_preprocess/artifacts")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_dataset_points()
    if df.empty:
        raise RuntimeError("dataset_points is empty. Run 1.1 and 1.2 first.")

    # target
    if args.target not in df.columns:
        raise ValueError(f"target '{args.target}' not found in dataset")

    # чистим таргет: убираем None/unknown, чтобы отбор признаков был осмысленный
    work = df.copy()
    work = work.dropna(subset=[args.target])
    if args.target == "land_type":
        work = work[work["land_type"].astype(str).str.lower() != "unknown"]

    # фичи
    X = work[FEATURE_COLS].copy()
    y = work[args.target].astype(str).copy()

    # VIF на сырых числах (после удаления NaN строк)
    vif_df = compute_vif(X)
    vif_path = os.path.join(args.out_dir, "vif.csv")
    vif_df.to_csv(vif_path, index=False)

    # важность
    X_num = X.copy()
    X_num = X_num.fillna(X_num.median(numeric_only=True))

    if args.method == "rf":
        imp = rf_importance(X_num, y)
    else:
        imp = mi_importance(X_num, y)

    imp_path = os.path.join(args.out_dir, f"importance_{args.method}.csv")
    imp.to_csv(imp_path, index=False)

    selected = pick_top_features(imp, args.top_k)

    # pipeline
    balance = None if args.balance == "none" else args.balance
    pipe = build_preprocess_pipeline(feature_cols=selected, balance=balance)

    pipe_path = os.path.join(args.out_dir, "preprocess_pipeline.pkl")
    joblib.dump(pipe, pipe_path)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows_used": int(len(work)),
        "target": args.target,
        "method": args.method,
        "top_k": int(args.top_k),
        "selected_features": selected,
        "balance": args.balance,
        "artifacts": {
            "vif_csv": "vif.csv",
            "importance_csv": os.path.basename(imp_path),
            "pipeline_pkl": "preprocess_pipeline.pkl",
        },
    }
    meta_path = os.path.join(args.out_dir, "run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(f"saved: {vif_path}")
    logger.info(f"saved: {imp_path}")
    logger.info(f"saved: {pipe_path}")
    logger.info(f"saved: {meta_path}")
    logger.info(f"selected_features={selected}")


if __name__ == "__main__":
    main()
