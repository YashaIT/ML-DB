from __future__ import annotations

import argparse
import os

from shared.logging import setup_logger
from agents.b2_labeling.services.labeling import (
    LabelingConfig,
    load_track_aggregates,
    cluster_and_label,
    save_labels_to_db,
    export_active_csv,
    import_review_csv,
)

logger = setup_logger("agent_b2_labeling")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--active_n", type=int, default=10)
    ap.add_argument("--out_dir", type=str, default="agents/b2_labeling/artifacts")
    ap.add_argument(
        "--import_review_csv",
        type=str,
        default="",
        help="Если указан путь — импортирует review_label и перезапишет метки как manual",
    )
    args = ap.parse_args()

    cfg = LabelingConfig(k=args.k, active_n=args.active_n, out_dir=args.out_dir)
    os.makedirs(cfg.out_dir, exist_ok=True)

    if args.import_review_csv:
        df_manual = import_review_csv(args.import_review_csv)
        n = save_labels_to_db(df_manual)
        logger.info(f"manual labels imported: {n}")
        return

    df = load_track_aggregates()
    logger.info(f"tracks aggregated: {len(df)}")

    labeled, active = cluster_and_label(df, cfg)
    n = save_labels_to_db(labeled[["track_id", "label", "method", "confidence"]])
    logger.info(f"labels saved: {n}")

    active_path = os.path.join(cfg.out_dir, "active_review.csv")
    export_active_csv(active, active_path)
    logger.info(f"active learning export: {active_path} (edit review_label and re-import)")

    labeled_path = os.path.join(cfg.out_dir, "labels_all.csv")
    labeled.to_csv(labeled_path, index=False, encoding="utf-8")
    logger.info(f"labels_all export: {labeled_path}")


if __name__ == "__main__":
    main()
