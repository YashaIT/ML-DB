from __future__ import annotations

import argparse
import os
import json
from datetime import datetime, timezone

from shared.logging import setup_logger

from agents.a4_dataset_report.services.load import load_dataset_points, FEATURE_COLS
from agents.a4_dataset_report.services.dictionary import dataset_dictionary, render_markdown_table
from agents.a4_dataset_report.services.map_folium import build_tracks_map
from agents.a4_dataset_report.services.stats_tests import pick_two_regions, run_region_tests
from agents.a4_dataset_report.services.profiling import make_profiling_report

logger = setup_logger("agent_a4_dataset_report")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="agents/a4_dataset_report/artifacts")
    ap.add_argument("--region_a", type=str, default="")
    ap.add_argument("--region_b", type=str, default="")
    ap.add_argument("--profile_rows_limit", type=int, default=20000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_dataset_points()
    if df.empty:
        raise RuntimeError("dataset_points is empty. Run 1.1 and 1.2 first.")

    # 1) Data dictionary
    dict_md = render_markdown_table(dataset_dictionary())
    dict_path = os.path.join(args.out_dir, "data_dictionary.md")
    with open(dict_path, "w", encoding="utf-8") as f:
        f.write("# Описание структуры набора данных\n\n")
        f.write(dict_md)
        f.write("\n")

    # 2) Комплексный авто-отчёт (profiling) — HTML
    profile_path = os.path.join(args.out_dir, "profiling_report.html")
    tool_name, tool_status = make_profiling_report(df, profile_path)

    # 3) Карта треков
    map_path = os.path.join(args.out_dir, "tracks_map.html")
    build_tracks_map(df, map_path)

    # 4) Стат. тесты по регионам (если есть >=2 региона)
    region_a = args.region_a.strip()
    region_b = args.region_b.strip()

    if not region_a or not region_b:
        picked = pick_two_regions(df)
        if picked is None:
            region_a = region_b = ""
            logger.warning("region tests skipped: less than 2 regions in data")
        else:
            region_a, region_b = picked

    tests_path = None
    if region_a and region_b and region_a != region_b:
        tests = run_region_tests(df, FEATURE_COLS, region_a, region_b)
        tests_path = os.path.join(args.out_dir, "region_tests.csv")
        tests.to_csv(tests_path, index=False)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows_total": int(len(df)),
        "artifacts": {
            "data_dictionary_md": "data_dictionary.md",
            "profiling_report_html": "profiling_report.html" if tool_status == "ok" else None,
            "tracks_map_html": "tracks_map.html",
            "region_tests_csv": "region_tests.csv" if tests_path else None,
        },
        "profiling": {
            "tool": tool_name,
            "status": tool_status,
            "rows_limit": int(args.profile_rows_limit),
        },
        "region_test_pair": {"region_a": region_a or None, "region_b": region_b or None},
    }

    meta_path = os.path.join(args.out_dir, "run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(f"saved: {dict_path}")
    if tool_status == "ok":
        logger.info(f"saved: {profile_path} (tool={tool_name})")
    else:
        logger.warning(f"profiling not generated: {tool_status}")
    logger.info(f"saved: {map_path}")
    if tests_path:
        logger.info(f"saved: {tests_path}")
    logger.info(f"saved: {meta_path}")


if __name__ == "__main__":
    main()
