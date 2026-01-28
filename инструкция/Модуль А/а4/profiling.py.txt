from __future__ import annotations

import pandas as pd


def make_profiling_report(df: pd.DataFrame, out_html: str) -> tuple[str, str]:
    """
    Возвращает (tool_name, status)
    tool_name: "ydata-profiling" | "sweetviz" | "none"
    status: "ok" | "skipped:<reason>"
    """
    # Ограничим объём, чтобы отчёт стабильно строился
    prof_df = df.copy()
    if len(prof_df) > 20000:
        prof_df = prof_df.sample(20000, random_state=42)

    # 1) Основной: ydata-profiling
    try:
        from ydata_profiling import ProfileReport

        profile = ProfileReport(
            prof_df,
            title="Dataset profiling report",
            explorative=True,
            minimal=True,  # стабильнее и быстрее
        )
        profile.to_file(out_html)
        return "ydata-profiling", "ok"
    except Exception as e1:
        # 2) Fallback: sweetviz
        try:
            import sweetviz as sv

            report = sv.analyze(prof_df)
            report.show_html(out_html, open_browser=False)
            return "sweetviz", "ok"
        except Exception as e2:
            return "none", f"skipped:ydata-profiling={e1}; sweetviz={e2}"
