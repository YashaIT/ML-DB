from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from shared.db import engine, get_session
from shared.logging import setup_logger
from shared.models import Base, Track, TrackPoint

from agents.a5_expand.services.osm_overpass import fetch_hiking_ways
from agents.a5_expand.services.synth_tracks import (
    SynthPoint,
    inject_time,
    jitter_points,
    extreme_weather_profile,
    obstacle_flags,
)
from agents.a5_expand.services.image_augment import augment_images, try_generate_diffusion_stub
from agents.a5_expand.services.fid_metrics import compute_fid, list_images
from agents.a5_expand.services.render_track_images import render_track_images

logger = setup_logger("agent_a5_expand")


def ensure_db() -> None:
    Base.metadata.create_all(bind=engine)


def main() -> None:
    ap = argparse.ArgumentParser()

    # OSM загрузка
    ap.add_argument("--region", type=str, default="OSM")
    ap.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("S", "W", "N", "E"),
        default=[59.85, 55.75, 60.25, 56.60],
    )
    ap.add_argument("--new_tracks", type=int, default=5)

    # синтетика
    ap.add_argument("--synth_per_track", type=int, default=1)
    ap.add_argument("--noise_m", type=float, default=20.0)

    # изображения
    ap.add_argument("--images_in", type=str, default="data/images")
    ap.add_argument("--aug_n", type=int, default=3)

    # выход
    ap.add_argument("--out_dir", type=str, default="agents/a5_expand/artifacts")

    args = ap.parse_args()

    ensure_db()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("agents/a5_expand/out/images_aug", exist_ok=True)
    os.makedirs("agents/a5_expand/out/images_gen", exist_ok=True)

    sess = get_session()

    # 1) новые треки из OSM (Overpass)
    ways = fetch_hiking_ways(tuple(args.bbox), limit=args.new_tracks)
    logger.info(f"OSM ways fetched: {len(ways)}")

    created_tracks = 0
    created_points = 0
    created_synth_tracks = 0

    for w in ways:
        geom = w["geometry"]

        track = Track(
            source="osm_overpass",
            source_id=str(w.get("osm_id")),
            region=args.region,
            started_at=datetime.now(timezone.utc),
        )
        sess.add(track)
        sess.flush()  # получить track.id

        pts: list[SynthPoint] = []
        for g in geom:
            pts.append(SynthPoint(lat=float(g["lat"]), lon=float(g["lon"]), ele=None, ts=None))

        # проставим время, если нет
        pts = inject_time(pts, start=datetime.now(timezone.utc), step_sec=5)

        for i, p in enumerate(pts):
            tp = TrackPoint(
                track_id=track.id,
                seq=i,
                lat=p.lat,
                lon=p.lon,
                ele=p.ele,
                ts=p.ts,
            )
            sess.add(tp)
            created_points += 1

        created_tracks += 1

        # 2) синтетические треки с аномалиями
        for k in range(args.synth_per_track):
            syn = jitter_points(pts, noise_m=args.noise_m, seed=(hash((track.id, k)) & 0xFFFFFFFF))

            syn_track = Track(
                source="synthetic",
                source_id=f"syn_from_osm_{w.get('osm_id')}_{k}",
                region=f"{args.region}_SYN",
                started_at=datetime.now(timezone.utc),
                meta_json=json.dumps(
                    {
                        "scenario": extreme_weather_profile(seed=100 + k),
                        "obstacles": obstacle_flags(seed=200 + k),
                        "noise_m": args.noise_m,
                        "parent_osm_id": w.get("osm_id"),
                    },
                    ensure_ascii=False,
                ),
            )
            sess.add(syn_track)
            sess.flush()

            for i, p in enumerate(syn):
                tp = TrackPoint(
                    track_id=syn_track.id,
                    seq=i,
                    lat=p.lat,
                    lon=p.lon,
                    ele=p.ele,
                    ts=p.ts,
                )
                sess.add(tp)
                created_points += 1

            created_synth_tracks += 1

    sess.commit()
    logger.info(
        f"created tracks={created_tracks}, synthetic_tracks={created_synth_tracks}, points={created_points}"
    )

    # 3) изображения: если папка пустая — генерим базовые изображения треков
    os.makedirs(args.images_in, exist_ok=True)
    if len([f for f in os.listdir(args.images_in) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) == 0:
        base_imgs = render_track_images(args.images_in, max_tracks=20, size=512)
        logger.info(f"generated base track images: {len(base_imgs)} -> {args.images_in}")

    # 4) обычная аугментация изображений
    aug_paths = augment_images(args.images_in, "agents/a5_expand/out/images_aug", n_per_image=args.aug_n)
    logger.info(f"augmented images: {len(aug_paths)}")

    # 5) генеративная аугментация (опционально)
    gen_ok, gen_status, gen_paths = try_generate_diffusion_stub(
        input_dir=args.images_in,
        out_dir="agents/a5_expand/out/images_gen",
        max_images=5,
    )
    if gen_ok:
        logger.info(f"generative images: {len(gen_paths)}")
    else:
        logger.warning(f"generative skipped: {gen_status}")

    # 6) FID (best-effort: cleanfid -> simplefid)
    real_dir = args.images_in
    fake_dir = "agents/a5_expand/out/images_aug"
    fid_score, fid_status = compute_fid(real_dir, fake_dir)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "osm_bbox": args.bbox,
        "new_tracks_requested": args.new_tracks,
        "synth_per_track": args.synth_per_track,
        "noise_m": args.noise_m,
        "images": {
            "input_dir": args.images_in,
            "real_count": len(list_images(real_dir)),
            "aug_dir": fake_dir,
            "aug_count": len(list_images(fake_dir)),
            "gen_dir": "agents/a5_expand/out/images_gen",
            "gen_count": len(list_images("agents/a5_expand/out/images_gen")),
        },
        "generative": {
            "status": gen_status,
            "enabled": bool(gen_ok),
        },
        "fid": {
            "score": fid_score,
            "status": fid_status,
        },
    }

    meta_path = os.path.join(args.out_dir, "run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(f"saved meta: {meta_path}")


if __name__ == "__main__":
    main()
