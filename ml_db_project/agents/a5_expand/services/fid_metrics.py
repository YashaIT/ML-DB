from __future__ import annotations

import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA


def list_images(d: str) -> list[str]:
    if not os.path.isdir(d):
        return []
    return [
        os.path.join(d, f)
        for f in os.listdir(d)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


def _img_to_feat(path: str, bins: int = 32) -> np.ndarray | None:
    try:
        img = Image.open(path).convert("RGB").resize((128, 128))
        arr = np.asarray(img, dtype=np.float32) / 255.0
    except Exception:
        return None

    feats = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=bins, range=(0.0, 1.0), density=True)
        feats.append(h.astype(np.float32))
    return np.concatenate(feats, axis=0)


def _frechet_gaussian(mu1: np.ndarray, cov1: np.ndarray, mu2: np.ndarray, cov2: np.ndarray) -> float:
    eps = 1e-6
    cov1 = cov1 + np.eye(cov1.shape[0]) * eps
    cov2 = cov2 + np.eye(cov2.shape[0]) * eps

    prod = cov1 @ cov2
    prod = (prod + prod.T) / 2.0
    vals, vecs = np.linalg.eigh(prod)
    vals = np.clip(vals, 0.0, None)
    sqrt_prod = vecs @ np.diag(np.sqrt(vals)) @ vecs.T

    diff = mu1 - mu2
    tr = np.trace(cov1 + cov2 - 2.0 * sqrt_prod)
    return float(diff @ diff + tr)


def compute_fid(real_dir: str, fake_dir: str) -> tuple[float | None, str]:
    """
    1) Пытаемся cleanfid (если вдруг доступен)
    2) Иначе считаем SimpleFID: Frechet distance на PCA-признаках RGB-гистограмм
    """
    # cleanfid (best-effort)
    try:
        from cleanfid import fid  # type: ignore

        score = float(fid.compute_fid(real_dir, fake_dir))
        return score, "ok:cleanfid"
    except Exception:
        pass

    real = list_images(real_dir)
    fake = list_images(fake_dir)
    if len(real) < 5 or len(fake) < 5:
        return None, "skipped:simplefid_not_enough_images"

    real_feats = []
    for p in real[:300]:
        f = _img_to_feat(p)
        if f is not None:
            real_feats.append(f)

    fake_feats = []
    for p in fake[:300]:
        f = _img_to_feat(p)
        if f is not None:
            fake_feats.append(f)

    if len(real_feats) < 5 or len(fake_feats) < 5:
        return None, "skipped:simplefid_not_enough_valid_images"

    Xr = np.stack(real_feats, axis=0)
    Xf = np.stack(fake_feats, axis=0)

    # n_components не может быть больше числа примеров и числа признаков
    n_comp = min(32, Xr.shape[1], Xr.shape[0], Xf.shape[0])
    if n_comp < 2:
        return None, "skipped:simplefid_not_enough_samples_for_pca"

    pca = PCA(n_components=n_comp, random_state=42)
    Zr = pca.fit_transform(Xr)
    Zf = pca.transform(Xf)

    mu1, mu2 = Zr.mean(axis=0), Zf.mean(axis=0)
    cov1 = np.cov(Zr, rowvar=False)
    cov2 = np.cov(Zf, rowvar=False)

    score = _frechet_gaussian(mu1, cov1, mu2, cov2)
    return score, "ok:simplefid"
