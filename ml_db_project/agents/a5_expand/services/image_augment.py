from __future__ import annotations

import os
import random
from PIL import Image, ImageEnhance, ImageOps


def aug_one(img: Image.Image, seed: int) -> Image.Image:
    rnd = random.Random(seed)

    x = img.convert("RGB")

    # геометрия
    if rnd.random() < 0.5:
        x = ImageOps.mirror(x)
    if rnd.random() < 0.2:
        x = ImageOps.flip(x)

    # небольшой поворот
    angle = rnd.uniform(-8, 8)
    x = x.rotate(angle, resample=Image.BICUBIC, expand=False)

    # цвет/контраст/яркость
    x = ImageEnhance.Color(x).enhance(rnd.uniform(0.85, 1.15))
    x = ImageEnhance.Contrast(x).enhance(rnd.uniform(0.85, 1.20))
    x = ImageEnhance.Brightness(x).enhance(rnd.uniform(0.90, 1.15))

    return x


def augment_images(input_dir: str, out_dir: str, n_per_image: int = 3) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    produced: list[str] = []

    for fn in os.listdir(input_dir):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        src = os.path.join(input_dir, fn)
        try:
            base = Image.open(src)
        except Exception:
            continue

        name, ext = os.path.splitext(fn)
        for i in range(n_per_image):
            out = aug_one(base, seed=hash((fn, i)) & 0xFFFFFFFF)
            out_fn = f"{name}__aug{i}{ext}"
            out_path = os.path.join(out_dir, out_fn)
            out.save(out_path, quality=95)
            produced.append(out_path)

    return produced


def try_generate_diffusion_stub(input_dir: str, out_dir: str, max_images: int = 5) -> tuple[bool, str, list[str]]:
    """
    Опционально: если есть torch+diffusers, сделаем генерацию вариаций.
    Если нет — вернём skipped и причину.
    """
    os.makedirs(out_dir, exist_ok=True)
    produced: list[str] = []

    try:
        import torch  # noqa
        from diffusers import StableDiffusionImg2ImgPipeline  # noqa
    except Exception as e:
        return False, f"skipped: missing torch/diffusers ({e})", produced

    # Если библиотека есть, но модель не скачана — это может занять время/интернет.
    # Поэтому делаем максимально аккуратно: один пайплайн, малая итерация, мало картинок.
    try:
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
    except Exception as e:
        return False, f"skipped: cannot init diffusion pipeline ({e})", produced

    prompt = "realistic satellite landscape variation, same geography, natural colors"
    strength = 0.35

    count = 0
    for fn in os.listdir(input_dir):
        if count >= max_images:
            break
        if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        src = os.path.join(input_dir, fn)
        try:
            init_image = Image.open(src).convert("RGB")
        except Exception:
            continue

        try:
            res = pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                guidance_scale=6.5,
                num_inference_steps=20,
            ).images[0]
            out_path = os.path.join(out_dir, f"{os.path.splitext(fn)[0]}__gen.png")
            res.save(out_path)
            produced.append(out_path)
            count += 1
        except Exception:
            continue

    return True, "ok", produced
