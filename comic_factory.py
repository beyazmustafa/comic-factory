# comic_factory.py

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920


def _open_rgb_image(image_path: str | Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    return image


def _fit_background(
    image: Image.Image,
    canvas_width: int,
    canvas_height: int,
    blur_radius: int = 36,
    darken_factor: float = 0.42,
    saturation_factor: float = 0.92,
) -> Image.Image:
    background = ImageOps.fit(
        image,
        (canvas_width, canvas_height),
        method=Image.Resampling.LANCZOS,
        bleed=0.0,
        centering=(0.5, 0.5),
    )

    background = background.filter(
        ImageFilter.GaussianBlur(radius=blur_radius)
    )

    background = ImageEnhance.Color(background).enhance(
        saturation_factor
    )

    background = ImageEnhance.Brightness(background).enhance(
        darken_factor
    )

    return background


def _resize_foreground_contain(
    image: Image.Image,
    max_width: int,
    max_height: int,
) -> Image.Image:
    foreground = image.copy()
    foreground.thumbnail(
        (max_width, max_height),
        Image.Resampling.LANCZOS,
    )
    return foreground


def _rounded_mask(
    size: tuple[int, int],
    radius: int,
) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(
        (0, 0, size[0] - 1, size[1] - 1),
        radius=radius,
        fill=255,
    )
    return mask


def _add_shadow(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    radius: int = 32,
    shadow_offset_y: int = 18,
    shadow_alpha: int = 120,
    blur_radius: int = 22,
) -> None:
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    x1, y1, x2, y2 = box
    shadow_box = (
        x1,
        y1 + shadow_offset_y,
        x2,
        y2 + shadow_offset_y,
    )

    draw.rounded_rectangle(
        shadow_box,
        radius=radius,
        fill=(0, 0, 0, shadow_alpha),
    )

    overlay = overlay.filter(
        ImageFilter.GaussianBlur(radius=blur_radius)
    )

    canvas.alpha_composite(overlay)


def compose_comic_frame(
    image_path: str | Path,
    output_path: str | Path | None = None,
    canvas_width: int = VIDEO_WIDTH,
    canvas_height: int = VIDEO_HEIGHT,
    side_padding: int = 56,
    top_safe_area: int = 70,
    bottom_safe_area: int = 330,
    panel_corner_radius: int = 28,
    panel_border_width: int = 4,
    panel_border_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Comic görselini dikey videoya kırpmadan yerleştirir.
    Ana panel tamamı görünür, boş alan blur arka planla doldurulur.
    """
    source = _open_rgb_image(image_path)

    background = _fit_background(
        image=source,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )

    available_width = canvas_width - (side_padding * 2)
    available_height = (
        canvas_height
        - top_safe_area
        - bottom_safe_area
    )

    foreground = _resize_foreground_contain(
        image=source,
        max_width=available_width,
        max_height=available_height,
    )

    foreground_width, foreground_height = foreground.size

    x = (canvas_width - foreground_width) // 2
    y = top_safe_area + (
        (available_height - foreground_height) // 2
    )

    canvas = background.convert("RGBA")
    _add_shadow(
        canvas=canvas,
        box=(x, y, x + foreground_width, y + foreground_height),
        radius=panel_corner_radius,
    )

    panel = Image.new(
        "RGBA",
        (foreground_width, foreground_height),
        (255, 255, 255, 0),
    )

    mask = _rounded_mask(
        (foreground_width, foreground_height),
        radius=panel_corner_radius,
    )

    panel.paste(
        foreground.convert("RGBA"),
        (0, 0),
        mask,
    )

    border_layer = Image.new(
        "RGBA",
        (foreground_width, foreground_height),
        (0, 0, 0, 0),
    )
    border_draw = ImageDraw.Draw(border_layer)
    border_draw.rounded_rectangle(
        (
            panel_border_width // 2,
            panel_border_width // 2,
            foreground_width - 1 - (panel_border_width // 2),
            foreground_height - 1 - (panel_border_width // 2),
        ),
        radius=panel_corner_radius,
        outline=panel_border_color + (255,),
        width=panel_border_width,
    )

    panel.alpha_composite(border_layer)
    canvas.alpha_composite(panel, (x, y))

    result = canvas.convert("RGB")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        result.save(output_path, quality=95)

    return result


def build_scene_image_for_video(
    source_image_path: str | Path,
    output_image_path: str | Path,
    subtitle_safe_area: int = 330,
) -> str:
    """
    Video sahnesinde kullanılacak görseli üretir.
    Bunu eski full-screen resize fonksiyonunun yerine kullan.
    """
    compose_comic_frame(
        image_path=source_image_path,
        output_path=output_image_path,
        canvas_width=VIDEO_WIDTH,
        canvas_height=VIDEO_HEIGHT,
        side_padding=56,
        top_safe_area=70,
        bottom_safe_area=subtitle_safe_area,
    )
    return str(output_image_path)


# --------------------------------------------------------------------
# ESKİ KULLANIMI BUNUNLA DEĞİŞTİR
# --------------------------------------------------------------------
#
# Eğer sende buna benzer bir şey varsa:
#
#     image = Image.open(scene_image_path).convert("RGB")
#     image = ImageOps.fit(image, (VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)
#     image.save(output_path)
#
# BUNUN YERİNE ŞUNU KULLAN:
#
#     build_scene_image_for_video(
#         source_image_path=scene_image_path,
#         output_image_path=output_path,
#         subtitle_safe_area=330,
#     )
#
# --------------------------------------------------------------------
# İSTEĞE BAĞLI TEST
# --------------------------------------------------------------------
#
# Tek bir görseli hızlı test etmek için:
#
# if __name__ == "__main__":
#     build_scene_image_for_video(
#         source_image_path="assets/test_comic.jpg",
#         output_image_path="assets/test_comic_vertical.jpg",
#         subtitle_safe_area=330,
#     )
#     print("Hazır: assets/test_comic_vertical.jpg")
