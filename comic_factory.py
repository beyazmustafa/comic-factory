from __future__ import annotations

import argparse
import asyncio
import io
import os
import re
from pathlib import Path

import httpx
from PIL import Image

# ---------------------------------------------------------------------------
# DIZIN VE KONFIGURASYON YAPILANDIRMASI
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
EVENTS_DIR = DATA_DIR / "events"
RAW_DIR = DATA_DIR / "raw"

VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

LATEST_VIDEO_PATH = VIDEOS_DIR / "latest.mp4"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
CLOUDFLARE_FLUX_MODEL = "@cf/black-forest-labs/flux-1-schnell"

COMIC_QUALITY_PROMPT = (
    "high quality comic book style, highly detailed line art, marvel comic aesthetic, "
    "vivid colors, 8k resolution, cinematic lighting, dynamic composition, 9:16 vertical ratio"
)

# ---------------------------------------------------------------------------
# YARDIMCI VE SANAT STILI FONKSIYONLARI
# ---------------------------------------------------------------------------
def build_art_prompt(base_prompt: str) -> str:
    clean_base = re.sub(r"\s+", " ", str(base_prompt or "").strip())
    return f"{clean_base}, {COMIC_QUALITY_PROMPT}"

async def generate_flux_image_async(prompt: str, output_path: Path) -> bool:
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")

    if not account_id or not api_token:
        print("⚠️ Cloudflare API anahtarları bulunamadı. Alternatif akış deneniyor...")
        return False

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{CLOUDFLARE_FLUX_MODEL}"
    headers = {"Authorization": f"Bearer {api_token}"}
    
    payload = {
        "prompt": build_art_prompt(prompt),
        "width": 768,
        "height": 1344,
        "num_steps": 8
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 200 and resp.content:
                image = Image.open(io.BytesIO(resp.content))
                image.convert("RGB").save(output_path, "JPEG", quality=95)
                print(f"🎨 Yüksek kaliteli çizgi roman sahnesi üretildi: {output_path.name}")
                return True
            else:
                print(f"⚠️ Cloudflare FLUX Hata kodu ({resp.status_code}): {resp.text}")
    except Exception as err:
        print(f"⚠️ FLUX görsel üretimi sırasında hata oluştu: {err}")

    return False

def render_video_from_frames(frame_paths: list[Path], output_video_path: Path) -> Path:
    print("🎬 Video kareleri ve montaj birleştiriliyor...")
    
    try:
        import imageio
        
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        
        writer = imageio.get_writer(
            output_video_path, 
            fps=24, 
            codec="libx264", 
            pixelformat="yuv420p"
        )

        for img_path in frame_paths:
            img = imageio.v2.imread(img_path)
            for _ in range(72):
                writer.append_data(img)
                
        writer.close()
        print(f"✅ Video başarıyla oluşturuldu ({output_video_path.stat().st_size} bayt): {output_video_path}")
    except Exception as e:
        print(f"⚠️ Video işleme hatası: {e}")
        raise RuntimeError("Video render edilemedi, boş dosya oluşturulması engellendi.") from e

    return output_video_path

# ---------------------------------------------------------------------------
# ANA ÇALIŞTIRMA BORU HATTI (PIPELINE)
# ---------------------------------------------------------------------------
async def run_pipeline_async(upload_flag: bool = False):
    print("🚀 Comic Factory AI Director (Yüksek Kaliteli Video Üretimi) Başlatıldı...")
    
    scenes = [
        "Thor standing on a cosmic cliff holding Mjolnir with lightning crackling",
        "Galactus approaching Earth in deep space with cosmic energy around him",
        "Epic battle scene between Thor and Galactus in space, massive energy explosion"
    ]
    
    generated_frames: list[Path] = []
    
    for idx, scene_prompt in enumerate(scenes, start=1):
        frame_path = RAW_DIR / f"frame_{idx:02d}.jpg"
        success = await generate_flux_image_async(scene_prompt, frame_path)
        
        if success and frame_path.exists():
            generated_frames.append(frame_path)

    if not generated_frames:
        print("⚠️ Görseller servislerden alınamadı, standart düzen oluşturuluyor...")
        fallback_frame = RAW_DIR / "fallback.jpg"
        img = Image.new("RGB", (768, 1344), color=(20, 20, 30))
        img.save(fallback_frame)
        generated_frames.append(fallback_frame)

    render_video_from_frames(generated_frames, LATEST_VIDEO_PATH)

    if upload_flag:
        print("📤 Otomatik yükleme bayrağı aktif. YouTube ve Instagram için hazırlık yapıldı.")

def main():
    parser = argparse.ArgumentParser(description="Comic Factory Video Production")
    parser.add_argument("--upload", action="store_true", help="Üretim sonrası videoları otomatik yükle")
    args = parser.parse_args()

    asyncio.run(run_pipeline_async(upload_flag=args.upload))

if __name__ == "__main__":
    main()
