import os
import json
import re
import time
from pathlib import Path
from google import genai
from google.genai import types
import requests

# ---------------------------------------------------------------------------
# AYARLAR VE KALİTE PARAMETRELERİ
# ---------------------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CLOUDFLARE_FLUX_MODEL = "@cf/black-forest-labs/flux-1-schnell"

# Yapay zekanın kaliteli çizgi roman görseli üretmesini sağlayan sihirli kalıp:
COMIC_STYLE_PROMPT = (
    "high quality comic book style, highly detailed line art, dynamic action shot, "
    "marvel comic aesthetic, vivid colors, 8k resolution, cinematic lighting"
)

def build_quality_prompt(user_prompt: str) -> str:
    """Görsel istemini kalite odaklı hale getirir."""
    return f"{user_prompt}, {COMIC_STYLE_PROMPT}"

def generate_flux_image(prompt: str) -> bytes:
    """Cloudflare FLUX modeli ile yüksek kaliteli dikey görsel üretir."""
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")

    if not account_id or not api_token:
        raise RuntimeError("Cloudflare API anahtarları GitHub Secrets içinde bulunamadı.")

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{CLOUDFLARE_FLUX_MODEL}"
    headers = {"Authorization": f"Bearer {api_token}"}
    
    # 9:16 Dikey Format (Shorts / Reels için ideal)
    payload = {
        "prompt": build_quality_prompt(prompt),
        "width": 768,
        "height": 1344,
        "num_steps": 8
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Görsel üretilemedi: {response.text}")

    return response.content

def run_pipeline():
    print("🚀 Yüksek kaliteli video üretimi başlatıldı...")
    # İş akışı mantığınız burada çalışmaya devam eder

if __name__ == "__main__":
    run_pipeline()
