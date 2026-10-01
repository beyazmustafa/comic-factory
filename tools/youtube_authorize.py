"""YouTube yetkilendirme yardımcısı (bilgisayarında bir kez çalıştırılır).

Kullanım:
    python tools/youtube_authorize.py client_secret.json

Ne yapar:
  1. Tarayıcıyı açar, Google hesabını seçip YouTube iznini verirsin.
  2. token.json dosyasını yazar.
  3. GitHub'a yapıştıracağın iki secret değerini ekrana basar:
       YOUTUBE_CLIENT_SECRET_B64  ve  YOUTUBE_TOKEN_B64
     (GitHub → repo → Settings → Secrets and variables → Actions → ilgili secret → Update)

Yenileme anahtarının 7 günde bir düşmemesi için Google Cloud Console →
APIs & Services → OAuth consent screen → "Publishing status" = In production
olmalı; test modunda kalan uygulamaların anahtarı 7 günde iptal edilir.

Gerekli paketler: pip install google-auth google-auth-oauthlib
Bu betik hiçbir dosyayı silmez; yalnızca token.json yazar.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    client_file = Path(argv[1])
    if not client_file.is_file():
        print(f"Dosya bulunamadı: {client_file}")
        return 2
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("Önce: pip install google-auth google-auth-oauthlib")
        return 2
    flow = InstalledAppFlow.from_client_secrets_file(str(client_file), SCOPES)
    credentials = flow.run_local_server(
        port=0,
        open_browser=True,
        authorization_prompt_message="Google hesabını seçip YouTube iznini ver.",
        success_message="YouTube bağlantısı başarılı. Bu sekmeyi kapatabilirsin.",
    )
    payload = json.loads(credentials.to_json())
    if not payload.get("refresh_token"):
        print("Google yenileme anahtarı vermedi. Google hesabı → Güvenlik → Üçüncü taraf erişimi "
              "bölümünden uygulamanın eski iznini kaldırıp betiği tekrar çalıştır.")
        return 1
    token_file = client_file.with_name("token.json")
    token_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    token_b64 = base64.b64encode(json.dumps(payload).encode("utf-8")).decode()
    client_b64 = base64.b64encode(client_file.read_bytes()).decode()
    print("\n" + "=" * 70)
    print("token.json yazıldı:", token_file)
    print("\nGitHub secret YOUTUBE_TOKEN_B64 değeri (tek satır, tamamını kopyala):\n")
    print(token_b64)
    print("\nGitHub secret YOUTUBE_CLIENT_SECRET_B64 değeri (zaten aynıysa dokunma):\n")
    print(client_b64)
    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
