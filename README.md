# Comic Factory Studio

Tek komut, tek GitHub Actions akışı: kaynaklı konu seçimi → Türkçe senaryo → ses → kelime zamanları → gerçek çizgi roman panelleri → montaj → önizleme → YouTube / Instagram.

## GitHub üzerinden kullanım

**Actions → Comic Factory Studio → Run workflow** ekranı tek çalışma alanıdır.

| İşlem | Davranış |
| --- | --- |
| `preview` | Videoyu üretir; önizleme ZIP dosyasını hazırlar. |
| `create_and_publish` | Yeni videoyu üretir, kontroller geçerse seçilen platformlara yükler. |
| `publish_preview` | Önceki çalışmanın videosunu yeniden üretmeden yükler. `preview_run_id` gerekir. |

`topic` alanına **Thor'un Galactus'u öldürdüğü olay — Thor (2020) #6** gibi belirli bir olay yaz. Boş bırakırsan sistem popüler kahramanlar ve az bilinen olaylar arasından araştırır. `duration` hedef süredir; sesin gerçek süresi konuşma hızına bağlıdır. `brief` alanına anlatım isteğini yazabilirsin.

Önce `preview` kullan. Çalışmanın **Artifacts** bölümündeki `comic-preview-<sayısal Actions ID>` dosyasını indir, ZIP'i çıkart, `review.html` dosyasını aç. Video, senaryo ve tüm sahneler aynı sayfadadır. Yükleme için aynı ekranı tekrar açıp `publish_preview` ve önceki çalışmanın sayısal ID değerini seç. ID çalışma özetinde yazılır. Önizlemeler 30 gün tutulur; sonrasında yerel dosyaları kullanabilir veya yeni önizleme üretebilirsin.

Bir platforma yükleme başarılı, diğerine başarısız olursa başarılı gönderim geçmişe kaydedilir. Sonucu belirsiz bir ağ kesintisinde program körlemesine yeniden paylaşmaz; `publication.json` ve `data/publishing` kaydı hangi platformun kontrol edilmesi gerektiğini gösterir.

İlk kalite değerlendirmesi için otomatik günlük yayın yerine elle çalıştırma kullanılır. Günlük çalışma istenirse aynı workflow'a zamanlama eklenebilir; yeni bir üretim dosyası gerekmez.

## Yerel kullanım

Python **3.12**, **FFmpeg + FFprobe** ve Türkçe karakter destekli **DejaVu Sans / Arial Bold** gerekir.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python comic_factory.py --check
.\.venv\Scripts\python comic_factory.py --topic "Thor (2020) #6: Thor ve Galactus" --duration 55
```

Kendi panel klasörünü kullanmak için:

```powershell
.\.venv\Scripts\python comic_factory.py --topic "Thor (2020) #6" --panel-dir "C:\ComicPaneller\Thor"
```

Önizlemeyi yüklemek için:

```powershell
.\.venv\Scripts\python comic_factory.py --publish-run "data\runs\CALISMA_KIMLIGI" --platforms both
```

## Dosyalar

| Dosya / klasör | İşlev |
| --- | --- |
| `comic_factory.py` | Tek çalıştırma komutu |
| `.github/workflows/comic-factory.yml` | Tek Actions ekranı |
| `factory.json` | Süre, ses, sahne, kalite ve deneme sınırları |
| `factory/engine.py` | Konu, senaryo, ses, görsel ve montaj |
| `factory/core.py` | Kelime hizalama, altyazı ve medya doğrulama |
| `factory/studio.py` | İş akışı ve önizleme paketleri |
| `factory/publishing.py` | Aynı çıktıyı platformlara gönderme ve gönderim geçmişi |
| `factory/publishers/` | Mevcut YouTube ve Instagram API kodlarının uyarlanmış halleri |
| `tests/` | Ağ çağrısı yapmayan işlev testleri |
| `privacy-policy.html` | Mevcut gizlilik sayfası |

Eski V3, V4 ve Thor prototipleri ayrı çalıştırılan dosyalar olarak kullanılmaz. V4'ün işe yarayan bölümleri tek motor içine taşınmıştır.

## Kalite davranışı

- Türkçe anlatım; özel isimlerin metni korunur. Gacrux varsayılan sestir. Anlatım talimatı `brief` ve senaryoya göre üretilir.
- Ekranda konuşulan kelime ve sonraki kelime bulunur. Konuşulan kelime renk değiştirir. Kaydırmalı pencere önceki kelimeyi ekranda tutmaz.
- Kelimeler gerçek ses zamanlarına küresel eşleştirmeyle bağlanır. Eksik tanınan kelimeler sesin dışına taşırılmaz ve hizalama puanını düşürür.
- Sahne sınırları kare ızgarasına oturur. Bağımsız yuvarlamalardan kaynaklanan birikimli ses/görüntü kayması önlenir.
- 1080 × 1920, H.264, AAC, 30 FPS; kontrollü yakınlaşma ve kaydırma; ses seviyesi normalizasyonu.
- Gerçek panellerde kırpma, boyutlandırma ve hafif netleştirme kullanılır. Ana görselin rengi değiştirilmez. AI çizimi varsayılan olarak kapalıdır.
- Arama ve üretim denemelerinin ayrı sınırları vardır. Konu açıkça istendiğinde başarısızlık başka kahramanın videosuna çevrilmez.
- Yapay zekâ kalite puanları editoryal tahmindir, olgusal doğruluk veya güzel video garantisi değildir. Puanları videoları izleyerek kalibre etmeliyiz. Yapısal kontroller, kaynak URL eşleştirmesi ve gerçek FFmpeg çözme kontrolü ayrıca uygulanır.
- `factory.json` içindeki görsel eşiği, kaynak panelin gerçek çözünürlüğüne göre ayarlanmıştır; tüm sayfayı panelden daha büyük olduğu için otomatik üstün sayan eski ölçüm düzeltilmiştir. Eşikler çalışma sırasında kendiliğinden düşürülmez.

Arama kaynakları ve bulunan paneller her konu için yeterli olmayabilir. Bu sürüm film klibini otomatik bulup kesmez; böyle bir işlev eklendi diye varsayılmamalıdır. Önce çizgi roman videosu akışının gerçek örneklerini değerlendireceğiz.

## Mevcut secrets

GEMINI / GROQ / INSTAGRAM / CLOUDINARY secret adları korunur. YouTube için mevcut `token.json`, `client_secret.json`, uygun eski `youtube_credentials.json` veya workflow'daki JSON/base64 secret eşlemeleri okunur. Paket yeni API anahtarı içermez ve senden secrets değerlerini paylaşmanı istemez.

Kimlik dosyaları artifact veya cache kapsamına alınmaz. Olay ve yükleme geçmişi Actions cache ile saklanır; GitHub cache'in silinmesi geçmişin kaybına yol açabilir. Önemli yayın geçmişlerini ayrıca yedeklemek gerekir.

## Ses eşleştirmesi başarısız olduğunda

Ses yaması `2026-09-12-audio-1` ile geçersiz zaman bilgisi yalnız o çözümleme denemesini başarısız yapar. Önce ayarlı Whisper modeli, gerekirse **aynı ses kaydında** `whisper-large-v3` denenir. İkisi de geçemezse yeni ses üretilir; varsayılan sınır üç ses kaydı ve en çok altı çözümleme çağrısıdır. SDK'nın ağ hatası tekrarları bu çağrıların içinde ayrıca çalışabilir. Ses denemeleri tükenirse konu değiştirilmeden hata raporu hazırlanır.

Sıfır süreli kelimeler kesin zaman verisi sayılmaz. Uygun aralık varsa tahmini zaman alırlar ve `timing_coverage` puanını düşürürler. Bölünmüş/birleşmiş Türkçe kelimeler iki yönde eşleşir; ASR'nin tek kelime olarak verdiği zamanın ikiye bölünmesi de tahmin olarak işaretlenir. Sırasız, ters veya kayıt dışındaki zamanlar yeni çözümleme gerektirir. 95 puan eşiği korunur; eksik kelimeler veya çözülemeyen zamanlar puan kaybettirir. Senaryo, doğruluğunu sınamak için ses tanıma isteğine hazır cevap olarak verilmez.

**Artifacts → `comic-diagnostics-...`** paketindeki `audio_diagnostics` klasörü denenen WAV kayıtlarını, beklenen anlatımı, ham tanıma yanıtlarını, hizalama puanlarını ve hata nedenlerini içerir. `attempts.json` her ses/model sonucunu gösterir. Bir hata sürerse bu ZIP'i paylaş; sesin gerçekten eksik okunmasını ve konuşma tanıma hatasını kayıt üzerinden ayırabiliriz. Eski sürümde yalnız bilinen ses hizalama hataları yüzünden elenen konular yeniden seçilebilir; geçmiş kayıtlar saklanır.

Düzeltmeyi repoya ekledikten sonra güncel daldan **Run workflow → preview** başlat. GitHub, eski çalışmadaki **Re-run jobs** için aynı commit'i kullanır; bu düğme yeni kodu almaz. [GitHub yeniden çalıştırma belgesi](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs).

## Kontroller

```powershell
python -m unittest discover -s tests -v
python comic_factory.py --check
```

`--check` kurulum kontrolüdür; API servislerinin çevrimiçi veya hesabın yetkili olduğunu iddia etmez. İlk gerçek üretim ve gerçek platform yüklemesi senin GitHub ortamında doğrulanacaktır.

## Teknik başvurular

Model adları ve ses arabirimi için [Google model listesi](https://ai.google.dev/gemini-api/docs/models) ve [Google ses üretimi belgesi](https://ai.google.dev/gemini-api/docs/speech-generation) kontrol edildi. Secrets aktarımı için [GitHub Actions secrets belgesi](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets) temel alındı.

Konuşma tanıma modelleri, Türkçe dil seçimi ve kelime/segment zamanları için [Groq ses tanıma belgesi](https://console.groq.com/docs/speech-to-text) temel alındı.
