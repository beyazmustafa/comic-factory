# Comic Factory Studio

Türkçe çizgi roman hikâyeleri için tek GitHub Actions çalışma alanı. Sürüm: **2026-10-01-archive-1**.

Sistem Internet Archive'dan **kamu malı (public domain) bir Altın Çağ çizgi roman sayısının tamamını** indirir, hikâye sayfalarını panellere böler, panellere bağlı özgün Türkçe anlatım yazar, Türkçe seslendirir ve kelime kelime altyazılı dikey video üretir. Günde iki kez kendi kendine çalışır ve YouTube kanalına yükler.

## Günde iki yükleme nasıl çalışır

`.github/workflows/comic-factory.yml` içindeki `schedule` satırı workflow'u her gün **09:00 ve 18:00** (Türkiye saati; cron `0 6,15 * * *` UTC) tetikler. Zamanlanmış koşuda elle girilen alanlar boş olduğu için şu varsayılanlar devreye girer:

| Ayar | Zamanlanmış değer | Nereden değişir |
| --- | --- | --- |
| İşlem | `create_and_publish` | — |
| Konu | otomatik (arşivden rastgele dilim) | — |
| Süre | 150 sn | `factory.json → target_seconds` |
| Platform | yalnız `youtube` | Repo variable `SCHEDULED_PLATFORMS` (`both`, `youtube`, `instagram`) |
| Görünürlük | `public` (Shorts rafına ancak public video düşer) | Repo variable `YOUTUBE_VISIBILITY` (`public`, `unlisted`, `private`) |

Videolar doğrudan public yayınlanır; başlık ve açıklamaya `#Shorts` eklenir. Önce görmek istersen `YOUTUBE_VISIBILITY=unlisted` değişkeni eklersin. Her videonun senaryosu ve kare şeridi `data/history/videos/<run_id>/` altına kaydedilir.

Kullanılan sayılar `data/history/issues/` altında **repoya commit edilir** (workflow `contents: write` izniyle kendi commit'ini atar). Böylece cache silinse bile aynı sayı bir daha seçilmez. Bu klasörü silmek geçmişi sıfırlar.

GitHub zamanlanmış koşuları yoğun saatlerde geciktirebilir; dakika gelince değil, birkaç dakika–yarım saat sonra başlaması normaldir. Zamanlanmış koşular yalnız varsayılan branch'te çalışır.

> **Dakika sınırı:** Private repoda ücretsiz Actions süresi ayda 2000 dakikadır. Bir üretim 30–60 dakika sürer; günde iki koşu ayda 1800–3600 dakika eder. Repo **public** yapılırsa Actions sınırsızdır (kodda hiçbir gizli bilgi yok; secrets zaten repo ayarlarında). Private kalacaksa `factory.json → max_minutes` değerini 45'e çekmek ve `cron` satırını günde bire indirmek gerekir.

## Kaynak: neden kamu malı arşiv

Referans alınan Shorts formatı (gerçek sayı, panel panel, kesintisiz anlatım) ancak bir sayının **tüm sayfalarıyla** çıkar. Marvel/DC gibi güncel sayıların tamamını indirip yüklemek telif ihlalidir ve Content ID ile kanalı kapattırır; bu yüzden sistem yalnız 1964 öncesi, telifi yenilenmemiş sayıları kullanır.

Seçim iki kademelidir (`factory/archive.py`):

1. Arşiv kaydı açık bir kamu malı beyanı taşıyorsa (`licenseurl`, `rights`, açıklamada "public domain") → kabul (`declared`).
2. Beyan yoksa: tarih ≤ `archive_max_year` (1963) **ve** yayıncı/koleksiyon, 1964'ten önce kapanmış ve kataloğu yenilenmemiş yayınevlerinden biriyse (Ace, Fox, Fiction House, Lev Gleason, Charlton, Nedor/Standard, Avon, Ajax-Farrell, Centaur, Prize, Hillman, Ziff-Davis, Youthful, Star, Toby, St. John, Quality, Fawcett…) → kabul (`inferred`).

Her iki kademede de hâlâ korunan markaların adı geçiyorsa (Disney, MAD, Marvel/Timely/Atlas, DC/National, Archie, Harvey, Dell, EC, Classics Illustrated, King Features karakterleri, Captain Marvel/Shazam, Plastic Man, The Spirit…) sayı reddedilir. Kanıt `research/archive_search.json` ve `events/*/sources/source_000.json` içine yazılır ve video açıklamasında kaynak bağlantısı verilir. Not: Internet Archive'daki üst veri gönüllü girilir; filtre dikkatli ama kusursuz değildir. Şüpheli bir sayı görürsen `data/history/issues/` kaydı dururken videoyu kaldırman yeterlidir; `PROTECTED_MARKERS` listesine kelime eklemek o yayıncıyı kalıcı olarak engeller.

Sorgu `factory.json → archive_query` ile değişir (varsayılan `mediatype:texts AND collection:(comics)`). Popüler kayıtlar listenin başında toplandığı için her koşu ilk sayfaya ek olarak iki **rastgele** sonuç sayfası okur; böylece günlük koşular farklı dilimler görür. Elle çalıştırırken "Konu" alanı arşiv aramasını daraltır (ör. `jungle`, `crime`, `horror`, `science fiction`, `Fox`).

## Dil, konu ve kendi kendine öğrenme

- `factory.json → language` (`en` varsayılan): anlatım, altyazı, ASR dili ve YouTube dil etiketi buradan gelir. Ses: Gemini TTS; kota dolarsa edge-tts `en-US-ChristopherNeural` (derin, enerjik anlatıcı), Türkçe için `tr-TR-AhmetNeural`.
- `channel_theme: superheroes`: arşiv araması süper kahraman sayılarına (`archive_theme_query`) daraltılır; model, bir Shorts izleyicisi için en çarpıcı olayı vadeden sayıyı seçer.
- Öğrenme döngüsü (`factory/learning.py`): her yüklenen video `data/history/performance.json` içine kurgu profili (kanca uzunluğu, sahne sayısı, vurgu dağılımı, süre) ve kalite raporuyla yazılır; sonraki koşular YouTube'dan izlenme/beğeni/yorum sayılarını çeker (saat başına izlenme ile normalize). Her koşunun başında model bu tabloyu, son kalite raporunu ve `data/history/playbook.md` dosyasını okuyup oyun kitabını yeniden yazar ve o video için **tek bir ölçülebilir deney** seçer (`data/history/experiments.json`). Senaryo yazarı oyun kitabı + deneyi alır; bir sonraki koşu deneyin rakamlarla sonucunu görür ve "tutuldu/düştü" kararı verir. Dosyalar repoya commit edildiği için ilerleme kalıcıdır ve elle düzenlenebilir (oyun kitabına kendi kuralını yazabilirsin).

## Yoğunluk hatalarına karşı sağlayıcı zinciri

Tek bir modele bağlı kalınmaz; her istek sırayla şunları dener (`factory/api.py`):

1. `gemini_model` (varsayılan `gemini-3.8-flash`; repo variable `GEMINI_MODEL` ile değişir)
2. `gemini_fallback_models` listesi: `gemini-3.7-flash, gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.0-flash` (`GEMINI_FALLBACK_MODELS` ile değişir)
3. Hepsi 429/503 verirse metin+görsel işler **Groq**'un görsel destekli modeline geçer (`groq_model`, varsayılan `meta-llama/llama-4-scout-17b-16e-instruct`; mevcut `GROQ_API_KEY` kullanılır).

503 veren model o koşuda bir daha denenmez; böylece her çağrıda dakikalar kaybedilmez. Ses için Gemini TTS düşerse ücretsiz ve anahtarsız **edge-tts** Türkçe sesleri (`Ahmet`, `Emel`) devreye girer; video kontrolünü yapacak Gemini yoksa karelerden görsel inceleme + Whisper'ın ölçtüğü senkron puanı kullanılır. Hangi yedeğin ne zaman devreye girdiği `diagnostics/provider_events.json` dosyasına yazılır.

## Sayfadan panele

- İndirme sırası: `_images.zip` (ham tarama) → `.cbz` → `_jp2.zip` → `.pdf` (poppler) → `.cbr` (7z). Biri açılmazsa sıradaki denenir; hepsi başarısızsa `download_errors.json` kaydedilir ve sıradaki aday sayıya geçilir.
- Paneller **kodla** kesilir (`factory/panels.py → segment_page`): sayfa gri tona çevrilir, kâğıt rengi kestirilir, neredeyse tamamen kâğıt olan satır/sütun şeritleri "oluk" sayılır ve sayfa önce yatay, sonra dikey şeritlere bölünerek paneller bulunur. Model koordinat tahmin etmez; sayfanın üstüne kırmızı numaralarla işaretlenmiş panelleri **tarif eder** ve görünen konuşmaları aynen yazar. Eski sistemdeki "panel tahmini tutmuyor" sorununun çözümü budur.
- Kapak (ilk sayfa), reklam ve düz metin sayfaları panel sayısı/kaplama oranıyla elenir; ≥4 ardışık panelli sayfadan oluşan ilk hikâye bloğu alınır (`max_pages` kadar).
- Anlatım kanıtı: sayfadaki görünür balon/altyazı metninden ≥18 karakterlik birebir alıntı. Arşivdeki OCR metni (`_djvu.txt`) varsa kaynağa eklenir.
- Senaryo bu kaynakta "tarih anlatımı" değil, **hikâyenin kendisini** panel sırasıyla anlatır (kim, ne oluyor, dönüş, gerçek son).

`factory.json → source` değeri `web` yapılırsa eski yayınevi-önizleme yolu kullanılır.

## Elle kullanım

GitHub → **Actions → Comic Factory Studio → Run workflow**.

| İşlem | Sonuç |
| --- | --- |
| `preview` | Otomatik sayı seçimi, indirme, video ve inceleme paketi; yüklemez. |
| `create_and_publish` | Kontrolleri geçen videoyu seçilen YouTube / Instagram hesabına yükler. |
| `publish_preview` | `preview_run_id` ile seçtiğin hazır videonun aynısını yükler. |
| `voice_test` | Sabit anlatım yönergesiyle Orus, Gacrux ve Fenrir seslerini karşılaştırır. |

Çalışma sonunda **comic-preview-RUN_ID** artifact ZIP dosyasını indir, çıkart ve **review.html** dosyasını aç. Video, ses örnekleri, her sahnenin paneli ve kaynak bağlantısı aynı sayfadadır. Yarım kalan üretimi `resume_run_id` ile sürdürebilirsin.

## Sabit kurgu ve görüntü

Geliştirme sırasında sağlanan 164,34 saniyelik 720×1280, 30 FPS MP4 açılıp başlangıç, orta ve son bölümlerinden kareler incelendi. Kurgu kuralları `factory/style.py` içine aktarıldı. Üretimde referans dosyası veya URL'si okunmaz, indirilmez, analiz edilmez ve referansla karşılaştırma yapılmaz. Pakete referans video eklenmez.

Tam ekran panel kadrajı, %12'ye kadar yaklaşma/uzaklaşma, yatay ve dikey pan; 5 karelik bulanık dikey geçiş kullanılır. Altyazı ekran yüksekliğinin %62'sinde, kalın siyah konturlu sarı büyük harflerdir. Tehlike, sürpriz ve yön değişimi sahneleri kırmızı, yeşil veya turkuaz vurgu alır. Kelimeler gerçek ses zamanlarına göre tek tek görünür. Font mevcut Türkçe destekli kalın fontla yaklaşık eşlenir; kaynak videonun birebir fontu veya tüm efektleri kopyalanmaz.

Her çalışmanın ayarları `editing_profile.json` dosyasındadır. Son kalite kontrolüne yalnız üretilen video gönderilir; Türkçe anlatım, senkron, panel/anlatım uyumu ve okunabilirlik değerlendirilir. Yerleşim hatalarında en fazla bir render düzeltmesi yapılır.

## Kaynak, ses ve eşleşme

- Varsayılan kaynak (`archive`) Internet Archive'daki kamu malı sayının gerçek taramalarıdır; `web` kaynağında yayınevi önizlemeleri indirilir. Görseli AI ile yeniden çizen bir adım yoktur; kırpma, ölçekleme ve hafif netleştirme yapılır.
- Her anlatım cümlesi bilinen panel ve kanıt kimliği taşır; kanıt arşiv üst verisi ve sayfada görünen metinden birebir alıntıdır. İkinci kontrol cümleyi gerçek kırpılmış panelle karşılaştırır.
- Gemini TTS kısa konuşma bölümleri üretir. Groq Whisper large-v3 gerçek kaydı çözer. Bozuk zamanlar gelirse aynı kayıtta ikinci ASR modeli denenir. Senaryo ASR'ye telkin eden bir prompt olarak verilmez.
- Ters/sırasız zamanlar kabul edilmez. Eksik/fazladan kelimeler ve tahmini zamanlar puanı düşürür. Eşik altında yalnız sorunlu ses bölümü yeniden üretilir; konu araştırması başa dönmez.
- Bölümler gerçek ses örneği sayısıyla birleştirilir; sahneler toplam 30 fps çizelgesine yerleştirilir. Türkçe büyük harfler korunur. Altyazı biçimi sabit kurgu profilinden gelir; konuşulan kelime vurgulanır.
- Çıktı 1080×1920, 30 fps H.264/AAC MP4'tür. Video tamamen decode edilir; ses akışı ve süre farkı kontrol edilir. Varsayılan müziksizdir. Kurgu profilinde müzik açılırsa basit özgün fon konuşmaya göre kısılır. Kendi müziğin `music_file` ile seçilebilir.

Model puanları öznel değerlendirmedir; ölçülmüş doğruluk yüzdesi değildir. Kaynak tanıma ve ASR hata yapabilir. Yeterli gerçek panel bulunmayan otomatik konuda sıradaki aday denenir. Sorunlu çıktı hazır ilan edilmez; mevcut önizleme ve raporlar korunur.

## YouTube yetkisi düştüğünde

Koşu `YouTube OAuth token yenilenemedi` derse Google yenileme anahtarını iptal etmiştir (test modundaki OAuth uygulamalarında 7 günde bir olur). Bilgisayarında bir kez:

```bash
pip install google-auth google-auth-oauthlib
python tools/youtube_authorize.py client_secret.json
```

Tarayıcı açılır, izin verirsin; betik `YOUTUBE_TOKEN_B64` değerini basar, GitHub'daki secret'ı bununla güncellersin. Kalıcı çözüm: Google Cloud Console → OAuth consent screen → **Publishing status: In production** (o zaman anahtar düşmez). Üretilmiş ama yüklenememiş videoyu `publish_preview` ile sonradan yükleyebilirsin; aynı sayı tekrar üretilmez.

## Mevcut secrets ve yayın

Workflow mevcut **18 secret eşlemesini** korur: Gemini, Groq, Instagram/Cloudinary ve YouTube JSON/base64 seçenekleri. Değerleri kod dosyalarına yazman gerekmez. `factory.json` yalnız üretim ayarları içindir.

Repo variables alanındaki `GEMINI_MODEL`, `GEMINI_TTS_MODEL` ve `GROQ_WHISPER_MODEL` varsayılan modelleri değiştirebilir. Actions menüsünde seçtiğin ses `GEMINI_TTS_VOICE` değişkeninden önceliklidir. Servis kotaları ve olası ücretler mevcut hesaplarına bağlıdır; ücretsiz çalışma garantisi yoktur.

Varsayılan sınırlar: 120 Gemini isteği, 3 aday sayı, ses bölümü başına 3 üretim, aynı kayıt için 2 ASR modeli ve 100 dakikalık işlem bütçesi. Üretim/kontrol sınırlarına ulaşıldığında dosyalar korunur.

Yayınlanan dosyanın özeti kaydedilir. Başarılı yükleme tekrar gönderilmez. Önceki gönderimin sonucu belirsizse otomatik tekrar gönderilmez; `publication.json` ve platform hesabından kontrol edilebilir. YouTube görünürlüğü `YOUTUBE_VISIBILITY` değişkeninden gelir; boşsa public. `preview` yayın yapmaz.

## Geliştirme

Tek giriş `comic_factory.py`, tek üretim workflow'u `.github/workflows/comic-factory.yml`.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python comic_factory.py --check
python comic_factory.py --topic "" --duration 150 --voice auto
```

Yerelde FFmpeg, FFprobe ve DejaVu fontları gerekir; Actions bunları kurar. Gerçek FFmpeg entegrasyon testi: `CF_RUN_MEDIA_TESTS=1 python -m unittest tests.test_media -v`. Test çizilmiş basit paneller ve ton sinyali kullanır; gerçek Türkçe ses kalitesini ölçmez.

`data/runs/` çıktıları, `data/cache-v2/` ses cache'ini, `data/events/` kullanılan olayları, `data/publishing/` yayın kayıtlarını tutar. Bunlar Git'e eklenmez. Önizleme artifact'ları 30 gün; cache kayıtları GitHub'ın cache politikası boyunca saklanır.

## API belgeleri

- [Gemini video ve YouTube analizi](https://ai.google.dev/gemini-api/docs/video-understanding)
- [Gemini Türkçe ses üretimi](https://ai.google.dev/gemini-api/docs/speech-generation)
- [Google Search ile araştırma](https://ai.google.dev/gemini-api/docs/google-search)
- [Groq konuşma çözümleme](https://console.groq.com/docs/speech-to-text)
- [GitHub Git Trees API](https://docs.github.com/en/rest/git/trees?apiVersion=2022-11-28)

## Studio 4: bağımsız kaynak araştırması

Gemini Google Search araç çağrısı kaldırıldı. DDGS ile kaynaklar bulunur, gerçek sayfa metinleri indirilir ve konu seçimine verilir. Konu adaylarında yalnız okunmuş kaynak URL adresleri kabul edilir. discovery.json arama ve erişim sonuçlarını saklar. Model ve ses kotaları geçerlidir. 2.5 Flash erişim hatası alındıysa GEMINI_MODEL değişkenini erişilebilir 3.7 Flash modeline ayarlayın.

## Studio 5: konu yanıtı uyumluluğu

Konu seçimi çağrısında kök JSON listesi events nesnesine dönüştürülür. Nesne yanıtları değişmez. Liste üyeleri nesne olmalıdır; diğer API çağrılarında beklenmeyen kök listeler reddedilir. Kaynak ve konu doğrulamaları aynı şekilde uygulanır.
