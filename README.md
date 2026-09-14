# Comic Factory Studio

Türkçe çizgi roman hikâyeleri için tek GitHub Actions çalışma alanı. Sürüm: **2026-09-14-studio-2**.

Sistem konuyu seçer, gerçek çizgi roman sayfalarını araştırır, panel ve kaynak kanıtına bağlı özgün Türkçe anlatım yazar. Türkçe ses, altyazı kelimeleri ve sahne değişimleri aynı metne bağlıdır.

## Kullanım

GitHub → **Actions → Comic Factory Studio → Run workflow**.

| İşlem | Sonuç |
| --- | --- |
| `preview` | Otomatik konu seçimi, araştırma, video ve inceleme paketi. |
| `create_and_publish` | Kontrolleri geçen videoyu seçilen YouTube / Instagram hesabına yükler. |
| `publish_preview` | `preview_run_id` ile seçtiğin hazır videonun aynısını yükler. |
| `voice_test` | Referansa göre Orus, Gacrux ve Fenrir seslerini karşılaştırır. |

Konu boşsa Marvel/DC tarihinden önemli, şaşırtıcı olaylar araştırılır; kahraman popülerliği ve bilginin niş değeri sıralamada kullanılır. Belirli olay istiyorsan konu alanına yaz. Süre 20–165 saniye arasında hedeftir; varsayılan 150 saniye. Gerçek süre doğrulanmış malzeme ve konuşma hızına bağlıdır.

Ses `auto` olduğunda aynı Türkçe metin üç sesle okunur. Kayıtları dinleyen model doğallık, telaffuz ve anlatım enerjisini değerlendirir. Ses adından seçim yapılmaz. Örnekleri inceleme sayfasından kendin de dinleyebilirsin; menüden istediğin sesi seçebilirsin.

Çalışma sonunda **comic-preview-RUN_ID** artifact ZIP dosyasını indir, çıkart ve **review.html** dosyasını aç. Video, ses örnekleri, her sahnenin paneli ve kaynak bağlantısı aynı sayfadadır. Başarısız üretimde de mevcut dosyalar kaydedilir.

Yarım kalan bu sürümün üretimine devam etmek için yeni bir `preview` çalışması açıp **resume_run_id** alanına eski sayısal çalışma ID'sini yaz. Aynı konu/süre/ses ayarlarıyla tamamlanan aşamalar korunur. Değişen ayarlar ilgili aşamayı geçersiz kılar. Başarılı ses bölümleri ortak cache kaybolsa bile indirilen devam kaydından kullanılabilir. Önceki sürümlerin kayıtları bu sürüme devam kaydı olamaz.

## Referans ve görüntü

[Gönderdiğin referans](https://www.youtube.com/watch?v=yIZLrxqUUbg) Gemini'ye video olarak verilir. Analiz; farklı zamanlardan gözlemler, panel yerleşimi, altyazı rengi/konumu ve kelime düzeni, zoom, geçiş, anlatım temposu ve hikâye yapısını çıkarır. Gözlem alınamazsa stil uydurulmaz. Erişebildiğin referans dosyası `reference.mp4` adıyla projeye konursa analiz ve son karşılaştırma bu dosyayı kullanır.

Profil **reference_profile.json** dosyasındadır. Paneli bütünüyle gösterme, alana doldurma, sayfa gösterme, pan/zoom, kesme, dissolve ve slide desteklenir. Font ailesi DejaVu Sans / Condensed Bold ile yaklaşık eşlenir. Özel maskeler, referansın tam font dosyası, logosu veya karmaşık efektleri otomatik kopyalanmaz. Desteklenmeyen gözlenen özellikler profilde tutulur.

Bu paket hazırlanırken referansın görüntü ve ses akışı yerel ortamda oynatılamadı; birebir benzerlik doğrulanmış değildir. İlk gerçek videonun görünümü GitHub'daki analizle belirlenecek. Son video da modele görüntü ve ses olarak verilip referansla karşılaştırılır. Yalnız yerleşim sorunu varsa bir kez render düzeltmesi yapılır. Son kararı ilk gerçek önizlemeyi izleyerek ver.

## Kaynak, ses ve eşleşme

- Yayınevi önizlemeleri ve kamuya açık resimli yazılardaki gerçek görseller indirilir. Kapak, fan art, ilgisiz sayfa ve reklamlar görsel kontrolde elenir. Görseli AI ile yeniden çizen bir adım yoktur; kırpma, ölçekleme ve hafif netleştirme yapılır.
- Sayfa, seri/sayı/yıl kanıtına ve kaynak metinde gerçekten bulunan bir alıntıya bağlanır. Her anlatım cümlesi bilinen panel ve kanıt kimliği taşır. İkinci kontrol cümleyi gerçek kırpılmış panelle karşılaştırır.
- Gemini TTS kısa konuşma bölümleri üretir. Groq Whisper large-v3 gerçek kaydı çözer. Bozuk zamanlar gelirse aynı kayıtta ikinci ASR modeli denenir. Senaryo ASR'ye telkin eden bir prompt olarak verilmez.
- Ters/sırasız zamanlar kabul edilmez. Eksik/fazladan kelimeler ve tahmini zamanlar puanı düşürür. Eşik altında yalnız sorunlu ses bölümü yeniden üretilir; konu araştırması başa dönmez.
- Bölümler gerçek ses örneği sayısıyla birleştirilir; sahneler toplam 30 fps çizelgesine yerleştirilir. Türkçe büyük harfler korunur. Altyazı kelime grubu referansın gözlenen biçiminden gelir; konuşulan kelime vurgulanır.
- Çıktı 1080×1920, 30 fps H.264/AAC MP4'tür. Video tamamen decode edilir; ses akışı ve süre farkı kontrol edilir. Referansta müzik varsa basit özgün bir fon üretilip konuşmaya göre kısılır. Referans müziği kopyalanmaz. Kendi müziğin `music_file` ile seçilebilir.

Model puanları öznel değerlendirmedir; ölçülmüş doğruluk yüzdesi değildir. Kaynak tanıma ve ASR hata yapabilir. Yeterli gerçek panel bulunmayan otomatik konuda sıradaki aday denenir. Sorunlu çıktı hazır ilan edilmez; mevcut önizleme ve raporlar korunur.

## Mevcut secrets ve yayın

Workflow mevcut **18 secret eşlemesini** korur: Gemini, Groq, Instagram/Cloudinary ve YouTube JSON/base64 seçenekleri. Değerleri kod dosyalarına yazman gerekmez. `factory.json` yalnız üretim ayarları içindir.

Repo variables alanındaki `GEMINI_MODEL`, `GEMINI_TTS_MODEL` ve `GROQ_WHISPER_MODEL` varsayılan modelleri değiştirebilir. Actions menüsünde seçtiğin ses `GEMINI_TTS_VOICE` değişkeninden önceliklidir. Servis kotaları ve olası ücretler mevcut hesaplarına bağlıdır; ücretsiz çalışma garantisi yoktur.

Varsayılan sınırlar: 120 Gemini isteği, 3 aday olay, ses bölümü başına 3 üretim, aynı kayıt için 2 ASR modeli ve 100 dakikalık işlem bütçesi. Üretim/kontrol sınırlarına ulaşıldığında dosyalar korunur.

Yayınlanan dosyanın özeti kaydedilir. Başarılı yükleme tekrar gönderilmez. Önceki gönderimin sonucu belirsizse otomatik tekrar gönderilmez; `publication.json` ve platform hesabından kontrol edilebilir. YouTube varsayılan görünürlüğü public'tir. `preview` yayın yapmaz.

## Geliştirme

Tek giriş `comic_factory.py`, tek üretim workflow'u `.github/workflows/comic-factory.yml`.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python comic_factory.py --check
python comic_factory.py --topic "" --duration 150 --voice auto
```

Yerelde FFmpeg, FFprobe ve DejaVu fontları gerekir; Actions bunları kurar. Gerçek FFmpeg entegrasyon testi: `CF_RUN_MEDIA_TESTS=1 python -m unittest tests.test_media -v`. Test çizilmiş basit paneller ve ton sinyali kullanır; gerçek Türkçe ses kalitesini ölçmez.

`data/runs/` çıktıları, `data/cache-v2/` analiz/ses cache'ini, `data/events/` kullanılan olayları, `data/publishing/` yayın kayıtlarını tutar. Bunlar Git'e eklenmez. Önizleme artifact'ları 30 gün; cache kayıtları GitHub'ın cache politikası boyunca saklanır.

## API belgeleri

- [Gemini video ve YouTube analizi](https://ai.google.dev/gemini-api/docs/video-understanding)
- [Gemini Türkçe ses üretimi](https://ai.google.dev/gemini-api/docs/speech-generation)
- [Google Search ile araştırma](https://ai.google.dev/gemini-api/docs/google-search)
- [Groq konuşma çözümleme](https://console.groq.com/docs/speech-to-text)
- [GitHub Git Trees API](https://docs.github.com/en/rest/git/trees?apiVersion=2022-11-28)
