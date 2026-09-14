# Comic Factory Studio

Türkçe çizgi roman hikâyeleri için tek GitHub Actions çalışma alanı. Sürüm: **2026-09-14-studio-4**.

Sistem konuyu seçer, gerçek çizgi roman sayfalarını araştırır, panel ve kaynak kanıtına bağlı özgün Türkçe anlatım yazar. Türkçe ses, altyazı kelimeleri ve sahne değişimleri aynı metne bağlıdır.

## Kullanım

GitHub → **Actions → Comic Factory Studio → Run workflow**.

| İşlem | Sonuç |
| --- | --- |
| `preview` | Otomatik konu seçimi, araştırma, video ve inceleme paketi. |
| `create_and_publish` | Kontrolleri geçen videoyu seçilen YouTube / Instagram hesabına yükler. |
| `publish_preview` | `preview_run_id` ile seçtiğin hazır videonun aynısını yükler. |
| `voice_test` | Sabit anlatım yönergesiyle Orus, Gacrux ve Fenrir seslerini karşılaştırır. |

Konu boşsa Çizgi roman tarihinden (Marvel/DC, bağımsız yayınlar, manga ve Avrupa çizgi romanları) önemli, şaşırtıcı olaylar araştırılır; kahraman popülerliği ve bilginin niş değeri sıralamada kullanılır. Belirli olay istiyorsan konu alanına yaz. Süre 20–165 saniye arasında hedeftir; varsayılan 150 saniye. Gerçek süre doğrulanmış malzeme ve konuşma hızına bağlıdır.

Ses `auto` olduğunda aynı Türkçe metin üç sesle okunur. Kayıtları dinleyen model doğallık, telaffuz ve anlatım enerjisini değerlendirir. Ses adından seçim yapılmaz. Örnekleri inceleme sayfasından kendin de dinleyebilirsin; menüden istediğin sesi seçebilirsin.

Çalışma sonunda **comic-preview-RUN_ID** artifact ZIP dosyasını indir, çıkart ve **review.html** dosyasını aç. Video, ses örnekleri, her sahnenin paneli ve kaynak bağlantısı aynı sayfadadır. Başarısız üretimde de mevcut dosyalar kaydedilir.

Yarım kalan bu sürümün üretimine devam etmek için yeni bir `preview` çalışması açıp **resume_run_id** alanına eski sayısal çalışma ID'sini yaz. Aynı konu/süre/ses ayarlarıyla tamamlanan aşamalar korunur. Değişen ayarlar ilgili aşamayı geçersiz kılar. Başarılı ses bölümleri ortak cache kaybolsa bile indirilen devam kaydından kullanılabilir. Önceki sürümlerin kayıtları bu sürüme devam kaydı olamaz.

## Sabit kurgu ve görüntü

Geliştirme sırasında sağlanan 164,34 saniyelik 720×1280, 30 FPS MP4 açılıp başlangıç, orta ve son bölümlerinden kareler incelendi. Kurgu kuralları `factory/style.py` içine aktarıldı. Üretimde referans dosyası veya URL'si okunmaz, indirilmez, analiz edilmez ve referansla karşılaştırma yapılmaz. Pakete referans video eklenmez.

Tam ekran panel kadrajı, %12'ye kadar yaklaşma/uzaklaşma, yatay ve dikey pan; 5 karelik bulanık dikey geçiş kullanılır. Altyazı ekran yüksekliğinin %62'sinde, kalın siyah konturlu sarı büyük harflerdir. Tehlike, sürpriz ve yön değişimi sahneleri kırmızı, yeşil veya turkuaz vurgu alır. Kelimeler gerçek ses zamanlarına göre tek tek görünür. Font mevcut Türkçe destekli kalın fontla yaklaşık eşlenir; kaynak videonun birebir fontu veya tüm efektleri kopyalanmaz.

Her çalışmanın ayarları `editing_profile.json` dosyasındadır. Son kalite kontrolüne yalnız üretilen video gönderilir; Türkçe anlatım, senkron, panel/anlatım uyumu ve okunabilirlik değerlendirilir. Yerleşim hatalarında en fazla bir render düzeltmesi yapılır.

## Kaynak, ses ve eşleşme

- Yayınevi önizlemeleri ve kamuya açık resimli yazılardaki gerçek görseller indirilir. Kapak, fan art, ilgisiz sayfa ve reklamlar görsel kontrolde elenir. Görseli AI ile yeniden çizen bir adım yoktur; kırpma, ölçekleme ve hafif netleştirme yapılır.
- Sayfa, seri/sayı/yıl kanıtına ve kaynak metinde gerçekten bulunan bir alıntıya bağlanır. Her anlatım cümlesi bilinen panel ve kanıt kimliği taşır. İkinci kontrol cümleyi gerçek kırpılmış panelle karşılaştırır.
- Gemini TTS kısa konuşma bölümleri üretir. Groq Whisper large-v3 gerçek kaydı çözer. Bozuk zamanlar gelirse aynı kayıtta ikinci ASR modeli denenir. Senaryo ASR'ye telkin eden bir prompt olarak verilmez.
- Ters/sırasız zamanlar kabul edilmez. Eksik/fazladan kelimeler ve tahmini zamanlar puanı düşürür. Eşik altında yalnız sorunlu ses bölümü yeniden üretilir; konu araştırması başa dönmez.
- Bölümler gerçek ses örneği sayısıyla birleştirilir; sahneler toplam 30 fps çizelgesine yerleştirilir. Türkçe büyük harfler korunur. Altyazı biçimi sabit kurgu profilinden gelir; konuşulan kelime vurgulanır.
- Çıktı 1080×1920, 30 fps H.264/AAC MP4'tür. Video tamamen decode edilir; ses akışı ve süre farkı kontrol edilir. Varsayılan müziksizdir. Kurgu profilinde müzik açılırsa basit özgün fon konuşmaya göre kısılır. Kendi müziğin `music_file` ile seçilebilir.

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

`data/runs/` çıktıları, `data/cache-v2/` ses cache'ini, `data/events/` kullanılan olayları, `data/publishing/` yayın kayıtlarını tutar. Bunlar Git'e eklenmez. Önizleme artifact'ları 30 gün; cache kayıtları GitHub'ın cache politikası boyunca saklanır.

## API belgeleri

- [Gemini video ve YouTube analizi](https://ai.google.dev/gemini-api/docs/video-understanding)
- [Gemini Türkçe ses üretimi](https://ai.google.dev/gemini-api/docs/speech-generation)
- [Google Search ile araştırma](https://ai.google.dev/gemini-api/docs/google-search)
- [Groq konuşma çözümleme](https://console.groq.com/docs/speech-to-text)
- [GitHub Git Trees API](https://docs.github.com/en/rest/git/trees?apiVersion=2022-11-28)

## Studio 4: bağımsız kaynak araştırması

Gemini Google Search araç çağrısı kaldırıldı. DDGS ile kaynaklar bulunur, gerçek sayfa metinleri indirilir ve konu seçimine verilir. Konu adaylarında yalnız okunmuş kaynak URL adresleri kabul edilir. discovery.json arama ve erişim sonuçlarını saklar. Model ve ses kotaları geçerlidir. 2.5 Flash erişim hatası alındıysa GEMINI_MODEL değişkenini erişilebilir 3.7 Flash modeline ayarlayın.
