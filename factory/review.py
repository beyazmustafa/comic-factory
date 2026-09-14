import html
import json


def write_review(directory):
    def read(name, default):
        path = directory / name
        return (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
        )

    esc = lambda value: html.escape(str(value), quote=True)
    run, story, bundle, selection, report = (
        read(name, {})
        for name in (
            "run.json",
            "story.json",
            "story_bundle.json",
            "voice_selection.json",
            "quality_review.json",
        )
    )
    panels = {p["id"]: p for p in bundle.get("panels", [])}
    rows = []
    for i, shot in enumerate(story.get("shots", []), 1):
        panel = panels.get(shot["panel_id"], {})
        rows.append(
            f'<article><img loading="lazy" src="{esc(panel.get("file", ""))}" alt="Sahne {i}"><div><small>SAHNE {i}</small><p>{esc(shot["narration"])}</p><a target="_blank" rel="noopener" href="{esc(panel.get("source_url", ""))}">Kaynak sayfası ↗</a></div></article>'
        )
    voices = [
        f'<div class="audition"><b>{esc(path.stem)}</b><audio controls preload="none" src="voice_audition/{esc(path.name)}"></audio></div>'
        for path in sorted((directory / "voice_audition").glob("*.wav"))
    ]
    status = {
        "ready": "Önizleme hazır",
        "failed": "Üretim tamamlanamadı",
        "voice_ready": "Ses örnekleri hazır",
    }.get(run.get("status"), "Çalışma kaydı")
    video = (
        '<video controls playsinline preload="metadata" src="video.mp4"></video>'
        if (directory / "video.mp4").exists()
        else ""
    )
    error = f'<p class="error">{esc(run["error"])}</p>' if run.get("error") else ""
    page = f"""<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Comic Factory · Önizleme</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#0c1018;color:#f1f3f7;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:32px 22px}}small,.muted{{color:#a4adc1}}h1{{font-size:clamp(28px,4vw,48px);line-height:1.15}}.top{{display:grid;grid-template-columns:minmax(220px,340px) 1fr;gap:38px;align-items:start}}video{{width:100%;max-height:76vh;border-radius:14px;background:#000}}a{{color:#9ef0d1}}.error{{background:#3c2029;padding:18px;border-radius:12px}}article{{display:grid;grid-template-columns:180px 1fr;gap:24px;background:#151c28;border:1px solid #253047;border-radius:14px;margin:18px 0;padding:16px}}article img{{width:100%;height:190px;object-fit:contain}}.audition{{display:flex;align-items:center;gap:20px;margin:12px 0;flex-wrap:wrap}}audio{{max-width:100%}}.tag{{display:inline-block;border:1px solid #486052;padding:5px 12px;border-radius:20px;color:#9ef0d1}}pre{{white-space:pre-wrap;word-break:break-word;background:#151c28;padding:18px}}@media(max-width:650px){{.top,article{{grid-template-columns:1fr}}video{{max-height:64vh}}article img{{height:260px}}}}</style>
<main><p class="muted">COMIC FACTORY / TÜRKÇE ÇİZGİ ROMAN HİKÂYELERİ</p><div class="top">{video}<div><span class="tag">{esc(status)}</span><h1>{esc(story.get("title", "Çalışma kaydı"))}</h1><p>{esc(story.get("description", ""))}</p>{error}<p>Seçilen ses: <b>{esc(selection.get("voice", "Henüz seçilmedi"))}</b></p><p>{esc(report.get("summary", ""))}</p><p class="muted">Üretilen videonun kalite puanları model değerlendirmesidir. Son kararı önizlemeyi dinleyerek ver.</p><a href="editing_profile.json">Kurgu ayarları</a></div></div><h2>Ses karşılaştırması</h2>{"".join(voices) or "<p>Bu kayıtta ses karşılaştırma örneği yok.</p>"}<h2>Sahneler ve kaynakları</h2>{"".join(rows)}<details><summary>Kontrol sonuçları</summary><pre>{esc(json.dumps(report, ensure_ascii=False, indent=2))}</pre></details></main></html>"""
    (directory / "review.html").write_text(page, encoding="utf-8")
