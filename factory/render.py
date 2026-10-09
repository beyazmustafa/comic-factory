import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageFont, ImageDraw
from . import fx
from .api import FactoryError
from .core import (
    ass_time,
    escape_ass,
    ffmpeg_binary,
    scene_timeline,
    check_video,
    normalize_word,
    save_json,
)
from .voice import RATE, write_wave

WIDTH, HEIGHT, FPS = 1080, 1920, 30


def command(args, directory, label, timeout=900):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        path = directory / "diagnostics" / (label + ".log")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.stderr[-18000:], encoding="utf-8")
        raise FactoryError(f"{label} başarısız; ayrıntı {path.name} dosyasında.")


def font(style, size):
    name = (
        "DejaVuSansCondensed-Bold.ttf"
        if style["font_style"] == "condensed_heavy"
        else "DejaVuSans-Bold.ttf"
    )
    for path in (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ):
        if path.is_file():
            family = (
                "Arial"
                if path.name == "arialbd.ttf"
                else "DejaVu Sans" + (" Condensed" if "Condensed" in path.name else "")
            )
            return ImageFont.truetype(str(path), size), family
    raise FactoryError("Türkçe altyazı fontu bulunamadı; DejaVu Sans gerekli.")


def upper(text, language="en"):
    """Uppercase with Turkish dotted/dotless I only for Turkish captions."""
    if language == "tr":
        return text.translate(str.maketrans({"i": "İ", "ı": "I"})).upper()
    return text.upper()


def ass_color(value):
    value = value.lstrip("#")
    return "&H00" + value[4:6] + value[2:4] + value[:2] + "&"


def captions(words, style, path, offset=0, hook=""):
    _, family = font(style, int(style["font_size"]))
    normal, active, stroke = (
        ass_color(style[key]) for key in ("text_color", "active_color", "stroke_color")
    )
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {WIDTH}
PlayResY: {HEIGHT}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Main,{family},{style["font_size"]},{normal},{normal},{stroke},&H60000000,-1,0,0,0,100,100,0,0,1,{style["stroke_width"]},{style.get("shadow_depth", 5)},5,100,140,260,1
Style: Hook,{family},100,&H000AD6FF&,&H000AD6FF&,&H00000000&,&H80000000&,-1,0,0,0,100,100,1,0,1,7,6,5,40,40,40,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    lines = [header]
    if hook:
        lines.extend(hook_events(hook, style, style.get("language", "en")))
    center_x = round(WIDTH * style["caption_x"])
    available = min(820, 2 * (center_x - 80), 2 * (WIDTH - 110 - center_x))
    for i, word in enumerate(words):
        emphasis = style.get("emphasis_colors", {}).get(word.get("emphasis"))
        current_color = ass_color(emphasis) if emphasis else active
        mode, count = style["caption_mode"], style["caption_words"]
        if mode == "single":
            first, last = i, i + 1
        elif mode == "current_next":
            first, last = i, min(len(words), i + 2)
        else:
            first = i // count * count
            last = min(len(words), first + count)
        language = style.get("language", "en")
        texts = [
            upper(str(w["word"]), language) if style["uppercase"] else str(w["word"])
            for w in words[first:last]
        ]
        size = int(style["font_size"])
        if word.get("impact"):
            size = round(size * 1.4)
        while (
            size >= 24 and font(style, size)[0].getlength(" ".join(texts)) > available
        ):
            size -= 2
        if size < 24:
            raise FactoryError("Altyazı güvenli ekran alanına sığmıyor.")
        text = " ".join(
            "{\\1c" + (current_color if first + j == i else normal) + "}" + escape_ass(token)
            for j, token in enumerate(texts)
        )
        start, end = max(0, word["start"] + offset), word["end"] + offset
        if i + 1 < len(words):
            next_start = words[i + 1]["start"] + offset
            end = next_start if 0 <= next_start - end < 0.18 else min(end, next_start)
        if end <= start or ass_time(end) == ass_time(start):
            raise FactoryError("Sıfır süreli altyazı kabul edilmiyor.")
        pop = style.get("word_pop_ms", 70)
        effect = f"\\fscx86\\fscy86\\t(0,{pop},\\fscx100\\fscy100)\\fad(25,0)" if pop else ""
        if word.get("impact"):
            # Impact word: slams in bigger with a shiver, white with a red edge.
            effect = (f"\\fscx150\\fscy150\\t(0,90,\\fscx100\\fscy100)\\frz-2\\t(90,170,\\frz2)\\t(170,250,\\frz0)"
                      f"\\1c&H00FFFFFF&\\3c&H002020E0&\\bord9")
        text = (
            f"{{\\an5\\pos({center_x},{round(HEIGHT * style['caption_y'])})\\fs{size}{effect}}}"
            + text
        )
        lines.append(
            f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Main,,0,0,0,,{text}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


HOOK_SECONDS = 1.15
HOOK_Y = 0.30  # vertical centre of the title card band (fraction of height)


def hook_lines(text, style, max_width=940):
    """Split a hook line into at most two balanced lines that fit the width."""
    words = str(text).split()
    if not words:
        return [], 0
    size = 118 if len(" ".join(words)) <= 18 else 104
    while size >= 56:
        measure = font(style, size)[0]
        if measure.getlength(" ".join(words)) <= max_width:
            return [" ".join(words)], size
        if size > 92:
            size -= 6  # prefer one big line while it can still be big
            continue
        best, best_gap = None, None
        for split in range(1, len(words)):
            first, second = " ".join(words[:split]), " ".join(words[split:])
            if max(measure.getlength(first), measure.getlength(second)) <= max_width:
                gap = abs(measure.getlength(first) - measure.getlength(second))
                if best_gap is None or gap < best_gap:
                    best, best_gap = [first, second], gap
        if best:
            return best, size
        size -= 6
    return [" ".join(words)], 56


def hook_events(text, style, language):
    """ASS events for the opening title card: dark band + big outlined words."""
    lines, size = hook_lines(upper(text, language), style)
    if not lines:
        return []
    band_h = round(size * 1.35 * len(lines) + 90)
    top = round(HEIGHT * HOOK_Y - band_h / 2)
    end = ass_time(HOOK_SECONDS)
    band = (f"Dialogue: 1,0:00:00.00,{end},Hook,,0,0,0,,{{\\an7\\pos(0,0)\\1c&H000000&\\1a&H48&\\bord0\\shad0\\fad(0,180)\\p1}}"
            f"m 0 {top} l {WIDTH} {top} l {WIDTH} {top + band_h} l 0 {top + band_h}{{\\p0}}")
    colour = ass_color(style.get("hook_color", "#FFD60A"))
    body = "\\N".join(escape_ass(line) for line in lines)
    words = (f"Dialogue: 2,0:00:00.00,{end},Hook,,0,0,0,,{{\\an5\\pos({WIDTH // 2},{round(HEIGHT * HOOK_Y)})\\fs{size}\\1c{colour}"
             f"\\bord7\\shad6\\fscx108\\fscy108\\t(0,120,\\fscx100\\fscy100)\\fad(0,180)}}{body}")
    return [band, words]


def thumbnail(base_png, text, style, language, output):
    """Same card as the first frame, saved as the upload thumbnail."""
    with Image.open(base_png) as source:
        picture = source.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    lines, size = hook_lines(upper(text, language), style)
    if lines:
        size = round(size * 1.08)
        measure = font(style, size)[0]
        band_h = round(size * 1.35 * len(lines) + 90)
        top = round(HEIGHT * HOOK_Y - band_h / 2)
        overlay = Image.new("RGBA", picture.size, (0, 0, 0, 0))
        ImageDraw.Draw(overlay).rectangle([0, top, WIDTH, top + band_h], fill=(0, 0, 0, 180))
        picture = Image.alpha_composite(picture.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(picture)
        y = top + 45
        for line in lines:
            width = measure.getlength(line)
            x = (WIDTH - width) / 2
            draw.text((x, y), line, font=measure, fill=style.get("hook_color", "#FFD60A"),
                      stroke_width=max(4, size // 14), stroke_fill="#000000")
            y += round(size * 1.35)
    picture.save(output, "JPEG", quality=90, optimize=True)
    return output


_GRADE_CACHE = {}


def cinematic(picture):
    """Photo grade for the space channel: a little more contrast and colour,
    a soft vignette and a dark gradient behind the caption band so yellow
    words read on bright nebulae. Cached masks: one per output size."""
    picture = ImageEnhance.Contrast(picture).enhance(1.08)
    picture = ImageEnhance.Color(picture).enhance(1.12)
    picture = picture.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=3))
    key = picture.size
    if key not in _GRADE_CACHE:
        w, h = key
        yy, xx = np.mgrid[0:h, 0:w]
        dx, dy = (xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)
        radial = np.clip(np.sqrt(dx * dx + dy * dy) - 0.55, 0, 1) / 0.75
        vignette = 1 - 0.38 * radial ** 1.6
        band = np.clip((yy / h - 0.50) / 0.28, 0, 1)
        gradient = 1 - 0.22 * np.sin(np.clip(band, 0, 1) * np.pi)
        _GRADE_CACHE[key] = (vignette * gradient).astype(np.float32)
    mask = _GRADE_CACHE[key]
    array = np.asarray(picture, dtype=np.float32)
    array *= mask[:, :, None]
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def polish(picture):
    """Make an upscaled old scan read crisp without changing its content:
    gentle levels, a touch of colour, and an unsharp mask tuned for line art."""
    picture = ImageOps.autocontrast(picture, cutoff=0.4, preserve_tone=True)
    picture = ImageEnhance.Color(picture).enhance(1.08)
    picture = ImageEnhance.Contrast(picture).enhance(1.05)
    return picture.filter(ImageFilter.UnsharpMask(radius=1.6, percent=85, threshold=2))


def base_image(panel, style, directory, output):
    if panel.get("framing") == "fill":
        # Photographs (space source): full-bleed 9:16 crop around the focus
        # point; no blurred page behind, nothing but the image.
        with Image.open(directory / panel["file"]) as source:
            picture = source.convert("RGB")
        focus = panel.get("focus") or [0.5, 0.5]
        try:
            centering = (min(1.0, max(0.0, float(focus[0]))), min(1.0, max(0.0, float(focus[1]))))
        except (TypeError, ValueError, IndexError):
            centering = (0.5, 0.5)
        framed = ImageOps.fit(picture, (WIDTH * 2, HEIGHT * 2), method=Image.Resampling.LANCZOS, centering=centering)
        framed = cinematic(framed)
        framed.save(output, "PNG", compress_level=1)
        return
    with Image.open(directory / panel["page_file"]) as source:
        page = source.convert("RGB")
    with Image.open(directory / panel["file"]) as source:
        picture = source.convert("RGB")
    if style["background"] == "solid":
        background = Image.new("RGB", (WIDTH, HEIGHT), style["background_color"])
    else:
        background = ImageOps.fit(
            page, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS
        )
        if style["background"] == "blurred_page":
            background = background.filter(ImageFilter.GaussianBlur(26))
    if style["panel_framing"] == "page":
        background = ImageOps.fit(
            page, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS
        )
    else:
        size = (
            round(WIDTH * style["panel_max_width"]),
            round(HEIGHT * style["panel_max_height"]),
        )
        framing = style["panel_framing"]
        target_ratio = size[0] / size[1]
        if framing == "fill" and picture.width / picture.height > target_ratio * 1.35:
            # A wide panel forced into 9:16 loses faces and balloons at the
            # sides. Crop it to a square around its focus point (when the
            # panel describer gave one) so it fills the width large, and show
            # that over the blurred page instead of a thin strip.
            ratio = picture.width / picture.height
            focus = panel.get("focus")
            if ratio >= 1.45:
                width = round(picture.height * (1.0 if ratio >= 1.8 else 1.2))
                fx = float(focus[0]) if isinstance(focus, (list, tuple)) and len(focus) == 2 else 0.5
                left = max(0, min(picture.width - width, round(fx * picture.width - width / 2)))
                picture = picture.crop((left, 0, left + width, picture.height))
            framing = "contain"
            if style["background"] == "page_fill":
                background = background.filter(ImageFilter.GaussianBlur(22))
        if framing == "fill":
            picture = ImageOps.fit(picture, size, method=Image.Resampling.LANCZOS)
        else:
            scale = min(size[0] / picture.width, size[1] / picture.height, 2.5)
            picture = picture.resize(
                (
                    max(1, round(picture.width * scale)),
                    max(1, round(picture.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
        picture = polish(picture)
        top = max(
            0,
            min(
                HEIGHT - picture.height,
                round(HEIGHT * style["panel_center_y"] - picture.height / 2),
            ),
        )
        background.paste(picture, ((WIDTH - picture.width) // 2, top))
    background.resize((WIDTH * 2, HEIGHT * 2), Image.Resampling.LANCZOS).save(
        output, "PNG", compress_level=1
    )


def motion_filter(motion, frames, amount):
    progress = f"on/{max(1, frames - 1)}"
    if motion == "pull":
        zoom = f"1+{amount}*(1-{progress})"
    elif motion == "hold":
        zoom = "1"
    elif motion in {"left", "right", "up", "down"}:
        zoom = str(1 + amount)
    else:
        zoom = f"1+{amount}*{progress}"
    x = "(iw-iw/zoom)/2"
    if motion == "right":
        x = f"(iw-iw/zoom)*{progress}"
    elif motion == "left":
        x = f"(iw-iw/zoom)*(1-{progress})"
    y = "(ih-ih/zoom)/2"
    if motion == "down":
        y = f"(ih-ih/zoom)*{progress}"
    elif motion == "up":
        y = f"(ih-ih/zoom)*(1-{progress})"
    if frames > round(3.0 * FPS) and motion != "hold":
        # Retention research: a visual change every ~2 s. Long shots get a
        # small punch-in at the midpoint so no frame sits still for 3+ s.
        zoom = f"({zoom})+0.045*gte(on,{frames // 2})"
    return f"zoompan=z='{zoom}':x='{x}':y='{y}':d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS},setsar=1,format=yuv420p"


def space_drone(path, duration):
    """Procedural cinematic bed for the space channel: detuned low pads that
    swell slowly, a sub-bass breath every ~11 s, no beat, no melody — a floor
    under the narration (ducked by the sidechain), never a song."""
    rng = np.random.default_rng(7)
    n = round(duration * RATE)
    t = np.arange(n, dtype=np.float64) / RATE
    output = np.zeros(n)
    for base in (55.0, 82.41, 110.0):
        for detune in (-0.6, 0.0, 0.7):
            freq = base * (1 + detune / 100)
            phase = rng.uniform(0, 2 * np.pi)
            lfo = 0.5 + 0.5 * np.sin(2 * np.pi * t / rng.uniform(13, 23) + phase)
            output += 0.035 * lfo * np.sin(2 * np.pi * freq * t + 0.4 * np.sin(2 * np.pi * 0.07 * t))
    swell = np.clip(np.sin(2 * np.pi * t / 11.0), 0, 1) ** 3
    output += 0.05 * swell * np.sin(2 * np.pi * 36.7 * t)
    shimmer = 0.012 * np.sin(2 * np.pi * 440.0 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * t / 17.0))
    output += shimmer * (0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t))
    # One-pole low-pass to keep it soft, then fade in/out.
    alpha = 0.08
    for i in range(1, n):
        output[i] = output[i - 1] + alpha * (output[i] - output[i - 1])
    output *= np.minimum(1, t / 2.5) * np.minimum(1, np.maximum(0, duration - t) / 3)
    peak = np.max(np.abs(output)) or 1
    write_wave(path, np.clip(output / peak * 0.5 * 32767, -32768, 32767).astype("<i2"))


def original_music(path, duration):
    t = np.arange(round(duration * RATE), dtype=np.float64) / RATE
    output = np.zeros_like(t)
    for i, frequency in enumerate((110.0, 87.307, 130.813, 97.999)):
        block = np.floor(t / 8).astype(int) % 4 == i
        envelope = np.sin(np.pi * (t % 8) / 8) ** 0.7
        for ratio, gain in (
            (1, 0.08),
            (1.189207, 0.025),
            (1.498307, 0.022),
            (2, 0.012),
        ):
            output += (
                block * envelope * gain * np.sin(2 * np.pi * frequency * ratio * t)
            )
    beat = t % 0.8
    output += 0.03 * np.exp(-beat * 18) * np.sin(2 * np.pi * 55 * beat)
    output *= np.minimum(1, t / 1.5) * np.minimum(1, np.maximum(0, duration - t) / 2)
    write_wave(path, np.clip(output * 32767, -32768, 32767).astype("<i2"))


def build(api, story, panels, style, audio_path, words, duration):
    directory = api.directory
    work = directory / "render"
    work.mkdir(exist_ok=True)
    caption_words, cursor = [], 0
    for shot in story["shots"]:
        count = len(shot["narration"].split())
        impact = normalize_word(str(shot.get("impact_word") or ""))
        marked = False
        for w in words[cursor:cursor + count]:
            hit = bool(impact) and not marked and normalize_word(str(w["word"])) == impact
            marked = marked or hit
            caption_words.append({**w, "emphasis": shot.get("emphasis", "normal"), "impact": hit})
        cursor += count
    if cursor != len(words):
        raise FactoryError("Sahne ve altyazı kelime sayısı uyuşmuyor.")
    language = getattr(api.settings, "language", "en")
    hook = str(story.get("hook_card") or "").strip()
    subtitle = captions(
        caption_words, {**style, "language": language},
        directory / "captions.ass", api.settings.caption_offset, hook=hook
    )
    timeline = scene_timeline(story["shots"], words, duration, FPS)
    lookup = {p["id"]: p for p in panels}
    pad = 0 if style["transition"] == "cut" else style["transition_frames"]
    clips, rows = [], []
    common = [ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y"]
    codec = [
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "15",
        "-threads",
        "2",
    ]
    graphics = {}
    for i, (shot, (start, end)) in enumerate(zip(story["shots"], timeline)):
        api.check()
        panel = lookup[shot["panel_id"]]
        picture, clip = work / f"shot_{i:03}.png", work / f"shot_{i:03}.mp4"
        frames = round((end - start) * FPS) + (pad if i + 1 < len(timeline) else 0)
        effect = shot.get("effect") if shot.get("effect") in fx.EFFECT_TYPES else None
        chain = []
        inputs = []
        if panel.get("video") and (directory / panel["video"]).is_file():
            # Real footage: loop it if the beat is longer than the clip.
            inputs = ["-stream_loop", "-1", "-ss", f"{float(panel.get('clip_start', 0)):.2f}",
                      "-t", f"{frames / FPS + 0.5:.3f}", "-i", str(directory / panel["video"])]
            chain.append(fx.video_filter(frames, panel.get("focus"), min(0.12, style["zoom_amount"] * 0.6)))
            base_image(panel, style, directory, picture)  # still frame for the thumbnail
        else:
            base_image(panel, style, directory, picture)
            inputs = ["-i", str(picture)]
            chain.append(motion_filter(shot["motion"], frames, style["zoom_amount"]))
        chain.extend(fx.effect_filters(effect, frames))
        graphic = fx.validate_graphic(shot.get("graphic"))
        if graphic:
            graphics[i] = graphic
            cards = fx.graphic_frames(graphic, frames, work / f"cards_{i:03}")
            inputs += ["-framerate", str(FPS), "-i", str(cards / "card_%04d.png")]
            graph = "[0:v]" + ",".join(chain) + "[bg];[bg][1:v]overlay=0:0:shortest=1,format=yuv420p[v]"
            args = common + inputs + ["-filter_complex_threads", "1", "-filter_complex", graph, "-map", "[v]"]
        else:
            args = common + inputs + ["-vf", ",".join(chain)]
        command(args + ["-frames:v", str(frames)] + codec + [str(clip)], directory, f"render_shot_{i:03}")
        clips.append(clip)
        if i == 0:
            try:
                thumbnail(picture, hook, {**style, "language": language}, language, directory / "thumbnail.jpg")
            except Exception as error:  # noqa: BLE001 - thumbnail is optional
                print(f"Kapak üretilemedi: {str(error)[:120]}", flush=True)
        rows.append(
            {
                **shot,
                "start": start,
                "end": end,
                "source_url": panel["source_url"],
                "panel": panel["file"],
            }
        )
    if pad:
        joined = []
        for i, clip in enumerate(clips):
            api.check()
            output = work / f"joined_{i:03}.mp4"
            frames = round((timeline[i][1] - timeline[i][0]) * FPS)
            if i:
                previous_frames = round((timeline[i - 1][1] - timeline[i - 1][0]) * FPS)
                effect = {"slide": "slideleft", "whip": "slideup"}.get(style["transition"], "fade")
                graph = f"[0:v]trim=start_frame={previous_frames},setpts=PTS-STARTPTS[tail];[1:v]setpts=PTS-STARTPTS[next];[tail][next]xfade=transition={effect}:duration={pad / FPS}:offset=0[v]"
                if style["transition"] == "whip":
                    graph = graph[:-3] + f"[transition];[transition]gblur=sigma=1:sigmaV=18:steps=2:enable='lt(t,{pad / FPS})'[v]"
                args = common + [
                    "-i",
                    str(clips[i - 1]),
                    "-i",
                    str(clip),
                    "-filter_complex_threads",
                    "1",
                    "-filter_complex",
                    graph,
                    "-map",
                    "[v]",
                ]
            else:
                args = common + ["-i", str(clip)]
            command(
                args + ["-frames:v", str(frames)] + codec + [str(output)],
                directory,
                f"join_{i:03}",
            )
            joined.append(output)
        clips = joined
    listing = work / "concat.txt"
    listing.write_text("\n".join(f"file '{p.name}'" for p in clips), encoding="utf-8")
    silent = work / "silent.mp4"
    command(
        common
        + ["-f", "concat", "-safe", "1", "-i", str(listing), "-c", "copy", str(silent)],
        directory,
        "join_shots",
    )
    # Drain normalization look-ahead before using the audio in a sidechain graph.
    # Otherwise very short inputs can produce no audio frames at all.
    normalized = work / "normalized.wav"
    command(
        common
        + [
            "-i",
            str(audio_path),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=9",
            "-ar",
            "48000",
            str(normalized),
        ],
        directory,
        "normalize_audio",
    )
    music = None
    if api.settings.music_file:
        music = Path(api.settings.music_file)
        if not music.is_absolute():
            music = Path(__file__).resolve().parents[1] / music
        if not music.is_file():
            raise FactoryError("Seçilen müzik dosyası bulunamadı.")
    elif style["music_present"]:
        music = work / "original_underscore.wav"
        if getattr(api.settings, "channel_theme", "") == "space":
            space_drone(music, duration)
        else:
            original_music(music, duration)
    sfx_path = None
    if getattr(api.settings, "channel_theme", "") == "space":
        try:
            sfx_path = fx.sfx_track(work / "sfx.wav", story["shots"], timeline, duration, graphics)
        except Exception as error:  # noqa: BLE001 - sound design is optional
            print(f"Ses tasarımı atlandı: {str(error)[:120]}", flush=True)
            sfx_path = None
    args = common + ["-i", str(silent), "-i", str(normalized)]
    sfx_gain = getattr(api.settings, "sfx_gain_db", -14)
    if music and sfx_path:
        args += ["-stream_loop", "-1", "-i", str(music), "-i", str(sfx_path)]
        af = (f"[1:a]asplit=2[speech][side];[2:a]volume={api.settings.music_gain_db}dB[music];"
              f"[music][side]sidechaincompress=threshold=0.02:ratio=6:attack=15:release=200[ducked];"
              f"[3:a]volume={sfx_gain}dB[sfx];[speech][ducked][sfx]amix=inputs=3:duration=first:normalize=0,alimiter=limit=0.95:latency=1[a]")
    elif music:
        args += ["-stream_loop", "-1", "-i", str(music)]
        af = f"[1:a]asplit=2[speech][side];[2:a]volume={api.settings.music_gain_db}dB[music];[music][side]sidechaincompress=threshold=0.02:ratio=6:attack=15:release=200[ducked];[speech][ducked]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95:latency=1[a]"
    elif sfx_path:
        args += ["-i", str(sfx_path)]
        af = f"[2:a]volume={sfx_gain}dB[sfx];[1:a][sfx]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95:latency=1[a]"
    else:
        af = "[1:a]anull[a]"
    escaped = (
        str(subtitle.resolve())
        .replace("\\", "/")
        .replace(":", r"\:")
        .replace("'", r"\'")
    )
    output = directory / "video.mp4"
    args += [
        "-filter_complex_threads",
        "1",
        "-filter_complex",
        f"[0:v]ass=filename='{escaped}'[v];" + af,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-t",
        f"{duration:.6f}",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "15",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        "-r",
        str(FPS),
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-threads",
        "2",
        str(output),
    ]
    command(args, directory, "final_render", 1800)
    report = check_video(output, audio_path)
    save_json(directory / "technical_check.json", report)
    save_json(directory / "timeline.json", rows)
    if not report["passed"]:
        raise FactoryError(
            "Video teknik kontrolden geçmedi; technical_check.json kaydedildi."
        )
    return output, report
