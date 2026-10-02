import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageFont
from .api import FactoryError
from .core import (
    ass_time,
    escape_ass,
    ffmpeg_binary,
    scene_timeline,
    check_video,
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


def upper(text):
    return text.translate(str.maketrans({"i": "İ", "ı": "I"})).upper()


def ass_color(value):
    value = value.lstrip("#")
    return "&H00" + value[4:6] + value[2:4] + value[:2] + "&"


def captions(words, style, path, offset=0):
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
Style: Main,{family},{style["font_size"]},{normal},{normal},{stroke},&H80000000,-1,0,0,0,100,100,0,0,1,{style["stroke_width"]},1,5,100,140,260,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    lines = [header]
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
        texts = [
            upper(str(w["word"])) if style["uppercase"] else str(w["word"])
            for w in words[first:last]
        ]
        size = int(style["font_size"])
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
        text = (
            f"{{\\an5\\pos({center_x},{round(HEIGHT * style['caption_y'])})\\fs{size}}}"
            + text
        )
        lines.append(
            f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Main,,0,0,0,,{text}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def base_image(panel, style, directory, output):
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
            # sides; show it whole over the blurred page instead.
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
        picture = ImageEnhance.Sharpness(picture).enhance(1.08)
        top = max(
            0,
            min(
                HEIGHT - picture.height,
                round(HEIGHT * style["panel_center_y"] - picture.height / 2),
            ),
        )
        background.paste(picture, ((WIDTH - picture.width) // 2, top))
    background.resize((WIDTH * 2, HEIGHT * 2), Image.Resampling.LANCZOS).save(
        output, "JPEG", quality=95
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
    return f"zoompan=z='{zoom}':x='{x}':y='{y}':d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS},setsar=1,format=yuv420p"


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
        caption_words.extend({**w, "emphasis": shot.get("emphasis", "normal")}
                             for w in words[cursor:cursor + count])
        cursor += count
    if cursor != len(words):
        raise FactoryError("Sahne ve altyazı kelime sayısı uyuşmuyor.")
    subtitle = captions(
        caption_words, style, directory / "captions.ass", api.settings.caption_offset
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
        "veryfast",
        "-crf",
        "18",
        "-threads",
        "2",
    ]
    for i, (shot, (start, end)) in enumerate(zip(story["shots"], timeline)):
        api.check()
        panel = lookup[shot["panel_id"]]
        picture, clip = work / f"shot_{i:03}.jpg", work / f"shot_{i:03}.mp4"
        base_image(panel, style, directory, picture)
        frames = round((end - start) * FPS) + (pad if i + 1 < len(timeline) else 0)
        command(
            common
            + [
                "-i",
                str(picture),
                "-vf",
                motion_filter(shot["motion"], frames, style["zoom_amount"]),
                "-frames:v",
                str(frames),
            ]
            + codec
            + [str(clip)],
            directory,
            f"render_shot_{i:03}",
        )
        clips.append(clip)
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
        original_music(music, duration)
    args = common + ["-i", str(silent), "-i", str(normalized)]
    if music:
        args += ["-stream_loop", "-1", "-i", str(music)]
        af = f"[1:a]asplit=2[speech][side];[2:a]volume={api.settings.music_gain_db}dB[music];[music][side]sidechaincompress=threshold=0.02:ratio=6:attack=15:release=200[ducked];[speech][ducked]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95:latency=1[a]"
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
        "18",
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
