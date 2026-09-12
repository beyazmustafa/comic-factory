"""Offline timing, captions, media inspection and file integrity helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from PIL import ImageFont


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_word(word: str) -> str:
    value = word.replace("İ", "i").replace("I", "ı").casefold()
    value = "".join(
        c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c)
    )
    return re.sub(r"[^\w]", "", value).replace("ı", "i")


def align_words(
    narration: str, timestamps: list[dict], audio_duration: float | None = None
) -> tuple[list[dict], dict]:
    """Global alignment prevents a repeated word from skipping the rest of a sentence.

    Split/joined tokens are matched in both directions. Zero-duration ASR
    words retain their text but are not reliable timing anchors. Estimated
    timestamps lower timing coverage; malformed/out-of-order data is rejected
    for re-transcription, never silently sorted into a different sentence.
    """
    tokens = narration.split()
    if not tokens or not timestamps:
        raise ValueError("Hizalama için metin ve kelime zamanları gerekli.")
    if audio_duration is not None and (
        not math.isfinite(audio_duration) or audio_duration <= 0
    ):
        raise ValueError("Ses süresi pozitif ve sonlu olmalı.")
    prepared = []
    previous_start = -1.0
    ignored_punctuation = 0
    for index, item in enumerate(timestamps):
        if not normalize_word(str(item.get("word", ""))):
            ignored_punctuation += 1
            continue
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{index + 1}. ASR kelimesinde zaman bilgisi eksik."
            ) from error
        if (
            not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end)
            or start < previous_start
        ):
            raise ValueError(
                f"Geçersiz veya sırasız ses zamanları: ASR kelimesi {index + 1}."
            )
        if audio_duration is not None:
            # Whisper timestamps have 20 ms resolution; allow only that rounding.
            if end > audio_duration + 0.02:
                raise ValueError(f"{index + 1}. ASR kelimesi kayıt süresinin dışında.")
            start, end = min(start, audio_duration), min(end, audio_duration)
        prepared.append({"word": str(item["word"]), "start": start, "end": end})
        previous_start = start
    if not prepared:
        raise ValueError("Konuşma tanıma sonucu zamanlanabilir kelime içermiyor.")
    timestamps = prepared
    a = [normalize_word(word) for word in tokens]
    b = [normalize_word(str(item["word"])) for item in timestamps]
    n, m = len(a), len(b)
    costs = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    parents = {}
    costs[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            value = costs[i][j]
            options = []
            if i < n:
                options.append((i + 1, j, value + 1.0, 0.0))
            if j < m:
                options.append((i, j + 1, value + 0.9, 0.0))
            if i < n and j < m:
                similarity = SequenceMatcher(None, a[i], b[j]).ratio()
                options.append(
                    (i + 1, j + 1, value + 1.8 * (1 - similarity), similarity)
                )
                if j + 1 < m:
                    merged = SequenceMatcher(None, a[i], b[j] + b[j + 1]).ratio()
                    if merged >= 0.96:
                        options.append(
                            (i + 1, j + 2, value + 1.8 * (1 - merged) + 0.10, merged)
                        )
                if i + 1 < n:
                    joined = SequenceMatcher(None, a[i] + a[i + 1], b[j]).ratio()
                    if joined >= 0.96:
                        options.append(
                            (i + 2, j + 1, value + 1.8 * (1 - joined) + 0.10, joined)
                        )
            for ni, nj, cost, similarity in options:
                if cost < costs[ni][nj]:
                    costs[ni][nj] = cost
                    parents[ni, nj] = (i, j, similarity)
    anchors = {}
    reliable_timing = set()
    scores = [0.0] * n
    i, j = n, m
    while i or j:
        pi, pj, similarity = parents[i, j]
        if i > pi and j > pj and similarity >= 0.5:
            start = timestamps[pj]["start"]
            end = max(word["end"] for word in timestamps[pj:j])
            weight = sum(max(1, len(word)) for word in a[pi:i])
            cursor = start
            for position in range(pi, i):
                scores[position] = similarity
                boundary = cursor + (end - start) * max(1, len(a[position])) / weight
                if end > start:
                    anchors[position] = (cursor, boundary)
                    if i - pi == 1 and all(
                        word["end"] > word["start"] for word in timestamps[pj:j]
                    ):
                        reliable_timing.add(position)
                cursor = boundary
        i, j = pi, pj
    total_end = max(word["end"] for word in timestamps)
    aligned = []
    index = 0
    while index < n:
        if index in anchors:
            start, end = anchors[index]
            aligned.append({"word": tokens[index], "start": start, "end": end})
            index += 1
            continue
        stop = index
        while stop < n and stop not in anchors:
            stop += 1
        left = aligned[-1]["end"] if aligned else 0.0
        right = anchors[stop][0] if stop < n else total_end
        # Never manufacture timestamps beyond the recording.
        width = max(0.0, right - left) / (stop - index)
        for position in range(index, stop):
            start = left + (position - index) * width
            aligned.append(
                {"word": tokens[position], "start": start, "end": start + width}
            )
        index = stop
    previous_end = 0.0
    for index, item in enumerate(aligned):
        item["start"] = max(previous_end, item["start"])
        item["end"] = max(item["start"], min(total_end, item["end"]))
        if item["end"] <= item["start"]:
            reliable_timing.discard(index)
        item["timing_source"] = "asr" if index in reliable_timing else "estimated"
        previous_end = item["end"]
    coverage = sum(value >= 0.72 for value in scores) / n * 100
    similarity = sum(scores) / n * 100
    timing_coverage = len(reliable_timing) / n * 100
    score = min(coverage * 0.72 + similarity * 0.28, timing_coverage)
    if any(item["end"] <= item["start"] for item in aligned):
        score = min(score, 50.0)
    return aligned, {
        "coverage": coverage,
        "similarity": similarity,
        "timing_coverage": timing_coverage,
        "score": score,
        "estimated_word_indices": [i for i in range(n) if i not in reliable_timing],
        "unmatched_words": [tokens[i] for i in range(n) if scores[i] < 0.72],
        "zero_duration_asr_words": sum(
            word["start"] == word["end"] for word in timestamps
        ),
        "ignored_asr_punctuation": ignored_punctuation,
    }


def scene_timeline(
    scenes: list[dict], words: list[dict], duration: float, fps: int = 30
) -> list[tuple[float, float]]:
    counts = [len(scene["narration"].split()) for scene in scenes]
    if not counts or min(counts) <= 0 or sum(counts) != len(words):
        raise ValueError("Sahne metinleri ile kelime zamanları eşleşmiyor.")
    frames = round(duration * fps)
    boundaries = [0]
    cursor = 0
    for count in counts[:-1]:
        cursor += count
        boundary = (float(words[cursor - 1]["end"]) + float(words[cursor]["start"])) / 2
        boundaries.append(round(boundary * fps))
    boundaries.append(frames)
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("Sahne zamanları sırasız veya ses süresinin dışında.")
    return [
        (left / fps, right / fps) for left, right in zip(boundaries, boundaries[1:])
    ]


def ass_time(seconds: float) -> str:
    total = max(0, round(seconds * 100))
    hours, total = divmod(total, 360000)
    minutes, total = divmod(total, 6000)
    seconds, centiseconds = divmod(total, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def caption_font(size: int):
    for filename in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ):
        try:
            return ImageFont.truetype(filename, size)
        except OSError:
            pass
    raise ValueError("Türkçe altyazı için DejaVu Sans veya Arial Bold fontu gerekli.")


def escape_ass(text: str) -> str:
    return (
        text.replace("\\", "＼").replace("{", "(").replace("}", ")").replace("\n", " ")
    )


def create_captions(words: list[dict], output: Path, offset: float = 0.0) -> Path:
    """Sliding current + next word; spoken word is amber, next word is white."""
    font_name = (
        "Arial" if Path("C:/Windows/Fonts/arialbd.ttf").exists() else "DejaVu Sans"
    )
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        f"Style: Main,{font_name},60,&H00FFFFFF,&H00FFFFFF,&H00101014,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,140,140,360,1",
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    for index, item in enumerate(words):
        current = (
            str(item["word"]).translate(str.maketrans({"i": "İ", "ı": "I"})).upper()
        )
        following = (
            str(words[index + 1]["word"])
            .translate(str.maketrans({"i": "İ", "ı": "I"}))
            .upper()
            if index + 1 < len(words)
            else ""
        )
        for size in range(60, 27, -2):
            text = (current + " " + following).strip()
            if caption_font(size).getlength(text) <= 760:
                break
        else:
            following = ""
            size = 36
            while size > 18 and caption_font(size).getlength(current) > 760:
                size -= 2
            if caption_font(size).getlength(current) > 760:
                raise ValueError("Tek kelime altyazı alanına sığmıyor.")
        start = max(0.0, float(item["start"]) + offset)
        end = float(item["end"]) + offset
        if index + 1 < len(words):
            end = min(end, float(words[index + 1]["start"]) + offset)
        if end <= start:
            continue
        text = f"{{\\fs{size}\\1c&H0046C7FF&}}{escape_ass(current)}"
        if following:
            text += f" {{\\1c&H00FFFFFF&}}{escape_ass(following)}"
        lines.append(
            f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Main,,0,0,0,,{text}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def ffmpeg_binary() -> str:
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def inspect_media(path: Path) -> dict:
    binary = shutil.which("ffprobe")
    if not binary:
        raise ValueError("FFprobe bulunamadı; FFmpeg kurulumu gerekli.")
    process = subprocess.run(
        [
            binary,
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if process.returncode:
        raise ValueError(f"Medya okunamadı: {path.name}: {process.stderr[-600:]}")
    return json.loads(process.stdout)


def check_video(video: Path, audio: Path) -> dict:
    metadata, reference = inspect_media(video), inspect_media(audio)
    streams = metadata.get("streams", [])
    picture = next((item for item in streams if item.get("codec_type") == "video"), {})
    sound = next((item for item in streams if item.get("codec_type") == "audio"), {})
    duration = float(metadata.get("format", {}).get("duration", 0))
    audio_duration = float(reference.get("format", {}).get("duration", 0))
    decode = subprocess.run(
        [
            ffmpeg_binary(),
            "-v",
            "error",
            "-xerror",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        timeout=600,
    )
    delta = abs(duration - audio_duration)
    passed = bool(
        decode.returncode == 0
        and picture.get("width") == 1080
        and picture.get("height") == 1920
        and picture.get("codec_name") == "h264"
        and picture.get("pix_fmt") == "yuv420p"
        and sound.get("codec_name") == "aac"
        and duration > 0
        and delta <= 0.12
    )
    return {
        "passed": passed,
        "width": picture.get("width", 0),
        "height": picture.get("height", 0),
        "has_audio": bool(sound),
        "video_duration": duration,
        "audio_duration": audio_duration,
        "duration_delta": delta,
        "decode_returncode": decode.returncode,
    }
