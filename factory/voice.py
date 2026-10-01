import base64
import hashlib
import json
import math
import os
import shutil
import wave
import subprocess
import sys
import numpy as np
from google.genai import types
from groq import Groq
from .api import FactoryError, ProviderOverloaded, SpeechFailure
from .core import align_words, ffmpeg_binary, file_hash, save_json

RATE = 24000
GEMINI_VOICES = ("Orus", "Gacrux", "Fenrir", "Puck")
# Gemini voice → Microsoft edge-tts voice used when Gemini TTS is unavailable.
EDGE_FALLBACK = {"Orus": "Ahmet", "Fenrir": "Ahmet", "Puck": "Ahmet", "Gacrux": "Emel"}
EDGE_VOICE_IDS = {"Ahmet": "tr-TR-AhmetNeural", "Emel": "tr-TR-EmelNeural"}
AUDITION_TEXT = "Bir kahramanın en büyük gücü, bir anda en korkunç düşmanına dönüşebilir. Thor bunu öğrendiğinde artık çok geçti. Çünkü asıl tehlike dışarıda değil, kendi bedeninin içindeydi. Peki bu noktaya nasıl geldi?"


def read_wave(path):
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (
            1,
            2,
            RATE,
        ):
            raise SpeechFailure("24 kHz mono PCM16 ses gerekli.")
        data = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").copy()
    if not len(data):
        raise SpeechFailure("Ses kaydı boş.")
    return data


def write_wave(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def trim_edges(samples):
    if not len(samples):
        raise SpeechFailure("Ses kaydı boş.")
    unit = RATE // 100
    levels = np.abs(samples.astype(np.float64)) / 32768
    rms = np.array(
        [
            np.sqrt(np.mean(levels[i : i + unit] ** 2))
            for i in range(0, len(samples), unit)
        ]
    )
    active = np.flatnonzero(rms > max(0.0015, float(rms.max()) * 0.015))
    if not len(active):
        raise SpeechFailure("Kayıt sessiz veya duyulamayacak kadar düşük.")
    start, end = (
        max(0, int(active[0] * unit - RATE * 0.08)),
        min(len(samples), int((active[-1] + 1) * unit + RATE * 0.12)),
    )
    return samples[start:end]  # Preserve pauses within speech.


def edge_synthesize(text, voice, path):
    """Free, keyless Microsoft neural TTS (edge-tts); output converted to 24 kHz PCM."""
    identifier = EDGE_VOICE_IDS.get(voice, voice)
    path.parent.mkdir(parents=True, exist_ok=True)
    media = path.with_suffix(".mp3")
    try:
        subprocess.run(
            [sys.executable, "-m", "edge_tts", "--voice", identifier, "--rate", "+4%",
             "--text", text, "--write-media", str(media)],
            check=True, capture_output=True, text=True, timeout=240,
        )
        subprocess.run(
            [ffmpeg_binary(), "-y", "-v", "error", "-i", str(media), "-ac", "1", "-ar", str(RATE),
             "-c:a", "pcm_s16le", str(path)],
            check=True, capture_output=True, text=True, timeout=240,
        )
    except (subprocess.SubprocessError, OSError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise SpeechFailure(f"edge-tts başarısız: {str(detail)[:300]}") from error
    finally:
        try:
            media.unlink()
        except OSError:
            pass
    write_wave(path, trim_edges(read_wave(path)))
    return path


def synthesize(api, text, voice, style, path):
    """Gemini TTS; on provider overload the matching edge-tts voice is used."""
    if voice in EDGE_VOICE_IDS:
        return edge_synthesize(text, voice, path)
    if getattr(api, "tts_fallback", False):
        return edge_synthesize(text, EDGE_FALLBACK.get(voice, "Ahmet"), path)
    try:
        return gemini_synthesize(api, text, voice, style, path)
    except ProviderOverloaded as error:
        api.tts_fallback = True
        api.note(f"Gemini TTS yanıt vermiyor ({str(error)[:120]}); edge-tts sesine geçildi.")
        return edge_synthesize(text, EDGE_FALLBACK.get(voice, "Ahmet"), path)


def gemini_synthesize(api, text, voice, style, path):
    prompt = f"""Read ONLY the exact Turkish text inside <transcript> once. No additions, omissions, translation, paraphrase, spoken instructions or music.
DELIVERY: {style.get("narrator_delivery", "")}
Fluent natural Turkish, confident comic-story energy, varied emphasis, short dramatic pauses, consistent narrator identity. Clear English proper names within Turkish. No newsreader monotone, shouting, growling or whispering. Roughly 125–150 Turkish words/minute.
<transcript>{text}</transcript>"""
    response = api.request(
        "Türkçe ses " + voice,
        lambda: api.client.models.generate_content(
            model=api.settings.tts_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice
                        )
                    )
                ),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        ),
    )
    blocks = []
    for candidate in response.candidates or []:
        for part in getattr(getattr(candidate, "content", None), "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if (
                inline
                and getattr(inline, "data", None)
                and str(getattr(inline, "mime_type", "")).startswith("audio/")
            ):
                blocks.append(
                    base64.b64decode(inline.data)
                    if isinstance(inline.data, str)
                    else inline.data
                )
        if blocks:
            break
    if not blocks:
        raise SpeechFailure("Ses modeli boş kayıt döndürdü.")
    data = b"".join(blocks)
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.startswith(b"RIFF"):
        path.write_bytes(data)
        samples = read_wave(path)
    else:
        if len(data) % 2:
            raise SpeechFailure("Eksik PCM örneği.")
        samples = np.frombuffer(data, dtype="<i2")
    write_wave(path, trim_edges(samples))
    return path


def transcribe(path, model, diagnostics):
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise SpeechFailure("GROQ_API_KEY bu çalışmaya aktarılmamış.")
    with (
        Groq(api_key=key, timeout=100, max_retries=2) as client,
        path.open("rb") as source,
    ):
        response = client.audio.transcriptions.create(
            file=source,
            model=model,
            language="tr",
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            temperature=0,
        )
    raw = response.model_dump(mode="json")
    save_json(diagnostics, {"model": model, "response": raw})
    return raw.get("words", [])


def align_clip(api, path, text, directory):
    duration = len(read_wave(path)) / RATE
    models = list(
        dict.fromkeys(
            [api.settings.whisper_model, "whisper-large-v3", "whisper-large-v3-turbo"]
        )
    )[:2]
    errors = []
    for i, model in enumerate(models):
        api.check()
        try:
            raw = transcribe(path, model, directory / f"asr_{i + 1}_raw.json")
            words, metrics = align_words(text, raw, duration)
            save_json(
                directory / f"asr_{i + 1}_alignment.json",
                {"words": words, "metrics": metrics},
            )
            print(f"Ses eşleştirme: {metrics['score']:.1f}/100 | {model}", flush=True)
            if metrics["score"] >= api.settings.alignment_threshold:
                return {
                    "words": words,
                    "metrics": metrics,
                    "duration": duration,
                    "model": model,
                }
            errors.append(f"{model}: {metrics['score']:.1f}/100")
        except Exception as error:
            errors.append(f"{model}: {type(error).__name__}: {str(error)[:350]}")
        save_json(directory / "alignment_errors.json", errors)
    raise SpeechFailure("Bu ses bölümü eşleşmedi: " + "; ".join(errors))


def select_voice(api, style, cache):
    if api.settings.voice != "auto":
        result = {"voice": api.settings.voice, "selection": "user"}
        save_json(api.directory / "voice_selection.json", result)
        return result
    if getattr(api, "tts_fallback", False):
        result = {"voice": "Ahmet", "selection": "edge_fallback"}
        save_json(api.directory / "voice_selection.json", result)
        return result
    key = hashlib.sha256(
        json.dumps(
            [
                api.settings.tts_model,
                api.settings.whisper_model,
                api.settings.alignment_threshold,
                style.get("narrator_delivery"),
                AUDITION_TEXT,
            ],
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    saved = cache / "voices" / key
    directory = api.directory / "voice_audition"
    if (saved / "selection.json").exists():
        result = json.loads((saved / "selection.json").read_text())
        if result.get("voice") in {"Orus", "Gacrux", "Fenrir"}:
            shutil.copytree(saved, directory, dirs_exist_ok=True)
            save_json(api.directory / "voice_selection.json", result)
            return result
    directory.mkdir(exist_ok=True)
    viable, errors, alignment = {}, {}, {}
    for name in ("Orus", "Gacrux", "Fenrir"):
        if getattr(api, "tts_fallback", False):
            break
        try:
            path = synthesize(
                api, AUDITION_TEXT, name, style, directory / (name + ".wav")
            )
            alignment[name] = align_clip(api, path, AUDITION_TEXT, directory / name)["metrics"]["score"]
            viable[name] = path
        except SpeechFailure as error:
            errors[name] = str(error)
    save_json(directory / "failed_auditions.json", errors)
    if getattr(api, "tts_fallback", False):
        result = {"voice": "Ahmet", "selection": "edge_fallback", "errors": errors}
        save_json(api.directory / "voice_selection.json", result)
        return result
    if not viable:
        raise SpeechFailure(
            "Hiçbir ses örneği eşleşme kontrolünü geçmedi; örnekler saklandı."
        )
    try:
        judged = api.json(
            "Türkçe ses karşılaştırması",
            f"""Listen to these SAME Turkish passages. Rank only supplied recordings from actual audio, never voice names. Assess natural Turkish pronunciation, proper names, engaging storytelling, clear articulation and target delivery: {style.get("narrator_delivery", "")}
Return {{"voices":[{{"voice":"","naturalness":0,"pronunciation":0,"energy":0,"reason":"Turkish audible evidence"}}]}}. Scores 0..100.""",
            audio=list(viable.items()),
            list_key="voices",
        )
    except FactoryError as error:
        # No model can listen right now: keep the voice Whisper understood best.
        api.note(f"Ses karşılaştırması yapılamadı ({str(error)[:120]}); eşleşme puanına göre seçildi.")
        best = max(viable, key=lambda name: alignment.get(name, 0))
        result = {
            "voice": best,
            "selection": "alignment_only",
            "alignment_scores": alignment,
            "tts_model": api.settings.tts_model,
        }
        save_json(directory / "selection.json", result)
        save_json(api.directory / "voice_selection.json", result)
        shutil.copytree(directory, saved, dirs_exist_ok=True)
        return result
    rankings = []
    for row in judged.get("voices", []):
        if not isinstance(row, dict) or row.get("voice") not in viable:
            continue
        try:
            scores = [float(row[k]) for k in ("naturalness", "pronunciation", "energy")]
            if any(not math.isfinite(s) or not 0 <= s <= 100 for s in scores):
                continue
        except (ValueError, TypeError, KeyError):
            continue
        rankings.append(
            {**row, "score": sum(s * w for s, w in zip(scores, (0.4, 0.4, 0.2)))}
        )
    rankings.sort(key=lambda r: r["score"], reverse=True)
    save_json(directory / "rankings.json", rankings)
    if not rankings:
        best = max(viable, key=lambda name: alignment.get(name, 0))
        rankings = [{"voice": best, "score": alignment.get(best, 0), "reason": "Jüri sıralaması alınamadı; eşleşme puanı."}]
    if rankings[0]["score"] < 85:
        # Every candidate already passed the Whisper intelligibility check; a
        # subjective "naturalness" score below target must not stop production.
        api.note(f"Ses jürisi düşük puan verdi ({rankings[0]['score']:.0f}); yine de en iyi ses ({rankings[0]['voice']}) kullanılıyor.")
    result = {
        "voice": rankings[0]["voice"],
        "rankings": rankings,
        "selection": "audio_audition",
        "subjective_assessment": True,
        "tts_model": api.settings.tts_model,
    }
    save_json(directory / "selection.json", result)
    save_json(api.directory / "voice_selection.json", result)
    shutil.copytree(directory, saved, dirs_exist_ok=True)
    return result


def chunks(shots, maximum):
    groups, current, count = [], [], 0
    for shot in shots:
        number = len(shot["narration"].split())
        if current and count + number > maximum:
            groups.append(current)
            current, count = [], 0
        current.append(shot)
        count += number
    if current:
        groups.append(current)
    return groups


def valid_cached(candidate, path, text, key, threshold):
    try:
        if (
            candidate.get("signature") != key
            or candidate.get("audio_sha256") != file_hash(path)
            or not threshold <= candidate["metrics"]["score"] <= 100
        ):
            return False
        duration = len(read_wave(path)) / RATE
        if (
            abs(duration - candidate["duration"]) > 1 / RATE
            or [w["word"] for w in candidate["words"]] != text.split()
        ):
            return False
        previous = 0
        for word in candidate["words"]:
            start, end = word["start"], word["end"]
            if not (
                math.isfinite(start)
                and math.isfinite(end)
                and previous <= start < end <= duration + 0.001
            ):
                return False
            previous = end
        return True
    except (ValueError, TypeError, KeyError, OSError, SpeechFailure):
        return False


def build_audio(api, story, voice, style, cache):
    samples, aligned, reports = [], [], []
    cursor = 0
    for i, group in enumerate(chunks(story["shots"], api.settings.chunk_words)):
        text = " ".join(s["narration"] for s in group)
        key = hashlib.sha256(
            json.dumps(
                [
                    text,
                    voice,
                    api.settings.tts_model,
                    api.settings.whisper_model,
                    style.get("narrator_delivery"),
                    3,
                ],
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        directory = api.directory / "audio" / f"chunk_{i:03}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "narration.txt").write_text(text, encoding="utf-8")
        cached = cache / "audio" / key
        accepted = None
        for saved in (directory, cached):
            if (
                not (saved / "accepted.json").exists()
                or not (saved / "narration.wav").exists()
            ):
                continue
            try:
                candidate = json.loads((saved / "accepted.json").read_text())
            except ValueError:
                continue
            if valid_cached(
                candidate,
                saved / "narration.wav",
                text,
                key,
                api.settings.alignment_threshold,
            ):
                if saved != directory:
                    shutil.copy2(saved / "narration.wav", directory / "narration.wav")
                accepted = candidate
                save_json(directory / "accepted.json", accepted)
                break
        failures = []
        if accepted is None:
            for attempt in range(api.settings.repair_attempts):
                api.check()
                work = directory / f"attempt_{attempt + 1}"
                work.mkdir(exist_ok=True)
                print(f"Ses bölümü {i + 1}, deneme {attempt + 1}", flush=True)
                try:
                    path = synthesize(api, text, voice, style, work / "narration.wav")
                    accepted = align_clip(api, path, text, work)
                    accepted.update(audio_sha256=file_hash(path), signature=key)
                    shutil.copy2(path, directory / "narration.wav")
                    save_json(directory / "accepted.json", accepted)
                    cached.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, cached / "narration.wav")
                    save_json(cached / "accepted.json", accepted)
                    break
                except SpeechFailure as error:
                    failures.append(str(error))
                    save_json(directory / "failures.json", failures)
        if accepted is None:
            raise SpeechFailure(
                f"{i + 1}. ses bölümü tamamlanamadı. Başarılı bölümler saklandı; aynı çalışmadan devam edilebilir."
            )
        data = read_wave(directory / "narration.wav")
        offset = cursor / RATE
        aligned.extend(
            {**w, "start": w["start"] + offset, "end": w["end"] + offset}
            for w in accepted["words"]
        )
        samples.append(data)
        cursor += len(data)
        reports.append(
            {
                "chunk": i,
                "offset": offset,
                "duration": len(data) / RATE,
                "score": accepted["metrics"]["score"],
                "voice": voice,
            }
        )
    if not samples or len(aligned) != len(story["narration"].split()):
        raise SpeechFailure("Ses ve senaryo kelime sayısı uyuşmuyor.")
    output = api.directory / "narration.wav"
    write_wave(output, np.concatenate(samples))
    duration = cursor / RATE
    save_json(
        api.directory / "aligned_words.json",
        {"words": aligned, "duration": duration, "chunks": reports},
    )
    if duration > 179:
        raise SpeechFailure(
            f"Anlatım {duration:.1f} saniye; daha kısa hedef süre seç. Başarılı sesler saklandı."
        )
    return output, aligned, duration
