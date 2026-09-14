"""Opt-in real FFmpeg check; uses drawn fixtures and tone audio, not API speech."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from PIL import Image, ImageDraw
from factory import core, render, voice
from factory.config import Settings
from factory.style import load_style
from tests.test_studio import script_fixture, style, timestamps


@unittest.skipUnless(os.getenv("CF_RUN_MEDIA_TESTS") == "1", "Opt-in real FFmpeg test")
class RealMediaTests(unittest.TestCase):
    def test_transitions_music_subtitles_and_real_audio_offset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(os.getenv("CF_MEDIA_QA_OUTPUT", temporary))
            root.mkdir(parents=True, exist_ok=True)
            script, panels, _ = script_fixture()
            page = Image.new("RGB", (1200, 1800), "#eeeecc")
            draw = ImageDraw.Draw(page)
            colors = ["#b94742", "#385b96", "#32896d", "#99782f", "#613e88", "#be7947"]
            for i, color in enumerate(colors):
                x, y = (i % 2) * 600 + 20, (i // 2) * 600 + 20
                draw.rectangle(
                    (x, y, x + 560, y + 560), fill=color, outline="black", width=9
                )
                draw.ellipse(
                    (x + 160, y + 80, x + 380, y + 300),
                    fill="#ffe4b9",
                    outline="black",
                    width=7,
                )
                draw.rectangle((x + 170, y + 300, x + 370, y + 500), fill="#172c40")
                draw.text(
                    (x + 30, y + 30),
                    f"TEST PANEL {i + 1}",
                    fill="white",
                    font=render.font(style(), 34)[0],
                )
                path = root / panels[i]["file"]
                path.parent.mkdir(parents=True, exist_ok=True)
                page.crop((x, y, x + 560, y + 560)).save(path)
            page_path = root / panels[0]["page_file"]
            page_path.parent.mkdir(parents=True, exist_ok=True)
            page.save(page_path)
            script["shots"][2]["motion"] = "up"
            script["shots"][3]["motion"] = "down"
            script["shots"][1]["emphasis"] = "danger"
            words = timestamps(script["narration"], 0.15)
            duration = len(words) * 0.15
            t = np.arange(round(duration * voice.RATE)) / voice.RATE
            samples = np.zeros_like(t)
            for i, w in enumerate(words):
                samples += (
                    ((t >= w["start"]) & (t <= w["end"]))
                    * 0.15
                    * np.sin(2 * np.pi * (210 + 27 * i) * t)
                )
            audio = root / "narration.wav"
            voice.write_wave(audio, (samples * 32767).astype("<i2"))
            results = []
            for transition in ("cut", "slide", "whip"):
                api = SimpleNamespace(directory=root, settings=Settings(), check=Mock())
                profile = {
                    **load_style(),
                    "transition": transition,
                    "music_present": transition == "slide",
                }
                output, result = render.build(
                    api, script, panels, profile, audio, words, duration
                )
                self.assertTrue(result["passed"])
                stream = next(
                    s
                    for s in core.inspect_media(output)["streams"]
                    if s["codec_type"] == "video"
                )
                self.assertEqual(stream["r_frame_rate"], "30/1")
                self.assertEqual(int(stream["nb_frames"]), round(duration * 30))
                raw = subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-i",
                        str(output),
                        "-vn",
                        "-f",
                        "f32le",
                        "-ac",
                        "1",
                        "-ar",
                        str(voice.RATE),
                        "-",
                    ],
                    capture_output=True,
                    check=True,
                ).stdout
                decoded = np.frombuffer(raw, dtype="<f4")[: len(samples)]
                n = 1 << (len(samples) + len(decoded) - 1).bit_length()
                cross = np.fft.irfft(
                    np.fft.rfft(decoded, n) * np.conj(np.fft.rfft(samples, n)), n
                )
                position = int(np.argmax(cross))
                lag = position if position < n // 2 else position - n
                self.assertLessEqual(abs(lag) / voice.RATE, 0.025)
                results.append(
                    {
                        "transition": transition,
                        **result,
                        "sample_lag_ms": lag / voice.RATE * 1000,
                        "frames": stream["nb_frames"],
                    }
                )
                output.replace(root / (transition + ".mp4"))
            (root / "media-results.json").write_text(json.dumps(results, indent=2))
