"""Cinematic vertical editor for real footage.

A plan lists shots (a file, an in/out point, a focus point, a zoom, a place
on the timeline) plus the hook card, per-shot labels, number callouts and the
end card. Each shot is decoded by ffmpeg straight into a 9:16 crop around its
focus point, graded (contrast, vignette, grain) on the way; Python then adds
the slow push-in, dissolves and the graphics, and streams the frames into a
near-lossless intermediate. Captions and sound are added by the caller.

Layout (1080x1920): a documentary HUD — mission label top-left, channel
handle top-right, thin progress line along the top edge; the hook card sits
in the upper middle for the first seconds; number callouts live on the left
below the HUD; captions (added later) sit at 70 % height, above the Shorts
UI; the end card dims the last shot.
"""

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1080, 1920, 30
ACCENT = (242, 194, 107)
WHITE = (255, 255, 255)
FONT_DIRS = [Path("/usr/share/fonts/opentype/inter"), Path("/usr/share/fonts/truetype/inter"),
             Path(__file__).resolve().parent.parent / "assets" / "fonts", Path("/usr/share/fonts/truetype/dejavu")]
FONT_FILES = {
    "display": ["InterDisplay-Bold.otf", "Inter-Bold.otf", "Inter-Bold.ttf", "DejaVuSans-Bold.ttf"],
    "bold": ["Inter-Bold.otf", "Inter-Bold.ttf", "DejaVuSans-Bold.ttf"],
    "semibold": ["Inter-SemiBold.otf", "Inter-SemiBold.ttf", "DejaVuSans-Bold.ttf"],
    "medium": ["Inter-Medium.otf", "Inter-Medium.ttf", "DejaVuSans.ttf"],
    "regular": ["Inter-Regular.otf", "Inter-Regular.ttf", "DejaVuSans.ttf"],
    "mono": ["DejaVuSansMono.ttf"],
}
_font_cache = {}


def font_path(kind):
    for name in FONT_FILES[kind]:
        for folder in FONT_DIRS:
            candidate = folder / name
            if candidate.is_file():
                return candidate
    return Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def font(kind, size):
    key = (kind, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(str(font_path(kind)), size)
    return _font_cache[key]


def caption_family():
    """Font family name for the ASS captions: Inter when installed, else DejaVu."""
    return "Inter" if "inter" in str(font_path("medium")).casefold() else "DejaVu Sans"


def ease_out(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = min(1.0, max(0.0, x))
    return 0.5 - 0.5 * math.cos(math.pi * x)


# ------------------------------------------------------------- text tools
def tracked(draw, xy, text, fnt, fill, tracking=2, anchor="l"):
    """Letter-spaced text (for small caps labels); anchor l or r for the x edge."""
    widths = [fnt.getlength(ch) for ch in text]
    total = sum(widths) + tracking * max(0, len(text) - 1)
    x, y = xy
    if anchor == "r":
        x -= total
    for ch, width in zip(text, widths):
        draw.text((x, y), ch, font=fnt, fill=fill)
        x += width + tracking
    return total


def wrap(text, fnt, max_width):
    words, lines, current = str(text).split(), [], ""
    for word in words:
        trial = (current + " " + word).strip()
        if fnt.getlength(trial) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def fit_lines(text, kind, size, max_width, max_lines, min_size=40):
    while size > min_size:
        lines = wrap(text, font(kind, size), max_width)
        if len(lines) <= max_lines and all(font(kind, size).getlength(l) <= max_width for l in lines):
            return lines, size
        size -= 4
    return wrap(text, font(kind, size), max_width)[:max_lines], size


def shadowed_text(layer, xy, text, fnt, fill, blur=6, alpha=170):
    """Text with a soft dark halo so it reads on bright footage."""
    x, y = xy
    halo = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    ImageDraw.Draw(halo).text((x, y), text, font=fnt, fill=(0, 0, 0, alpha))
    halo = halo.filter(ImageFilter.GaussianBlur(blur))
    layer.alpha_composite(halo)
    ImageDraw.Draw(layer).text((x, y), text, font=fnt, fill=fill)


# -------------------------------------------------------------- layers
class Layer:
    """An RGBA picture with its top-left position; composited with a scalar opacity."""

    def __init__(self, image, x, y):
        self.rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        self.alpha = np.asarray(image.getchannel("A"), dtype=np.float32)[..., None] / 255.0
        self.x, self.y = int(x), int(y)
        self.h, self.w = self.alpha.shape[:2]

    def paint(self, frame, opacity=1.0, dx=0, dy=0):
        if opacity <= 0.002:
            return
        x0, y0 = self.x + int(round(dx)), self.y + int(round(dy))
        x1, y1 = min(W, x0 + self.w), min(H, y0 + self.h)
        sx, sy = max(0, -x0), max(0, -y0)
        x0, y0 = max(0, x0), max(0, y0)
        if x1 <= x0 or y1 <= y0:
            return
        a = self.alpha[sy:sy + (y1 - y0), sx:sx + (x1 - x0)] * opacity
        region = frame[y0:y1, x0:x1].astype(np.float32)
        rgb = self.rgb[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
        frame[y0:y1, x0:x1] = (region * (1 - a) + rgb * a).astype(np.uint8)


def text_layer(text, kind, size, fill=WHITE, max_width=None, halo=True, tracking=0):
    fnt = font(kind, size)
    width = int(fnt.getlength(text) + tracking * max(0, len(text) - 1)) + 40
    height = int(size * 1.5) + 20
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if tracking:
        draw = ImageDraw.Draw(image)
        if halo:
            halo_img = Image.new("RGBA", image.size, (0, 0, 0, 0))
            tracked(ImageDraw.Draw(halo_img), (20, 10), text, fnt, (0, 0, 0, 150), tracking)
            image.alpha_composite(halo_img.filter(ImageFilter.GaussianBlur(5)))
        tracked(draw, (20, 10), text, fnt, fill + (255,), tracking)
    elif halo:
        shadowed_text(image, (20, 10), text, fnt, fill + (255,))
    else:
        ImageDraw.Draw(image).text((20, 10), text, font=fnt, fill=fill + (255,))
    return image


def gradient_scrim(top, bottom, peak=0.55, feather=0.35):
    """Per-row darkening factor: 1 outside [top, bottom], (1 - peak) in the middle, feathered edges."""
    rows = np.ones(H, dtype=np.float32)
    span = max(1, bottom - top)
    for y in range(max(0, top), min(H, bottom)):
        f = (y - top) / span
        edge = min(f, 1 - f) / feather
        rows[y] = 1 - peak * min(1.0, edge)
    return rows[:, None, None]


def hook_card(hook, kicker, handle):
    """The opening title: kicker, big hook (≤3 lines), accent rule. Returns the layers and the scrim."""
    lines, size = fit_lines(hook, "display", 88, 900, 3, min_size=56)
    layers = []
    y = 600
    layers.append(("kicker", Layer(text_layer(kicker.upper(), "semibold", 26, ACCENT, tracking=4), 64, y - 60)))
    for index, line in enumerate(lines):
        layers.append((f"line{index}", Layer(text_layer(line, "display", size), 64, y)))
        y += int(size * 1.12)
    rule = Image.new("RGBA", (140, 4), ACCENT + (255,))
    layers.append(("rule", Layer(rule, 84, y + 22)))
    scrim = gradient_scrim(440, y + 160, peak=0.62, feather=0.3)
    return layers, scrim


def end_card(line, handle, note):
    lines, size = fit_lines(line, "semibold", 60, 880, 3, min_size=44)
    layers, y = [], 760
    for index, text in enumerate(lines):
        image = text_layer(text, "semibold", size)
        layers.append(Layer(image, (W - image.width) // 2, y))
        y += int(size * 1.15)
    image = text_layer(handle, "medium", 38, ACCENT)
    layers.append(Layer(image, (W - image.width) // 2, y + 36))
    image = text_layer(note, "regular", 28, (220, 220, 230))
    layers.append(Layer(image, (W - image.width) // 2, y + 100))
    return layers


_number = re.compile(r"^(?P<prefix>[^\d]*?)(?P<num>-?\d[\d,]*(?:\.\d+)?)(?P<suffix>.*)$")


def callout_layers(callout):
    """Big number + small label; the number counts up when it is a single figure."""
    value, label = str(callout.get("value", "")), str(callout.get("label", ""))
    match = _number.match(value)
    frames = []
    if match and value.count(" ") <= 3 and len(re.findall(r"\d[\d,]*", value)) == 1:
        prefix, raw, suffix = match.group("prefix"), match.group("num"), match.group("suffix")
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        target = float(raw.replace(",", ""))
        grouped = "," in raw or abs(target) >= 10000
        for k in range(28):  # 28 frames ≈ 0.93 s of counting
            f = ease_out((k + 1) / 28)
            current = target * f
            text = f"{current:,.{decimals}f}" if grouped else f"{current:.{decimals}f}"
            frames.append(prefix + text + suffix)
        frames.append(value)
    else:
        frames = [value]
    value_layers = [Layer(text_layer(t, "bold", 112), 64, 980) for t in dict.fromkeys(frames)]
    sequence = [list(dict.fromkeys(frames)).index(t) for t in frames]
    label_layer = Layer(text_layer(label, "regular", 34, (235, 235, 240)), 72, 1120)
    return value_layers, sequence, label_layer


# ------------------------------------------------------------- decoding
def crop_window(width, height, focus):
    """9:16 window inside the source around the focus point (fractions)."""
    ch = min(height, int(width * 16 / 9))
    cw = int(ch * 9 / 16)
    cw -= cw % 2
    ch -= ch % 2
    fx, fy = focus
    cx = int(round(fx * width - cw / 2))
    cy = int(round(fy * height - ch / 2))
    cx = max(0, min(width - cw, cx))
    cy = max(0, min(height - ch, cy))
    return cw, ch, cx, cy


class ShotReader:
    """Frames of one shot, decoded by ffmpeg straight into the graded 9:16 crop
    with the slow push-in (zoompan) already applied."""

    def __init__(self, shot, frames_needed, grade=True):
        self.shot = shot
        self.frames_needed = frames_needed
        self.count = 0
        self.last = None
        self.buffer = []
        width, height = int(shot["width"]), int(shot["height"])
        focus = shot.get("focus", (0.5, 0.5))
        cw, ch, cx, cy = crop_window(width, height, focus)
        fx = (focus[0] * width - cx) / cw  # focus inside the crop, as fractions
        fy = (focus[1] * height - cy) / ch
        speed = float(shot.get("speed", 1.0))
        length = (shot["src_out"] - shot["src_in"]) / speed + 0.5
        filters = [f"crop={cw}:{ch}:{cx}:{cy}", f"scale={W}:{H}:flags=lanczos"]
        if abs(speed - 1.0) > 0.01:
            filters.append(f"setpts=PTS/{speed:.4f}")
        if grade:
            if H / max(1, ch) > 1.25:
                # an upscaled 1080p source gets a little acutance back
                filters.append("unsharp=5:5:0.45:3:3:0.0")
            filters += ["eq=contrast=1.07:saturation=1.08:gamma=0.98", "noise=alls=4:allf=t+u"]
        filters.append(f"fps={FPS}")
        z0, z1 = shot.get("zoom", (1.0, 1.06))
        if abs(z1 - z0) > 0.001 or abs(z0 - 1.0) > 0.001:
            n = max(1, frames_needed - 1)
            zoom = f"{z0:.4f}+({z1 - z0:.4f})*(0.5-0.5*cos(PI*min(1,on/{n})))"
            filters.append(f"zoompan=z='{zoom}':x='min(max(0,{fx:.4f}*iw-iw/zoom/2),iw-iw/zoom)':"
                           f"y='min(max(0,{fy:.4f}*ih-ih/zoom/2),ih-ih/zoom)':d=1:s={W}x{H}:fps={FPS}")
        self.process = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{shot['src_in']:.3f}", "-i", str(shot["file"]), "-t", f"{length:.3f}",
             "-vf", ",".join(filters), "-frames:v", str(frames_needed + 2), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=W * H * 3 * 4)

    def _raw(self):
        data = self.process.stdout.read(W * H * 3)
        if len(data) < W * H * 3:
            return None
        return np.frombuffer(data, dtype=np.uint8).reshape(H, W, 3)

    def _next(self):
        frame = self._raw()
        if frame is None:
            frame = self.last if self.last is not None else np.zeros((H, W, 3), dtype=np.uint8)
        self.last = frame
        self.count += 1
        return frame

    def read(self):
        """Next frame; the last frame is held if the source runs out."""
        if self.buffer:
            return self.buffer.pop(0)
        return self._next()

    def peek(self, n):
        """Read n frames ahead (for a dissolve) without consuming them."""
        while len(self.buffer) < n:
            self.buffer.append(self._next())
        return self.buffer[:n]

    def close(self):
        try:
            self.process.stdout.close()
            self.process.kill()
        except Exception:
            pass


# --------------------------------------------------------------- render
def render(plan, output, log=print):
    """Stream the edited, graded, labelled frames into a near-lossless intermediate (no sound)."""
    output = Path(output)
    duration = float(plan["duration"])
    total = int(round(duration * FPS))
    shots = sorted(plan["shots"], key=lambda s: s["t0"])
    handle = str(plan.get("handle", "")).strip()
    dissolve_frames = int(plan.get("dissolve_frames", 10))
    # graphics
    hook_layers, hook_scrim = hook_card(plan["hook"], plan.get("kicker", "Real footage"), handle)
    hook_until = float(plan.get("hook_seconds", 3.4))
    end_layers = end_card(plan.get("closing", ""), handle, plan.get("end_note", "Follow for more real footage.")) if plan.get("closing") else []
    end_from = duration - float(plan.get("end_seconds", 2.8))
    handle_layer = Layer(text_layer(handle, "medium", 26, (255, 255, 255), tracking=1), 0, 88)
    handle_layer.x = W - 56 - handle_layer.w
    label_layers = {}
    callouts = {}
    for shot in shots:
        if shot.get("label"):
            label_layers[shot["t0"]] = Layer(text_layer(str(shot["label"]).upper(), "semibold", 24, (255, 255, 255), tracking=3), 44, 90)
        if shot.get("callout"):
            callouts[shot["t0"]] = callout_layers(shot["callout"])
    encoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    readers = {}
    progress_track = np.full((4, W, 3), 255, dtype=np.uint8)

    def reader_for(index):
        if index not in readers:
            shot = shots[index]
            frames_needed = int(round((shot["t1"] - shot["t0"]) * FPS)) + dissolve_frames
            readers[index] = ShotReader(shot, frames_needed)
        return readers[index]

    current = 0
    try:
        for f in range(total):
            t = f / FPS
            while current + 1 < len(shots) and t >= shots[current + 1]["t0"] - 1e-6:
                if current in readers:
                    readers.pop(current).close()
                current += 1
            shot = shots[current]
            reader = reader_for(current)
            frame = reader.read().copy()
            # dissolve into the next shot
            if current + 1 < len(shots) and shots[current + 1].get("transition", "cut") == "dissolve":
                remaining = int(round((shots[current + 1]["t0"] - t) * FPS))
                if 0 < remaining <= dissolve_frames:
                    nxt = reader_for(current + 1).peek(dissolve_frames)
                    k = ease_in_out(1 - remaining / dissolve_frames)
                    incoming = nxt[dissolve_frames - remaining]
                    frame = (frame.astype(np.float32) * (1 - k) + incoming.astype(np.float32) * k).astype(np.uint8)
            # hook card
            if t < hook_until:
                fade_in = ease_out(t / 0.5)
                fade_out = 1.0 if t < hook_until - 0.4 else max(0.0, (hook_until - t) / 0.4)
                frame = np.clip(frame.astype(np.float32) * (1 - (1 - hook_scrim) * min(fade_in, fade_out)), 0, 255).astype(np.uint8)
                for index, (name, layer) in enumerate(hook_layers):
                    rise = ease_out((t - 0.08 * index) / 0.55)
                    layer.paint(frame, opacity=min(rise, fade_out), dy=(1 - rise) * 26)
            # HUD: label, handle, progress
            if t >= 0.9:
                hud = ease_out((t - 0.9) / 0.5)
                if t >= end_from:
                    hud *= max(0.0, 1 - (t - end_from) / 0.4)
                if shot.get("label") and shot["t0"] in label_layers:
                    local = t - shot["t0"]
                    fade = min(ease_out(local / 0.35), 1.0 if shot["t1"] - t > 0.3 else max(0.0, (shot["t1"] - t) / 0.3))
                    label_layers[shot["t0"]].paint(frame, opacity=0.92 * hud * fade, dx=(1 - min(1.0, local / 0.35)) * -10)
                if handle:
                    handle_layer.paint(frame, opacity=0.7 * hud)
                width = int(W * t / duration)
                if width > 0:
                    frame[0:4, :width] = (frame[0:4, :width].astype(np.float32) * 0.45 + progress_track[:, :width] * 0.55).astype(np.uint8)
            # callout
            if shot.get("callout") and shot["t0"] in callouts and t >= shot["t0"] + 0.25:
                values, sequence, label_layer = callouts[shot["t0"]]
                local = t - shot["t0"] - 0.25
                frame_index = min(len(sequence) - 1, int(local * FPS))
                show = min(ease_out(local / 0.3), max(0.0, min(1.0, (shot["t1"] - t) / 0.35)))
                values[sequence[frame_index]].paint(frame, opacity=show, dy=(1 - ease_out(local / 0.4)) * 18)
                label_layer.paint(frame, opacity=0.95 * show * ease_out((local - 0.2) / 0.4))
            # end card
            if end_layers and t >= end_from:
                k = ease_out((t - end_from) / 0.5)
                frame = (frame.astype(np.float32) * (1 - 0.55 * k)).astype(np.uint8)
                for index, layer in enumerate(end_layers):
                    rise = ease_out((t - end_from - 0.1 * index) / 0.5)
                    layer.paint(frame, opacity=rise, dy=(1 - rise) * 20)
            encoder.stdin.write(frame.tobytes())
            if f % (FPS * 10) == 0 and f:
                log(f"  kurgu {t:.0f}/{duration:.0f} sn")
    finally:
        for reader in readers.values():
            reader.close()
        encoder.stdin.close()
        error = encoder.stderr.read().decode("utf-8", "ignore")
        encoder.wait()
    if encoder.returncode != 0:
        raise RuntimeError("Kurgu kodlanamadı: " + error[-300:])
    return output


def still(frame_path, hook, handle, output, kicker="Real footage", focus=(0.5, 0.5)):
    """A cover: the chosen frame with the hook typography, for the thumbnail."""
    with Image.open(frame_path) as picture:
        picture = picture.convert("RGB")
        cw, ch, cx, cy = crop_window(picture.width, picture.height, focus)
        picture = picture.crop((cx, cy, cx + cw, cy + ch)).resize((W, H), Image.LANCZOS)
    frame = np.asarray(picture).copy()
    layers, scrim = hook_card(hook, kicker, handle)
    frame = np.clip(frame.astype(np.float32) * scrim, 0, 255).astype(np.uint8)
    for _, layer in layers:
        layer.paint(frame)
    if handle:
        layer = Layer(text_layer(handle, "medium", 30, (255, 255, 255), tracking=1), 0, 88)
        layer.x = W - 56 - layer.w
        layer.paint(frame, opacity=0.8)
    Image.fromarray(frame).save(output, "JPEG", quality=92)
    return output
