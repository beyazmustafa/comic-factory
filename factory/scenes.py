"""Scene modules for simulation Shorts beyond orbiting dots.

Same chrome as factory.sim (title, phase line, stats row, caption, optional
field grid), different middle panels:

- ImageScene     a real NASA photograph fitted to the panel with a slow drift
- ScaleScene     true-scale discs or bars growing in, with labels
- TimelineScene  a horizontal timeline with a travelling marker
- LightScene     a photon crossing a distance while a timer counts
- CutawayScene   concentric layers of a world with labels

Each scene exposes draw(frame, renderer, t, phase_t) and stats(t). Scenes
draw into the virtual 1080x1920 space through renderer's ScaledDraw.
"""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import sim

PANEL = (90, 610, 990, 1100)   # default middle panel; renderer.panel overrides (tall layout when no grid)


def panel_of(renderer):
    return getattr(renderer, "panel", PANEL)
INK = (235, 236, 250)
GREY = (150, 152, 178)


def _ease(x):
    x = max(0.0, min(1.0, x))
    return 1 - (1 - x) ** 3


def _fmt(number):
    if isinstance(number, str):
        return number
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    if float(number).is_integer():
        return str(int(number))
    return f"{number:.1f}"


class Scene:
    uses_grid = False
    masses = ()

    def draw(self, frame, renderer, t, phase_t):
        raise NotImplementedError

    def stats(self, t):
        return {}

    def step(self, dt):
        pass


# -------------------------------------------------------------------- image
class ImageScene(Scene):
    """A photograph in the panel: fitted around its focus, drifting slowly."""

    def __init__(self, path, focus=(0.5, 0.5), credit="NASA", drift=0.06, full=False):
        with Image.open(path) as source:
            self.image = source.convert("RGB")
        self.focus = focus
        self.credit = credit
        self.drift = drift
        self.full = full
        self._cache = {}

    def _frame_image(self, t, panel, duration_hint=30.0):
        x0, y0, x1, y1 = panel
        w, h = (x1 - x0) * sim.SS, (y1 - y0) * sim.SS
        zoom = 1.0 + self.drift * _ease(t / duration_hint)
        iw, ih = self.image.size
        scale = max(w / iw, h / ih) * zoom
        sw, sh = max(w, round(iw * scale)), max(h, round(ih * scale))
        key = (sw, sh)
        if key not in self._cache:
            self._cache = {key: self.image.resize((sw, sh), Image.Resampling.LANCZOS)}
        big = self._cache[key]
        fx, fy = self.focus
        left = int(min(max(0, fx * sw - w / 2), sw - w))
        top = int(min(max(0, fy * sh - h / 2), sh - h))
        crop = big.crop((left, top, left + w, top + h))
        return crop, (x0, y0, x1, y1)

    def draw(self, frame, renderer, t, phase_t):
        crop, (x0, y0, x1, y1) = self._frame_image(t, panel_of(renderer))
        layer = Image.new("RGBA", crop.size, (0, 0, 0, 0))
        layer.paste(crop, (0, 0))
        # rounded corners + soft inner vignette so the photo sits in the design
        mask = Image.new("L", crop.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, crop.size[0] - 1, crop.size[1] - 1), radius=28 * sim.SS, fill=255)
        vign = Image.new("L", crop.size, 255)
        vd = ImageDraw.Draw(vign)
        for k in range(0, 36):
            a = int(255 * (1 - 0.35 * (1 - k / 36) ** 1.6))
            vd.rounded_rectangle((k * sim.SS, k * sim.SS, crop.size[0] - 1 - k * sim.SS, crop.size[1] - 1 - k * sim.SS), radius=28 * sim.SS, outline=a, width=sim.SS)
        mask = Image.fromarray((np.asarray(mask, dtype=np.float32) * np.asarray(vign, dtype=np.float32) / 255).astype(np.uint8))
        layer.putalpha(mask)
        frame.alpha_composite(layer, (x0 * sim.SS, y0 * sim.SS))
        d = sim.ScaledDraw(frame)
        if self.credit:
            tag = self.credit.split("/")[0].split(",")[0].strip()[:28] or "NASA"
            d.text((x1 - 14, y1 - 20), tag.upper(), font=sim.font("mono", 16), fill=(150, 152, 178, 170), anchor="rm")


# -------------------------------------------------------------------- scale
class ScaleScene(Scene):
    """Discs (or bars) at true relative size, growing in; labels beneath."""

    def __init__(self, items, unit="", mode="discs"):
        # items: [(label, value, color)] ; value relative
        self.items = items
        self.unit = unit
        self.mode = mode

    def draw(self, frame, renderer, t, phase_t):
        d = sim.ScaledDraw(frame)
        x0, y0, x1, y1 = panel_of(renderer)
        p = _ease(phase_t / 1.6)
        values = [v for _, v, _ in self.items]
        biggest = max(values) or 1
        if self.mode == "discs":
            # pack discs left to right; biggest disc radius fits the panel height
            max_r = min(300, (y1 - y0) * 0.36, (x1 - x0) * 0.42 / max(1, len(values)) * 1.6)
            radii = [max(3.0, max_r * (v / biggest) ** 0.5) if v / biggest > 0.0004 else 3.0 for v in values]
            total = sum(2 * r for r in radii) + 70 * (len(radii) - 1)
            x = 540 - total / 2
            cy = (y0 + y1) / 2 - 50
            for (label, value, color), r in zip(self.items, radii):
                rr = r * p
                cx = x + r
                glow = renderer.glow(color, int(min(60, rr) + 12))
                frame.alpha_composite(glow, (int((cx - (min(60, rr) + 12)) * sim.SS), int((cy - (min(60, rr) + 12)) * sim.SS)))
                d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=color + (255,))
                if rr <= 6:
                    d.ellipse((cx - 22, cy - 22, cx + 22, cy + 22), outline=(255, 255, 255, 170), width=1)
                d.text((cx, cy + max_r + 30), label.upper(), font=sim.font("bold", 22), fill=INK + (240,), anchor="mm")
                d.text((cx, cy + max_r + 62), f"{_fmt(value)} {self.unit}".strip(), font=sim.font("mono", 24), fill=color + (255,), anchor="mm")
                x += 2 * r + 70
        else:
            # horizontal bars
            n = len(self.items)
            gap = (y1 - y0 - 60) / n
            for k, (label, value, color) in enumerate(self.items):
                yy = y0 + 40 + k * gap
                length = (x1 - x0 - 300) * (value / biggest) * p
                d.text((x0 + 10, yy), label.upper(), font=sim.font("bold", 22), fill=INK + (240,), anchor="lm")
                d.rectangle((x0 + 250, yy - 10, x0 + 250 + max(4, length), yy + 10), fill=color + (230,))
                d.text((x0 + 262 + max(4, length), yy), f"{_fmt(value)} {self.unit}".strip(), font=sim.font("mono", 24), fill=color + (255,), anchor="lm")


# ----------------------------------------------------------------- timeline
class TimelineScene(Scene):
    """A line of events; the marker travels from the first to the last."""

    def __init__(self, events, unit="years", span_seconds=20.0, log=False):
        self.events = events  # [(value, label, color)]
        self.unit = unit
        self.span = span_seconds
        self.log = log

    def _pos(self, value, panel=PANEL):
        """Events are spaced evenly (a ten-million-year gap and a hundred-year
        gap both get room for their labels); values are shown as text."""
        values = [v for v, _, _ in self.events]
        if len(values) < 2:
            return (panel[0] + panel[2]) / 2
        # fractional index of value between its neighbours
        order = sorted(range(len(values)), key=lambda i: values[i])
        sorted_values = [values[i] for i in order]
        if value <= sorted_values[0]:
            f = 0.0
        elif value >= sorted_values[-1]:
            f = 1.0
        else:
            k = max(i for i in range(len(sorted_values)) if sorted_values[i] <= value)
            lo, hi = sorted_values[k], sorted_values[k + 1]
            f = (k + (value - lo) / max(1e-9, hi - lo)) / (len(sorted_values) - 1)
        return panel[0] + 80 + f * (panel[2] - panel[0] - 160)

    def _current(self, t):
        values = sorted(v for v, _, _ in self.events)
        progress = _ease(min(1.0, t / self.span))
        k = progress * (len(values) - 1)
        i = min(len(values) - 2, int(k))
        return values[i] + (values[i + 1] - values[i]) * (k - i)

    def draw(self, frame, renderer, t, phase_t):
        d = sim.ScaledDraw(frame)
        panel = panel_of(renderer)
        y = (panel[1] + panel[3]) / 2
        d.line([(panel[0] + 40, y), (panel[2] - 40, y)], fill=(90, 100, 170, 200), width=2)
        current = self._current(t)
        for k, (value, label, color) in enumerate(self.events):
            x = self._pos(value, panel)
            reached = value <= current + 1e-9
            d.ellipse((x - 9, y - 9, x + 9, y + 9), fill=color + (255,) if reached else (40, 44, 80, 255), outline=color + (255,), width=2)
            above = k % 2 == 0
            ty = y - 70 if above else y + 70
            d.text((x, ty), label, font=sim.font("regular", 26), fill=INK + (235 if reached else 120,), anchor="mm")
            d.text((x, ty + (-36 if above else 36)), _fmt(value), font=sim.font("mono", 22), fill=color + (220,), anchor="mm")
        mx = self._pos(current, panel)
        glow = renderer.glow((255, 255, 255), 30)
        frame.alpha_composite(glow, (int((mx - 30) * sim.SS), int((y - 30) * sim.SS)))
        d.ellipse((mx - 7, y - 7, mx + 7, y + 7), fill=(255, 255, 255, 255))

    def stats(self, t):
        return {"time": [(_fmt(round(self._current(t))), INK, "")]}


# -------------------------------------------------------------------- light
class LightScene(Scene):
    """A photon travelling between two bodies; the timer shows light delay."""

    def __init__(self, a, b, seconds, span_seconds=18.0):
        self.a, self.b = a, b          # (label, color, radius)
        self.seconds = seconds         # real light-travel seconds
        self.span = span_seconds

    def draw(self, frame, renderer, t, phase_t):
        d = sim.ScaledDraw(frame)
        panel = panel_of(renderer)
        y = (panel[1] + panel[3]) / 2
        xa, xb = panel[0] + 120, panel[2] - 120
        for (label, color, r), x in ((self.a, xa), (self.b, xb)):
            glow = renderer.glow(color, r + 30)
            frame.alpha_composite(glow, (int((x - r - 30) * sim.SS), int((y - r - 30) * sim.SS)))
            d.ellipse((x - r, y - r, x + r, y + r), fill=color + (255,))
            d.text((x, y + r + 40), label.upper(), font=sim.font("bold", 22), fill=INK + (240,), anchor="mm")
        f = min(1.0, t / self.span)
        px = xa + self.a[2] + (xb - xa - self.a[2] - self.b[2]) * f
        d.line([(xa + self.a[2], y), (px, y)], fill=(255, 240, 200, 150), width=2)
        glow = renderer.glow((255, 250, 220), 26)
        frame.alpha_composite(glow, (int((px - 26) * sim.SS), int((y - 26) * sim.SS)))
        d.ellipse((px - 5, y - 5, px + 5, y + 5), fill=(255, 255, 255, 255))

    def stats(self, t):
        f = min(1.0, t / self.span)
        elapsed = self.seconds * f
        if elapsed >= 3600:
            text, unit = f"{elapsed / 3600:.1f}", "h"
        elif elapsed >= 60:
            text, unit = f"{elapsed / 60:.1f}", "min"
        else:
            text, unit = f"{elapsed:.1f}", "s"
        return {"time": [(text, INK, unit)]}


# ------------------------------------------------------------------ cutaway
class CutawayScene(Scene):
    """Concentric layers of a world, revealed outside-in, with labels."""

    def __init__(self, layers):
        self.layers = layers  # [(label, fraction_of_radius, color)] outermost first

    def draw(self, frame, renderer, t, phase_t):
        d = sim.ScaledDraw(frame)
        panel = panel_of(renderer)
        cx, cy = panel[0] + (panel[2] - panel[0]) * 0.34, (panel[1] + panel[3]) / 2
        R = min(300, (panel[3] - panel[1]) * 0.36)
        p = _ease(phase_t / 1.4)
        for k, (label, frac, color) in enumerate(self.layers):
            r = R * frac * p
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color + (255,))
        # a wedge of darkness so the layers read as a cut
        wedge = Image.new("RGBA", (sim.WIDTH * sim.SS, sim.HEIGHT * sim.SS), (0, 0, 0, 0))
        ImageDraw.Draw(wedge).pieslice(((cx - R - 4) * sim.SS, (cy - R - 4) * sim.SS, (cx + R + 4) * sim.SS, (cy + R + 4) * sim.SS), -90, 0, fill=sim.BG + (255,))
        frame.alpha_composite(wedge)
        for k, (label, frac, color) in enumerate(self.layers):
            lx = panel[0] + (panel[2] - panel[0]) * 0.66
            ly = cy - 26 * (len(self.layers) - 1) + k * 52
            d.line([(cx + R * frac * 0.72 * p, cy - R * frac * 0.72 * p), (lx - 14, ly)], fill=color + (150,), width=1)
            d.ellipse((lx - 6, ly - 6, lx + 6, ly + 6), fill=color + (255,))
            d.text((lx + 14, ly), label, font=sim.font("regular", 26), fill=INK + (240,), anchor="lm")


# -------------------------------------------------------------------- orbit
class OrbitScene(Scene):
    """The original simulation (bodies, particles, trails) as a scene."""

    uses_grid = True

    def __init__(self, world, cfg):
        self.world, self.cfg = world, cfg
        self.renderer = None
        self._accumulator = 0.0

    @property
    def masses(self):
        return [b for b in self.world.bodies if b.alive]

    def step(self, dt):
        """One video frame: cfg['steps'] integration steps, trails recorded."""
        previous = [b.pos.copy() for b in self.world.bodies]
        self._accumulator += float(self.cfg.get("steps", 4))
        while self._accumulator >= 1.0:
            self.world.step(self.cfg["dt"])
            self._accumulator -= 1.0
        if self.renderer is not None:
            for body, prev in zip(self.world.bodies, previous):
                if body.alive and (body.mass > 0.00005 or body.name in self.cfg.get("speed_bodies", [])):
                    self.renderer.add_trail(body, prev, body.pos)

    def stats(self, t):
        return sim._stats(self.world, self.cfg, t)


# ------------------------------------------------------------------ helpers
def starfield(size=(1800, 1000), seed=3):
    """Fallback backdrop when no photograph could be fetched."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", size, (8, 9, 22))
    d = ImageDraw.Draw(img)
    for _ in range(900):
        x, y = rng.uniform(0, size[0]), rng.uniform(0, size[1])
        r = rng.choice([0.6, 0.9, 1.3, 2.0], p=[0.5, 0.3, 0.15, 0.05])
        c = int(rng.uniform(150, 255))
        d.ellipse((x - r, y - r, x + r, y + r), fill=(c, c, min(255, c + 20)))
    return img.filter(ImageFilter.GaussianBlur(0.4))
