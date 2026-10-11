"""Simulation Shorts: code-generated physics visualisations in the style of
the reference video — a dark canvas, one bold title, a line of explanation
that changes as the simulation moves through its phases, a row of live
numbers in monospace, glowing bodies leaving thin trails, and a perspective
"gravitational field" grid below them with a funnel under every mass.

Nothing is filmed or downloaded: every frame is computed. A scenario is a
small physics setup (bodies, particles, time scale) plus the texts; the
engine integrates it, renders at 30 fps, and loops cleanly at the end.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

WIDTH, HEIGHT, FPS = 1080, 1920, 30
BG = (9, 9, 20)
PALETTE = [(120, 220, 255), (255, 110, 170), (255, 220, 110), (170, 140, 255), (120, 255, 190), (255, 160, 90)]
GRID_COLOR = (92, 104, 178)

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def font(kind, size):
    name = {"bold": "DejaVuSans-Bold.ttf", "regular": "DejaVuSans.ttf", "light": "DejaVuSans-ExtraLight.ttf",
            "mono": "DejaVuSansMono.ttf", "mono_bold": "DejaVuSansMono-Bold.ttf"}[kind]
    path = FONT_DIR / name
    if not path.is_file():
        path = FONT_DIR / "DejaVuSans.ttf"
    return ImageFont.truetype(str(path), size)


# ------------------------------------------------------------------ physics
@dataclass
class Body:
    pos: np.ndarray
    vel: np.ndarray
    mass: float
    color: tuple
    radius: float = 7.0
    name: str = ""
    alive: bool = True


@dataclass
class World:
    bodies: list
    particles: np.ndarray | None = None      # (n, 4) x, y, vx, vy ; massless
    particle_color: tuple = (200, 200, 230)
    particle_alive: np.ndarray | None = None
    softening: float = 0.02
    G: float = 1.0
    drag: float = 0.0                          # for infalling discs
    sink_radius: float = 0.0                   # bodies/particles inside vanish (black hole)
    merge: bool = False                        # particles merge into bodies on contact
    time: float = 0.0
    events: dict = field(default_factory=dict)

    def accelerations(self, positions):
        acc = np.zeros_like(positions)
        for body in self.bodies:
            if not body.alive:
                continue
            d = body.pos - positions
            r2 = (d ** 2).sum(axis=1) + self.softening ** 2
            acc += (self.G * body.mass / (r2 * np.sqrt(r2)))[:, None] * d
        return acc

    def step(self, dt):
        alive = [b for b in self.bodies if b.alive]
        if alive:
            pos = np.array([b.pos for b in alive])
            vel = np.array([b.vel for b in alive])
            mass = np.array([b.mass for b in alive])

            def acc_of(p):
                a = np.zeros_like(p)
                for j in range(len(alive)):
                    d = p[j] - p
                    r2 = (d ** 2).sum(axis=1) + self.softening ** 2
                    r2[j] = np.inf
                    a += (self.G * mass[j] / (r2 * np.sqrt(r2)))[:, None] * d
                return a

            # velocity Verlet keeps orbits from spiralling numerically
            a0 = acc_of(pos)
            pos = pos + vel * dt + 0.5 * a0 * dt * dt
            a1 = acc_of(pos)
            vel = vel + 0.5 * (a0 + a1) * dt
            if self.drag:
                vel *= (1 - self.drag * dt)
            for k, body in enumerate(alive):
                body.pos, body.vel = pos[k], vel[k]
                if self.sink_radius and np.hypot(*body.pos) < self.sink_radius and body.mass < 50:
                    body.alive = False
        if self.particles is not None and len(self.particles):
            p = self.particles
            acc = self.accelerations(p[:, :2])
            p[:, 2:] += acc * dt
            if self.drag:
                p[:, 2:] *= (1 - self.drag * dt)
            p[:, :2] += p[:, 2:] * dt
            if self.sink_radius:
                r = np.hypot(p[:, 0], p[:, 1])
                self.particle_alive &= r > self.sink_radius
            if self.merge:
                for body in self.bodies:
                    if not body.alive:
                        continue
                    d = np.hypot(p[:, 0] - body.pos[0], p[:, 1] - body.pos[1])
                    hit = (d < body.radius * 0.012) & self.particle_alive
                    if hit.any():
                        gained = hit.sum()
                        body.mass += gained * 0.002
                        body.radius = min(18.0, body.radius + gained * 0.08)
                        self.particle_alive &= ~hit
        self.time += dt


# ----------------------------------------------------------------- renderer
class Renderer:
    def __init__(self, scenario):
        self.s = scenario
        self.center = np.array([540.0, scenario.get("sim_center_y", 860.0)])
        self.scale = scenario.get("scale", 120.0)           # pixels per world unit
        self.trail = np.zeros((HEIGHT, WIDTH, 4), dtype=np.float32)
        self.trail_fade = scenario.get("trail_fade", 0.992)
        self.glow_cache = {}
        self.horizon = 1185
        self.grid_cam_depth = 2.2
        self.grid_cam_height = 1.0
        self.grid_focal = 1500.0
        self.handle = scenario.get("handle", "")
        self.noise = self._noise()

    def _noise(self):
        rng = np.random.default_rng(1)
        n = rng.normal(0, 2.2, (HEIGHT, WIDTH, 1)).astype(np.float32)
        return n

    def world_to_screen(self, p):
        return self.center + np.array([p[0], -p[1]]) * self.scale

    def glow(self, color, size):
        key = (color, size)
        if key not in self.glow_cache:
            layer = Image.new("RGBA", (size * 2, size * 2), (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)
            for k in range(size, 0, -2):
                alpha = int(90 * (1 - k / size) ** 2.2)
                draw.ellipse((size - k, size - k, size + k, size + k), fill=color + (alpha,))
            self.glow_cache[key] = layer.filter(ImageFilter.GaussianBlur(size * 0.18))
        return self.glow_cache[key]

    # --- grid with wells
    def grid_project(self, x, y, z):
        depth = y + self.grid_cam_depth
        sx = 540 + self.grid_focal * x / depth
        sy = self.horizon + self.grid_focal * (self.grid_cam_height - z) / depth
        return sx, sy

    def well_depth(self, gx, gy, masses):
        z = np.zeros_like(gx)
        for (px, py, m, _) in masses:
            r = np.sqrt((gx - px) ** 2 + (gy - py) ** 2)
            depth = 0.95 * min(1.6, math.sqrt(max(m, 0.02)))
            z -= depth * np.exp(-((r / 0.33) ** 1.25))
        return z

    def draw_grid(self, draw, masses):
        # masses: list of (world_x, world_y, mass, color) mapped into grid plane coords
        xs = np.linspace(-7.0, 7.0, 43)
        ys = np.concatenate([np.linspace(0.15, 4.0, 22), np.geomspace(4.4, 22.0, 12)])
        gx, gy = np.meshgrid(xs, ys)
        z = self.well_depth(gx, gy, masses)
        sx, sy = self.grid_project(gx, gy, z)
        for i in range(len(ys)):
            alpha = int(50 + 90 * (1 - i / len(ys)))
            pts = list(zip(sx[i], sy[i]))
            draw.line(pts, fill=GRID_COLOR + (alpha,), width=1)
        for j in range(len(xs)):
            pts = list(zip(sx[:, j], sy[:, j]))
            draw.line(pts, fill=GRID_COLOR + (70,), width=1)
        # coloured rings around each well (the funnel look)
        for (px, py, m, color) in masses:
            for k, r in enumerate((0.05, 0.09, 0.14, 0.2, 0.27, 0.36, 0.47, 0.6, 0.76, 0.95, 1.2)):
                ang = np.linspace(0, 2 * math.pi, 56)
                cx, cy = px + r * np.cos(ang), py + r * np.sin(ang)
                cz = self.well_depth(cx, cy, masses)
                rx, ry = self.grid_project(cx, cy, cz)
                alpha = int(170 * (1 - k / 11) ** 1.3)
                draw.line(list(zip(rx, ry)) + [(rx[0], ry[0])], fill=color + (alpha,), width=1)
            # a few meridians down the funnel wall
            for ang in np.linspace(0, 2 * math.pi, 10, endpoint=False):
                rr = np.linspace(0.03, 1.2, 14)
                cx, cy = px + rr * np.cos(ang), py + rr * np.sin(ang)
                cz = self.well_depth(cx, cy, masses)
                rx, ry = self.grid_project(cx, cy, cz)
                draw.line(list(zip(rx, ry)), fill=color + (70,), width=1)

    def to_grid_plane(self, p):
        # world (x, y) → grid plane: x stays, y maps into depth 0.6..5.2
        return float(p[0] * 1.1), float(2.6 - p[1] * 0.8)

    # --- text
    def draw_text_block(self, draw, phase, alpha, stats_values):
        white = (240, 242, 255, int(255 * alpha))
        grey = (190, 192, 215, int(255 * alpha))
        if self.handle:
            draw.text((540, 165), self.handle, font=font("mono", 22), fill=(150, 152, 175, 200), anchor="mm")
        title = phase.get("title", "")
        lines = title.split("\n")
        size = 84 if len(lines) == 1 else 66
        while size > 40 and max(draw.textlength(line, font=font("bold", size)) for line in lines) > 940:
            size -= 4
        y = 380 if len(lines) == 1 else 350
        for line in lines:
            draw.text((540, y), line, font=font("bold", size), fill=white, anchor="mm")
            y += size * 1.15
        draw.text((540, 465 if len(lines) == 1 else y + 10), phase.get("subtitle", ""), font=font("light", 40), fill=grey, anchor="mm")
        # stats
        label_font, value_font, unit_font = font("bold", 21), font("mono", 34), font("regular", 22)
        stats = self.s.get("stats", [])
        for index, stat in enumerate(stats[:2]):
            x = 105 if index == 0 else 430
            draw.text((x if index == 0 else 605, 528), stat["label"], font=label_font, fill=(150, 152, 178, 255), anchor="lm")
            values = stats_values.get(stat["id"], [])
            if index == 0:
                for value, color, unit in values[:1]:
                    draw.text((x + 90, 567), f"{value}", font=value_font, fill=(235, 236, 250, 255), anchor="mm")
                    if unit:
                        draw.text((x + 90 + draw.textlength(f"{value}", font=value_font) / 2 + 10, 569), unit, font=unit_font, fill=(190, 192, 215, 230), anchor="lm")
                continue
            # right block: up to three coloured values spread across the remaining width;
            # a single value sits under its label like the time column
            cx = x if len(values) >= 3 else 605
            widths = [draw.textlength(f"{v}", font=value_font) + (draw.textlength(u, font=unit_font) + 8 if u else 0) + 26 for v, _, u in values[:3]]
            gap = max(24, min(60, (WIDTH - 60 - x - sum(widths)) / max(1, len(values[:3]) - 1))) if widths else 0
            for (value, color, unit), w in zip(values[:3], widths):
                draw.ellipse((cx - 3, 567 - 3, cx + 3, 567 + 3), fill=color + (255,))
                draw.text((cx + 16, 567), f"{value}", font=value_font, fill=color + (255,), anchor="lm")
                if unit:
                    draw.text((cx + 22 + draw.textlength(f"{value}", font=value_font), 569), unit, font=unit_font, fill=color + (230,), anchor="lm")
                cx += w + gap
        caption = self.s.get("caption", "")
        if caption:
            draw.text((105, 1130), caption, font=font("mono", 26), fill=(165, 168, 195, 255), anchor="lm")

    # --- frame
    def render_frame(self, world, phase, phase_alpha, stats_values, fade=1.0):
        frame = Image.new("RGBA", (WIDTH, HEIGHT), BG + (255,))
        # grid first (behind everything else in its band)
        band = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        gdraw = ImageDraw.Draw(band)
        masses = []
        for body in world.bodies:
            if body.alive:
                gx, gy = self.to_grid_plane(body.pos)
                masses.append((gx, gy, body.mass, body.color))
        self.draw_grid(gdraw, masses)
        # crop grid to its band with a soft top edge
        mask = Image.new("L", (WIDTH, HEIGHT), 0)
        ImageDraw.Draw(mask).rectangle((0, self.horizon - 10, WIDTH, HEIGHT), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(14))
        band.putalpha(Image.fromarray((np.asarray(band.getchannel("A"), dtype=np.float32) * np.asarray(mask, dtype=np.float32) / 255).astype(np.uint8)))
        # glow dots at the bottom of each funnel
        bdraw = ImageDraw.Draw(band)
        for (gx, gy, m, color) in masses:
            z = self.well_depth(np.array([gx]), np.array([gy]), masses)[0]
            sx, sy = self.grid_project(np.array([gx]), np.array([gy]), np.array([z]))
            bdraw.ellipse((sx[0] - 4, sy[0] - 4, sx[0] + 4, sy[0] + 4), fill=color + (255,))
        # a faint haze where the field meets the sky
        haze = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        hz = ImageDraw.Draw(haze)
        for k in range(0, 260, 4):
            a = int(38 * (1 - k / 260) ** 1.5)
            hz.line([(0, self.horizon + k), (WIDTH, self.horizon + k)], fill=(70, 72, 140, a), width=4)
        frame.alpha_composite(haze)
        frame.alpha_composite(band)
        # trails
        self.trail[:, :, 3] *= self.trail_fade
        trail_img = Image.fromarray(np.clip(self.trail, 0, 255).astype(np.uint8), "RGBA")
        frame.alpha_composite(trail_img)
        # particles
        if world.particles is not None and len(world.particles):
            pdraw = ImageDraw.Draw(frame)
            alive = world.particle_alive if world.particle_alive is not None else np.ones(len(world.particles), bool)
            for (x, y, _, _), ok in zip(world.particles, alive):
                if not ok:
                    continue
                sx, sy = self.world_to_screen((x, y))
                if 0 <= sx < WIDTH and 560 <= sy < 1110:
                    pdraw.ellipse((sx - 2.0, sy - 2.0, sx + 2.0, sy + 2.0), fill=world.particle_color + (235,))
        # black hole: a disc of nothing with a thin photon ring
        if self.s.get("horizon_radius"):
            hole = self.world_to_screen((0.0, 0.0))
            r = self.s["horizon_radius"] * self.scale
            hd = ImageDraw.Draw(frame)
            ring = self.glow((255, 200, 140), int(r * 1.6))
            frame.alpha_composite(ring, (int(hole[0] - r * 1.6), int(hole[1] - r * 1.6)))
            hd.ellipse((hole[0] - r, hole[1] - r, hole[0] + r, hole[1] + r), fill=(4, 4, 10, 255), outline=(255, 215, 160, 150), width=2)
        # bodies with glow
        for body in world.bodies:
            if body.name == "hole":
                continue
            if not body.alive:
                continue
            sx, sy = self.world_to_screen(body.pos)
            if not (-200 <= sx <= WIDTH + 200 and 560 <= sy <= 1110):
                continue
            size = int(28 + body.radius * 4)
            g = self.glow(body.color, size)
            frame.alpha_composite(g, (int(sx - size), int(sy - size)))
            d = ImageDraw.Draw(frame)
            r = body.radius
            d.ellipse((sx - r, sy - r, sx + r, sy + r), fill=tuple(min(255, c + 60) for c in body.color) + (255,))
        # texts
        tdraw = ImageDraw.Draw(frame)
        self.draw_text_block(tdraw, phase, phase_alpha, stats_values)
        out = np.asarray(frame.convert("RGB"), dtype=np.float32)
        out = np.clip(out + self.noise, 0, 255)
        if fade < 1.0:
            out *= fade
        return Image.fromarray(out.astype(np.uint8))

    def add_trail(self, body, previous, current):
        a, b = self.world_to_screen(previous), self.world_to_screen(current)
        layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        ImageDraw.Draw(layer).line([tuple(a), tuple(b)], fill=body.color + (170,), width=2)
        arr = np.asarray(layer, dtype=np.float32)
        mask = arr[:, :, 3] > 0
        self.trail[mask] = np.maximum(self.trail[mask], arr[mask])


# ----------------------------------------------------------------- scenarios
def three_body(texts):
    rng = np.random.default_rng(texts.get("seed", 4))
    bodies = []
    for k in range(3):
        ang = 2 * math.pi * k / 3 + rng.uniform(-0.3, 0.3)
        r = 1.0 + rng.uniform(-0.15, 0.15)
        pos = np.array([r * math.cos(ang), r * math.sin(ang)])
        speed = 0.62 + rng.uniform(-0.08, 0.08)
        vel = np.array([-math.sin(ang), math.cos(ang)]) * speed
        bodies.append(Body(pos, vel, 1.0, PALETTE[k], 7.0, f"star{k}"))
    world = World(bodies, softening=0.08)
    return world, {"dt": 0.004, "steps": 6, "scale": 150, "trail_fade": 0.993,
                   "stats": [{"id": "time", "label": "TIME ELAPSED · YEARS"}, {"id": "speeds", "label": "STAR SPEEDS"}],
                   "speed_unit": "km/s", "speed_scale": 24.0, "time_scale": 9.5, "caption": texts.get("caption", "Stars' gravitational fields"),
                   "escape_check": True}


def gravity_assist(texts):
    planet = Body(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 6.0, PALETTE[2], 13.0, "planet")
    sun_dir = np.array([0.0, -0.18])
    planet.vel = sun_dir * 1.0
    probe = Body(np.array([-3.6, -0.55]), np.array([1.9, 0.0]), 0.0001, PALETTE[0], 5.0, "probe")
    world = World([planet, probe], softening=0.03)
    return world, {"dt": 0.004, "steps": 5, "scale": 150, "trail_fade": 0.996,
                   "stats": [{"id": "time", "label": "TIME ELAPSED · DAYS"}, {"id": "speeds", "label": "PROBE SPEED"}],
                   "speed_unit": "km/s", "speed_scale": 7.5, "time_scale": 30.0, "caption": texts.get("caption", "Planet's gravitational field"),
                   "speed_bodies": ["probe"]}


def roche_limit(texts):
    planet = Body(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 8.0, PALETTE[0], 16.0, "planet")
    rng = np.random.default_rng(3)
    # moon as a cloud of self-bound-ish particles spiralling in (drag models tidal decay)
    n = 420
    ang = rng.uniform(0, 2 * math.pi, n)
    rad = 0.16 * np.sqrt(rng.uniform(0, 1, n))
    moon_pos = np.array([2.3, 0.0])
    px, py = moon_pos[0] + rad * np.cos(ang), moon_pos[1] + rad * np.sin(ang)
    v_orb = math.sqrt(8.0 / 2.3)
    particles = np.stack([px, py, np.zeros(n), np.full(n, v_orb)], axis=1)
    world = World([planet], particles=particles, particle_alive=np.ones(n, bool), particle_color=(230, 200, 170), softening=0.05, drag=0.09)
    return world, {"dt": 0.004, "steps": 5, "scale": 150, "trail_fade": 0.99,
                   "stats": [{"id": "time", "label": "TIME ELAPSED · YEARS"}, {"id": "count", "label": "MOON FRAGMENTS"}],
                   "time_scale": 1200.0, "caption": texts.get("caption", "Planet's gravitational field")}


def black_hole(texts):
    hole = Body(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 60.0, (40, 40, 60), 12.0, "hole")
    rng = np.random.default_rng(5)
    n = 700
    ang = rng.uniform(0, 2 * math.pi, n)
    rad = rng.uniform(0.9, 3.2, n)
    px, py = rad * np.cos(ang), rad * np.sin(ang)
    v = np.sqrt(60.0 / rad) * rng.uniform(0.92, 1.02, n)
    particles = np.stack([px, py, -v * np.sin(ang), v * np.cos(ang)], axis=1)
    world = World([hole], particles=particles, particle_alive=np.ones(n, bool), particle_color=(255, 205, 130), softening=0.05, drag=0.085, sink_radius=0.42)
    return world, {"dt": 0.002, "steps": 6, "scale": 150, "trail_fade": 0.98,
                   "stats": [{"id": "time", "label": "TIME ELAPSED · YEARS"}, {"id": "count", "label": "MATTER REMAINING"}],
                   "time_scale": 3.0, "caption": texts.get("caption", "Black hole's gravitational field"), "horizon_radius": 0.42}


def escape_velocity(texts):
    planet = Body(np.array([0.0, -1.1]), np.array([0.0, 0.0]), 5.0, PALETTE[0], 16.0, "planet")
    launches = []
    for k, (speed, color, name) in enumerate([(1.7, PALETTE[1], "slow"), (2.13, PALETTE[2], "orbit"), (3.05, PALETTE[3], "escape")]):
        launches.append(Body(np.array([0.0, -1.1 + 1.1]), np.array([speed, 0.0]), 0.00001, color, 5.0, name))
    world = World([planet] + launches, softening=0.03)
    return world, {"dt": 0.004, "steps": 5, "scale": 150, "trail_fade": 0.997,
                   "stats": [{"id": "time", "label": "TIME ELAPSED · HOURS"}, {"id": "speeds", "label": "LAUNCH SPEEDS"}],
                   "speed_unit": "km/s", "speed_scale": 3.7, "time_scale": 2.4, "caption": texts.get("caption", "Planet's gravitational field"),
                   "speed_bodies": ["slow", "orbit", "escape"]}


def binary_planet(texts):
    a, b = Body(np.array([-0.45, 0.0]), np.array([0.0, -0.62]), 1.2, PALETTE[2], 9.0, "A"), Body(np.array([0.55, 0.0]), np.array([0.0, 0.74]), 1.0, PALETTE[1], 8.0, "B")
    planet = Body(np.array([2.6, 0.0]), np.array([0.0, 0.93]), 0.0005, PALETTE[0], 5.0, "planet")
    world = World([a, b, planet], softening=0.05)
    return world, {"dt": 0.004, "steps": 6, "scale": 140, "trail_fade": 0.993,
                   "stats": [{"id": "time", "label": "TIME ELAPSED · YEARS"}, {"id": "speeds", "label": "ORBITAL SPEEDS"}],
                   "speed_unit": "km/s", "speed_scale": 30.0, "time_scale": 4.0, "caption": texts.get("caption", "Stars' gravitational fields")}


SCENARIOS = {
    "three-body": (three_body, "Three stars bound by gravity; chaotic orbits; close passes exchange energy until one star is flung out."),
    "gravity-assist": (gravity_assist, "A probe sweeps past a moving planet and leaves faster than it arrived: the planet's own motion hands it speed."),
    "roche-limit": (roche_limit, "A moon spirals toward its planet; inside the Roche limit tides pull it apart into a ring."),
    "black-hole": (black_hole, "Gas orbits a black hole, loses energy, spirals inward and vanishes past the horizon."),
    "escape-velocity": (escape_velocity, "Three launches from a planet at different speeds: one falls back, one circles forever, one leaves for good."),
    "binary-planet": (binary_planet, "A planet circles two stars at once; the stars swing around each other while the planet rides the combined pull."),
}


# ------------------------------------------------------------------- driver
def _phase_for(t, phases, duration):
    current = phases[0]
    for phase in phases:
        at = phase.get("at", 0.0)
        if isinstance(at, (int, float)) and t >= at * duration:
            current = phase
    return current


def render(scenario_key, texts, output, duration=28.0, note=print):
    """Render one simulation Short to output (mp4, silent, 30 fps)."""
    make, _ = SCENARIOS[scenario_key]
    world, cfg = make(texts)
    cfg["handle"] = texts.get("handle", "")
    phases = texts["phases"]
    renderer = Renderer(cfg)
    frames_dir = output.parent / "sim_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    total = int(duration * FPS)
    loop_tail = int(1.6 * FPS)
    live = total - loop_tail
    dt, steps = cfg["dt"], cfg["steps"]
    first_frames = []
    escaped_at = None
    for n in range(live):
        previous = [b.pos.copy() for b in world.bodies]
        for _ in range(steps):
            world.step(dt)
        for body, prev in zip(world.bodies, previous):
            if body.alive and body.mass > 0.00005 or body.name in cfg.get("speed_bodies", []):
                renderer.add_trail(body, prev, body.pos)
        # escape detection for the three-body phase switch
        if cfg.get("escape_check") and escaped_at is None:
            for body in world.bodies:
                others = [o for o in world.bodies if o is not body]
                d = min(np.hypot(*(body.pos - o.pos)) for o in others)
                v = np.hypot(*body.vel)
                if d > 2.4 and v > 0.5 and n > live * 0.35:
                    escaped_at = n
        t = n / FPS
        # phases: fixed fractions, except an "event" phase that waits for the escape
        active = None
        for phase in phases:
            at = phase.get("at", 0.0)
            if at == "event":
                if escaped_at is not None and n >= escaped_at:
                    active = phase
            elif t >= float(at) * duration:
                active = phase
        phase = active or phases[0]
        stats_values = _stats(world, cfg, t)
        fade = 1.0
        if n < int(0.5 * FPS):
            fade = n / (0.5 * FPS)
        frame = renderer.render_frame(world, phase, 1.0, stats_values, fade)
        frame.save(frames_dir / f"f_{n:05d}.jpg", "JPEG", quality=93)
        if n < loop_tail:
            first_frames.append(frames_dir / f"f_{n:05d}.jpg")
        if n % 120 == 0:
            note(f"Simülasyon karesi {n}/{total}")
    # loop tail: fade down, then the opening frames again so the Short replays seamlessly
    for k in range(loop_tail):
        src = Image.open(frames_dir / f"f_{live - 1:05d}.jpg") if k < loop_tail // 2 else Image.open(first_frames[min(len(first_frames) - 1, k - loop_tail // 2)])
        f = 1 - k / (loop_tail // 2) if k < loop_tail // 2 else (k - loop_tail // 2) / (loop_tail // 2)
        arr = (np.asarray(src, dtype=np.float32) * max(0.0, min(1.0, f))).astype(np.uint8)
        Image.fromarray(arr).save(frames_dir / f"f_{live + k:05d}.jpg", "JPEG", quality=93)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(FPS),
                    "-i", str(frames_dir / "f_%05d.jpg"), "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(output)], check=True, timeout=1800)
    for path in frames_dir.glob("f_*.jpg"):
        path.unlink()
    return output


def _stats(world, cfg, t):
    values = {}
    time_value = int(t * cfg.get("time_scale", 1.0))
    values["time"] = [(time_value, (235, 236, 250), "")]
    if any(s["id"] == "speeds" for s in cfg.get("stats", [])):
        rows = []
        wanted = cfg.get("speed_bodies")
        for body in world.bodies:
            if not body.alive or (wanted and body.name not in wanted) or (not wanted and body.mass < 0.00005):
                continue
            speed = float(np.hypot(*body.vel)) * cfg.get("speed_scale", 1.0)
            rows.append((int(round(speed)), body.color, cfg.get("speed_unit", "")))
        values["speeds"] = rows[:3]
    if any(s["id"] == "count" for s in cfg.get("stats", [])):
        alive = int(world.particle_alive.sum()) if world.particle_alive is not None else 0
        total = len(world.particle_alive) if world.particle_alive is not None else 1
        values["count"] = [(f"{alive}", (235, 236, 250), f"of {total}")]
    return values
