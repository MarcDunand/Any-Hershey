"""
Preview window shown after converting and before saving.

Shows the final strokes (what the SVG will contain) with optional layers:
length colors, true pen width, pen-up path, stroke ends, the rasterized text
(the mask AnyHershey skeletonized), skeleton pixels, junctions and a mm grid.
A scrubber replays the plot in drawing order. "Export preview SVG" saves
exactly what is visible: on a transparent background, or on black with
"White on black" on.

Everything is drawn from one list of shapes in mm (build_ops), which either
Pillow (the on-screen image) or the SVG exporter renders, so the two match.
"""

import base64
import io
import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageTk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# =========================
# Look
# =========================

# Colored layers keep their colors on either background.
SHORT_COLOR = "#f07f1a"   # length colors: dots and the shortest strokes
LONG_COLOR = "#1f5fd1"    # length colors: the longest stroke
JUNCTION = "#d93a3a"

# Every gray. "White on black" swaps in DARK, where each gray is inverted
# (keeping its slight cool tint), so all of these flip together.
LIGHT = {
    "bg": "#ffffff",
    "ink": "#1b1e22",        # strokes, pen marker
    "pen_up": "#9aa0a6",     # pen-up path and its arrows
    "ends": "#3b4046",       # stroke start dots and end rings
    "mask": "#d6dadf",
    "skeleton": "#788088",
    "grid_minor": "#eceef1",
    "grid_major": "#d3d7dc",
}


def _invert_gray(h: str) -> str:
    rgb = [int(h[i:i + 2], 16) for i in (1, 3, 5)]
    m = sum(rgb) / 3
    return "#" + "".join(f"{min(255, max(0, round(255 - m + (c - m)))):02x}" for c in rgb)


DARK = {k: _invert_gray(v) for k, v in LIGHT.items()}

# On-screen sizes in display px. The SVG export converts them to mm at the
# current zoom, so it matches what is on screen.
LINE_PX = 2.0
DASH_PX, GAP_PX = 4.0, 3.0
ARROW_PX = 4.0
START_R_PX, END_R_PX = 2.6, 3.2
JUNCTION_R_PX = 4.5
HEAD_R_PX = 5.0

SUPERSAMPLE = 3          # still frames: drawn 3x larger, then scaled down
SUPERSAMPLE_MOVING = 1   # while the scrubber animates
PLAY_SECONDS = 6.0       # holding an arrow plays the whole plot in this long
FRAME_MS = 33


# =========================
# Data
# =========================

@dataclass
class PreviewData:
    strokes: List[np.ndarray]     # final strokes in drawing order, each (n, 2) in mm
    mask: np.ndarray              # bool, True = ink; pixel (r, c) is centered on (c, r) / px_per_mm
    skel: np.ndarray              # bool skeleton, same frame as mask
    px_per_mm: float
    size_mm: Tuple[float, float]  # page size of the saved SVG
    pen_mm: float = 0.3

    lengths: np.ndarray = field(init=False)
    ups: np.ndarray = field(init=False)
    total: float = field(init=False)
    ranks: np.ndarray = field(init=False)
    junctions: List[Tuple[float, float]] = field(init=False)

    def __post_init__(self):
        self.strokes = [np.asarray(s, dtype=float).reshape(-1, 2) for s in self.strokes if len(s)]
        self.strokes = [np.vstack([s, s]) if len(s) == 1 else s for s in self.strokes]
        self.lengths = np.array([_length(s) for s in self.strokes])
        self.ups = np.array([
            math.dist(a[-1], b[0]) for a, b in zip(self.strokes, self.strokes[1:])
        ])
        self.total = float(self.lengths.sum() + self.ups.sum())
        self.ranks = _rank01(self.lengths)
        self.junctions = _junctions(self.skel, self.px_per_mm)

    def stats_text(self) -> str:
        w, h = self.size_mm
        points = sum(len(s) for s in self.strokes)
        return (f"{len(self.strokes):,} strokes   {points:,} points   "
                f"drawing {self.lengths.sum():,.0f} mm   pen-up {self.ups.sum():,.0f} mm   "
                f"page {w:.1f} x {h:.1f} mm")

    def at(self, t: float):
        """What has been plotted after moving t mm along the plot (pen down and pen up).

        Returns (strokes, ups, head): strokes as (index, points, finished), pen-up
        moves as (start, end), and the pen position, or None once the plot is done.
        """
        strokes, ups = [], []
        left = t
        n = len(self.strokes)
        if t >= self.total - 1e-9:
            left = math.inf  # finished (sums of floats can land a hair short of the total)
        for i, pts in enumerate(self.strokes):
            if left < self.lengths[i]:
                part = _partial(pts, left)
                strokes.append((i, part, False))
                return strokes, ups, part[-1]
            strokes.append((i, pts, True))
            left -= self.lengths[i]
            if i + 1 < n:
                a, b = pts[-1], self.strokes[i + 1][0]
                if left < self.ups[i]:
                    f = left / self.ups[i] if self.ups[i] else 0.0
                    end = a + (b - a) * f
                    ups.append((a, end))
                    return strokes, ups, end
                ups.append((a, b))
                left -= self.ups[i]
        return strokes, ups, None


def _length(pts: np.ndarray) -> float:
    return float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 0.0


def _partial(pts: np.ndarray, d: float) -> np.ndarray:
    """The first d mm of a polyline."""
    seg = np.diff(pts, axis=0)
    sl = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(sl)])
    k = int(np.searchsorted(cum, d, side="right")) - 1
    if k >= len(sl):
        return pts
    f = (d - cum[k]) / sl[k] if sl[k] else 0.0
    return np.vstack([pts[:k + 1], pts[k] + f * seg[k]])


def _rank01(values: np.ndarray) -> np.ndarray:
    """Rank of each value scaled to 0..1 (ties share their average rank)."""
    n = len(values)
    if n < 2:
        return np.zeros(n)
    s = np.sort(values)
    lo = np.searchsorted(s, values, side="left")
    hi = np.searchsorted(s, values, side="right") - 1
    return (lo + hi) / 2 / (n - 1)


def _junctions(skel: np.ndarray, px_per_mm: float) -> List[Tuple[float, float]]:
    """Skeleton pixels with 3+ skeleton neighbors, clustered; one point (mm) per cluster."""
    if not skel.any():
        return []
    s = np.pad(skel.astype(np.uint8), 1)
    deg = sum(np.roll(np.roll(s, dy, 0), dx, 1)
              for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
    pix = set(map(tuple, np.argwhere((deg >= 3) & (s == 1)) - 1))
    out = []
    while pix:
        stack = [pix.pop()]
        group = []
        while stack:
            y, x = stack.pop()
            group.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (y + dy, x + dx)
                    if q in pix:
                        pix.remove(q)
                        stack.append(q)
        gy, gx = np.mean(group, axis=0)
        out.append((gx / px_per_mm, gy / px_per_mm))
    return out


# Length colors are mixed in OKLab, so the midpoint isn't a muddy gray-brown.
def _to_linear(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055) * 255


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def _hex_to_oklab(h):
    rgb = np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    return _M2 @ np.cbrt(_M1 @ _to_linear(rgb))


def _oklab_to_hex(lab):
    rgb = _to_srgb(np.linalg.inv(_M1) @ (np.linalg.inv(_M2) @ lab) ** 3)
    return "#" + "".join(f"{int(round(v)):02x}" for v in rgb)


_SHORT_LAB, _LONG_LAB = _hex_to_oklab(SHORT_COLOR), _hex_to_oklab(LONG_COLOR)


def length_color(r: float) -> str:
    return _oklab_to_hex(_SHORT_LAB + (_LONG_LAB - _SHORT_LAB) * r)


# =========================
# Shapes (all in mm)
# =========================

LAYERS = [
    # key, label, on by default
    ("grid", "mm grid", False),
    ("mask", "Rasterized text (mask)", False),
    ("skel", "Skeleton pixels", False),
    ("penup", "Pen-up path", False),
    ("strokes", "Strokes", True),
    ("colors", "Length colors (orange short, blue long)", False),
    ("penwidth", "True pen width", False),
    ("junctions", "Junctions", False),
    ("ends", "Stroke ends (dot = start, ring = end)", False),
]


def build_ops(d: PreviewData, on: dict, t: float, scale: float, pen_mm: float, pal: dict = LIGHT):
    """Shapes to draw, bottom to top, as (layer, kind, ...) tuples in mm.

    scale is display px per mm, used to turn the on-screen px sizes into mm.
    pal is LIGHT or DARK.
    """
    px = 1.0 / scale
    ops = []
    W, H = d.size_mm
    strokes, ups, head = d.at(t)

    if on["grid"]:
        # 1 mm lines, or 5 mm when 1 mm would be too dense to read; every 10 mm darker.
        step = 1 if scale >= 5 else 5
        for major in (False, True):
            color = pal["grid_major"] if major else pal["grid_minor"]
            for x in range(0, int(W) + 1, step):
                if (x % 10 == 0) == major:
                    ops.append(("grid", "line", np.array([[x, 0], [x, H]]), color, px, None))
            for y in range(0, int(H) + 1, step):
                if (y % 10 == 0) == major:
                    ops.append(("grid", "line", np.array([[0, y], [W, y]]), color, px, None))
    if on["mask"]:
        ops.append(("mask", "image", "mask"))
    if on["skel"]:
        ops.append(("skel", "image", "skel"))
    if on["penup"]:
        for a, b in ups:
            ops.append(("penup", "line", np.array([a, b]), pal["pen_up"], 1.2 * px, (DASH_PX * px, GAP_PX * px)))
            seg = b - a
            L = float(np.hypot(*seg))
            if L > 3 * ARROW_PX * px:
                u = seg / L
                nrm = np.array([-u[1], u[0]])
                m = a + seg / 2
                s = ARROW_PX * px / 2
                ops.append(("penup", "line", np.array([m - u * s + nrm * s, m + u * s, m - u * s - nrm * s]),
                            pal["pen_up"], 1.2 * px, None))
    if on["strokes"]:
        w = pen_mm if on["penwidth"] else LINE_PX * px
        for i, pts, _ in strokes:
            color = length_color(d.ranks[i]) if on["colors"] else pal["ink"]
            ops.append(("strokes", "line", pts, color, w, None))
    if on["junctions"]:
        for x, y in d.junctions:
            ops.append(("junctions", "dot", (x, y), JUNCTION_R_PX * px, None, JUNCTION, 1.5 * px))
    if on["ends"]:
        for _, pts, finished in strokes:
            ops.append(("ends", "dot", tuple(pts[0]), START_R_PX * px, pal["ends"], None, 0))
            if finished:
                ops.append(("ends", "dot", tuple(pts[-1]), END_R_PX * px, pal["bg"], pal["ends"], 1.2 * px))
    if head is not None:
        ops.append(("head", "dot", tuple(head), HEAD_R_PX * px, None, pal["ink"], 1.5 * px))
    return ops


def _dashes(a, b, dash, gap):
    seg = b - a
    L = float(np.hypot(*seg))
    if L == 0:
        return []
    u = seg / L
    out, s = [], 0.0
    while s < L:
        e = min(s + dash, L)
        out.append((a + u * s, a + u * e))
        s = e + gap
    return out


# =========================
# Rendering
# =========================

class Renderer:
    def __init__(self, data: PreviewData):
        self.d = data
        self._cache = {}

    def _rgba(self, key, pal):
        """The mask or skeleton as a transparent RGBA image, one pixel per mask pixel."""
        src = self.d.mask if key == "mask" else self.d.skel
        h = pal["mask" if key == "mask" else "skeleton"]
        a = np.zeros(src.shape + (4,), dtype=np.uint8)
        a[src] = [int(h[i:i + 2], 16) for i in (1, 3, 5)] + [255]
        return Image.fromarray(a, "RGBA")

    def _layer_image(self, key, size, pal):
        """The mask or skeleton scaled to `size`."""
        ck = (key, size, pal["bg"])
        if ck not in self._cache:
            img = self._rgba(key, pal)
            resample = Image.NEAREST if key == "skel" and size[0] > img.size[0] else Image.BOX
            self._cache[ck] = img.resize(size, resample)
        return self._cache[ck]

    def png(self, on, t, scale, pen_mm, pal=LIGHT, ss=SUPERSAMPLE) -> Image.Image:
        W, H = self.d.size_mm
        S = scale * ss
        size = (max(1, round(W * S)), max(1, round(H * S)))
        img = Image.new("RGBA", size, pal["bg"])
        draw = ImageDraw.Draw(img)

        def P(p):
            return (p[0] * S, p[1] * S)

        for op in build_ops(self.d, on, t, scale, pen_mm, pal):
            kind = op[1]
            if kind == "image":
                # Mask pixel (r, c) is centered on (c, r) / px_per_mm, so the image starts
                # half a pixel up and left of the origin; crop that half pixel off.
                h, w = self.d.mask.shape
                layer = self._layer_image(op[2], (max(1, round(w / self.d.px_per_mm * S)),
                                                  max(1, round(h / self.d.px_per_mm * S))), pal)
                off = round(0.5 / self.d.px_per_mm * S)
                if off:
                    layer = layer.crop((off, off, layer.size[0], layer.size[1]))
                img.alpha_composite(layer.crop((0, 0, min(layer.size[0], size[0]), min(layer.size[1], size[1]))))
                draw = ImageDraw.Draw(img)
            elif kind == "line":
                _, _, pts, color, w, dash = op
                wpx = max(1.0, w * S)
                if dash:
                    for a, b in _dashes(pts[0], pts[1], dash[0], dash[1]):
                        draw.line([P(a), P(b)], fill=color, width=round(wpx))
                    continue
                draw.line([P(p) for p in pts], fill=color, width=round(wpx), joint="curve")
                r = wpx / 2
                if op[0] in ("strokes", "penup") and r >= 1:
                    for p in (pts[0], pts[-1]):
                        x, y = P(p)
                        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
            elif kind == "dot":
                _, _, c, r, fill, stroke, sw = op
                x, y = P(c)
                R = r * S
                draw.ellipse([x - R, y - R, x + R, y + R], fill=fill, outline=stroke,
                             width=max(1, round(sw * S)) if stroke else 0)
        if ss != 1:
            img = img.resize((max(1, round(W * scale)), max(1, round(H * scale))), Image.LANCZOS)
        return img

    def svg(self, on, t, scale, pen_mm, pal=LIGHT) -> str:
        W, H = self.d.size_mm
        groups = {}
        if pal is DARK:
            # White lines need the black behind them; its own group, so it's easy to delete.
            groups["background"] = [f'<rect width="{W:.3f}" height="{H:.3f}" fill="{pal["bg"]}"/>']
        for op in build_ops(self.d, on, t, scale, pen_mm, pal):
            layer, kind = op[0], op[1]
            g = groups.setdefault(layer, [])
            if kind == "image":
                g.append(self._svg_image(op[2], pal))
            elif kind == "line":
                _, _, pts, color, w, dash = op
                d = "M" + " L".join(f"{x:.3f} {y:.3f}" for x, y in pts)
                extra = f' stroke-dasharray="{dash[0]:.3f} {dash[1]:.3f}"' if dash else ""
                cap = "round" if not dash else "butt"
                g.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w:.3f}" '
                         f'stroke-linecap="{cap}" stroke-linejoin="round"{extra}/>')
            elif kind == "dot":
                _, _, (x, y), r, fill, stroke, sw = op
                st = f' stroke="{stroke}" stroke-width="{sw:.3f}"' if stroke else ""
                g.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{r:.3f}" fill="{fill or "none"}"{st}/>')
        body = "\n".join(f'  <g id="{k}">\n    ' + "\n    ".join(v) + "\n  </g>" for k, v in groups.items())
        return (f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.3f}mm" height="{H:.3f}mm" '
                f'viewBox="0 0 {W:.3f} {H:.3f}">\n{body}\n</svg>\n')

    def _svg_image(self, key, pal):
        buf = io.BytesIO()
        self._rgba(key, pal).save(buf, "PNG", optimize=True)
        h, w = self.d.mask.shape
        half = 0.5 / self.d.px_per_mm
        style = ' style="image-rendering:pixelated"' if key == "skel" else ""
        return (f'<image x="{-half:.4f}" y="{-half:.4f}" width="{w / self.d.px_per_mm:.4f}" '
                f'height="{h / self.d.px_per_mm:.4f}" preserveAspectRatio="none"{style} '
                f'href="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"/>')


# =========================
# Window
# =========================

class PreviewWindow(tk.Toplevel):
    """Modal preview. After it closes, .result is "save" or "cancel"."""

    def __init__(self, master, data: PreviewData):
        super().__init__(master)
        self.title("AnyHershey preview")
        self.transient(master)
        self.result = "cancel"
        self.d = data
        self.r = Renderer(data)
        self.t = data.total
        self._imgtk = None
        self._pending = None
        self._playing = None

        self.on = {k: tk.BooleanVar(value=v) for k, _, v in LAYERS}
        self.dark = tk.BooleanVar(value=True)
        self.pen_mm = tk.StringVar(value=f"{data.pen_mm:g}")
        self.scrub = tk.DoubleVar(value=self.t)

        self._build()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{int(sw * 0.75)}x{int(sh * 0.7)}")
        self.minsize(700, 420)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.bind("<Escape>", lambda e: self._cancel())
        self.grab_set()
        self.focus_set()

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # A Canvas, not a Label: a Label grows to fit its image, which would fight
        # the fit-to-window scaling below.
        self.view = tk.Canvas(self, background=self._pal()["bg"], highlightthickness=0, width=200, height=120)
        self.view.grid(row=0, column=0, sticky="nsew", padx=(10, 0), pady=10)
        self.view.bind("<Configure>", lambda e: self._redraw_later())

        side = ttk.Frame(self, padding=(12, 10))
        side.grid(row=0, column=1, sticky="ns")
        ttk.Checkbutton(side, text="White on black", variable=self.dark,
                        command=self._on_dark).pack(anchor="w", pady=(0, 10))
        ttk.Label(side, text="Show:").pack(anchor="w", pady=(0, 4))
        for key, label, _ in LAYERS:
            indent = 18 if key in ("colors", "penwidth") else 0
            row = ttk.Frame(side)
            row.pack(anchor="w", padx=(indent, 0), pady=1)
            ttk.Checkbutton(row, text=label, variable=self.on[key], command=self._redraw_later).pack(side="left")
            if key == "penwidth":
                e = ttk.Entry(row, textvariable=self.pen_mm, width=5)
                e.pack(side="left", padx=(4, 2))
                ttk.Label(row, text="mm").pack(side="left")
                e.bind("<KeyRelease>", lambda ev: self._redraw_later())

        bottom = ttk.Frame(self, padding=(10, 0, 10, 10))
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew")
        bottom.columnconfigure(1, weight=1)

        back = ttk.Button(bottom, text="<", width=3)
        back.grid(row=0, column=0)
        ttk.Scale(bottom, from_=0.0, to=max(self.d.total, 1e-6), variable=self.scrub,
                  command=self._on_scrub).grid(row=0, column=1, sticky="ew", padx=6)
        fwd = ttk.Button(bottom, text=">", width=3)
        fwd.grid(row=0, column=2)
        for btn, direction in ((back, -1), (fwd, 1)):
            btn.bind("<ButtonPress-1>", lambda e, s=direction: self._play(s))
            btn.bind("<ButtonRelease-1>", lambda e: self._stop())

        ttk.Label(bottom, text=self.d.stats_text()).grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

        btns = ttk.Frame(bottom)
        btns.grid(row=2, column=0, columnspan=3, sticky="e", pady=(8, 0))
        ttk.Button(btns, text="Export preview SVG…", command=self._export).pack(side="left", padx=(0, 16))
        ttk.Button(btns, text="Cancel", command=self._cancel).pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Save", command=self._save).pack(side="left")

    # --- state ---

    def _layers(self):
        return {k: v.get() for k, v in self.on.items()}

    def _pal(self):
        return DARK if self.dark.get() else LIGHT

    def _on_dark(self):
        self.view.configure(background=self._pal()["bg"])
        self._redraw_later()

    def _pen(self):
        try:
            return max(0.01, float(self.pen_mm.get()))
        except ValueError:
            return self.d.pen_mm

    def _scale(self):
        W, H = self.d.size_mm
        vw, vh = max(50, self.view.winfo_width() - 16), max(50, self.view.winfo_height() - 16)
        return min(vw / W, vh / H)

    # --- drawing ---

    def _redraw_later(self):
        if self._pending is None:
            self._pending = self.after(15, self._redraw)

    def _redraw(self, ss=SUPERSAMPLE):
        self._pending = None
        img = self.r.png(self._layers(), self.t, self._scale(), self._pen(), self._pal(), ss=ss)
        self._imgtk = ImageTk.PhotoImage(img.convert("RGB"))
        self.view.delete("all")
        self.view.create_image(self.view.winfo_width() // 2, self.view.winfo_height() // 2,
                               image=self._imgtk, anchor="center")

    def _on_scrub(self, _value):
        self.t = float(self.scrub.get())
        if self._playing is None:
            self._redraw_later()

    # --- scrubber playback (press and hold an arrow) ---

    def _play(self, direction):
        self._stop()
        step = self.d.total / (PLAY_SECONDS * 1000 / FRAME_MS)

        def tick():
            self.t = min(self.d.total, max(0.0, self.t + direction * step))
            self.scrub.set(self.t)
            self._redraw(ss=SUPERSAMPLE_MOVING)
            if 0.0 < self.t < self.d.total:
                self._playing = self.after(FRAME_MS, tick)
            else:
                self._playing = None
                self._redraw()

        tick()

    def _stop(self):
        if self._playing is not None:
            self.after_cancel(self._playing)
            self._playing = None
            self._redraw()

    # --- buttons ---

    def _export(self):
        path = filedialog.asksaveasfilename(
            parent=self, title="Export preview SVG", defaultextension=".svg",
            filetypes=[("SVG files", "*.svg")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.r.svg(self._layers(), self.t, self._scale(), self._pen(), self._pal()))
        except Exception as e:
            messagebox.showerror("Export failed", str(e), parent=self)

    def _save(self):
        self.result = "save"
        self._close()

    def _cancel(self):
        self.result = "cancel"
        self._close()

    def _close(self):
        self._stop()
        self.grab_release()
        self.destroy()
