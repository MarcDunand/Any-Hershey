#!/usr/bin/env python3
"""
text_to_centerline_svg.py

Standalone project: type text -> generate a path-only SVG of single-line vectors.

Pipeline:
  1) Use Inkscape headless to convert TEXT to OUTLINE PATHS (font outlines).
  2) Sample outline paths into polylines (mm units).
  3) Convert outlines -> CENTERLINES using raster skeletonization:
       - rasterize filled outlines into a binary mask
       - skeletonize mask to 1-pixel-wide strokes
       - vectorize skeleton pixels back into polylines
  4) Export resulting polylines as SVG paths (stroke only, no fill).

Dependencies:
  pip install numpy pillow scikit-image svgpathtools

Requires:
  Inkscape installed (we call it headlessly).
"""

import os
import math
import time
import json
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional
from math import hypot


import numpy as np
from PIL import Image, ImageDraw, ImageChops, ImageTk
from skimage.morphology import skeletonize, binary_closing, square

from svgpathtools import svg2paths2

# UI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


#More error messages
import faulthandler
faulthandler.enable()


#Silence irrelevant warnings suggesting upgrades to skimage that break skeletonization
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*`square` is deprecated.*"
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*`binary_closing` is deprecated.*"
)




# =========================
# CONFIG
# =========================

# Windows default shown — change if needed
INKSCAPE_EXE = r"C:\Program Files\Inkscape\bin\inkscape.exe"

# Inkscape uses 96 px per inch, so:
PX_PER_MM = 96.0 / 25.4

# Filename containing preset values for text in a specific alphabet
PRESETS_FILENAME = "language_presets.json"

# Methods used to create the rasterized bitmask of characters
MASK_METHODS = ["Inkscape Raster", "XOR"]



# =========================
# Utilities
# =========================

def now():
    return time.strftime("%H:%M:%S")

def file_exists_or_raise(path: str, msg: str):
    if not os.path.exists(path):
        raise FileNotFoundError(msg)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))



# =========================
# Language presets (JSON)
# =========================

def load_language_presets() -> dict:
    """
    Load language presets from language_presets.json (same folder as this script).
    Returns a dict: { preset_name: {key: value, ...}, ... }
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, PRESETS_FILENAME)

    if not os.path.exists(path):
        # Safe fallback: one default option
        return {"Default": {}}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{PRESETS_FILENAME} must contain a JSON object at the top level.")
    return data


# =========================
# Text -> Outline Paths (Inkscape)
# =========================

def make_text_svg(text: str, font_family: str, font_size_mm: float) -> str:
    """
    Create a basic SVG that contains a <text> element.

    NOTE: this SVG still depends on fonts, so we immediately run it through
    Inkscape 'object-to-path' to bake it into geometric outlines.
    """
    font_px = font_size_mm * PX_PER_MM

    # Large canvas to avoid clipping.
    # Baseline at y = font_px.
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="1000mm" height="300mm"
     viewBox="0 0 {1000*PX_PER_MM:.2f} {300*PX_PER_MM:.2f}">
  <text x="0" y="{font_px:.2f}" font-family="{font_family}"
        font-size="{font_px:.2f}" xml:space="preserve">{text}</text>
</svg>'''

def inkscape_text_to_paths(text: str, font_family: str, font_size_mm: float) -> str:
    """
    Run Inkscape headless to convert text -> outline paths.
    Returns a filename to a temp SVG containing only paths.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        in_svg  = os.path.join(tmpdir, "in.svg")
        out_svg = os.path.join(tmpdir, "out.svg")

        with open(in_svg, "w", encoding="utf-8") as f:
            f.write(make_text_svg(text, font_family, font_size_mm))

        out_path = out_svg.replace("\\", "/")

        actions = (
            "select-all:all;"
            "object-to-path;"
            f"export-filename:{out_path};"
            "export-plain-svg;"
            "export-do;"
            "file-close"
        )

        proc = subprocess.run(
            [INKSCAPE_EXE, in_svg, "--actions", actions],
            capture_output=True,
            text=True
        )

        if proc.returncode != 0:
            raise RuntimeError(
                "Inkscape failed.\n\n"
                f"Return code: {proc.returncode}\n\n"
                f"STDOUT:\n{proc.stdout}\n\n"
                f"STDERR:\n{proc.stderr}\n"
            )

        if not os.path.exists(out_svg):
            raise RuntimeError(
                "Inkscape returned success but did not create out.svg.\n\n"
                f"Expected: {out_svg}\n\n"
                f"STDOUT:\n{proc.stdout}\n\n"
                f"STDERR:\n{proc.stderr}\n"
            )

        with open(out_svg, "r", encoding="utf-8") as f:
            data = f.read()

    final_svg = os.path.join(tempfile.gettempdir(), f"outline_{int(time.time()*1000)}.svg")
    with open(final_svg, "w", encoding="utf-8") as f:
        f.write(data)
    return final_svg



def rasterize_outline_svg_to_bw(outline_svg: str, px_per_mm: int) -> np.ndarray:
    """
    Use Inkscape to rasterize the outline SVG to a filled PNG (black on white),
    then threshold into a boolean mask (True = ink).
    """
    dpi = px_per_mm * 25.4  # 1 inch = 25.4 mm

    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = os.path.join(tmpdir, "mask.png")
        out_png_slash = out_png.replace("\\", "/")

        # Export just the drawing bbox, as a PNG
        actions = (
            f"export-filename:{out_png_slash};"
            "export-area-drawing;"
            "export-type:png;"
            f"export-dpi:{dpi};"
            "export-background:white;"
            "export-background-opacity:1.0;"
            "export-do;"
            "file-close"
        )

        proc = subprocess.run(
            [INKSCAPE_EXE, outline_svg, "--actions", actions],
            capture_output=True,
            text=True
        )

        if proc.returncode != 0 or not os.path.exists(out_png):
            raise RuntimeError(
                "Inkscape raster export failed.\n\n"
                f"Return code: {proc.returncode}\n\n"
                f"STDOUT:\n{proc.stdout}\n\n"
                f"STDERR:\n{proc.stderr}\n"
            )

        img = Image.open(out_png).convert("L")
        arr = np.asarray(img, dtype=np.uint8)

    # Black shapes on white background
    bw = arr < 128
    return bw




# =========================
# Outline paths -> polylines (sampling)
# =========================

@dataclass
class ShapedRun:
    text: str
    width_mm: float
    height_mm: float
    polylines_mm: List[List[Tuple[float, float]]]  # (x,y) in mm, origin at top-left of bbox

def load_and_sample(svg_file: str, sample_step_mm: float) -> ShapedRun:
    """
    Read outline SVG paths and sample them into polylines.

    Important: we *preserve* document order and subpath order so the result
    is stable and “glyph-ish” before centerline conversion.
    """
    paths, attrs, svg_attr = svg2paths2(svg_file)
    if not paths:
        return ShapedRun(text="", width_mm=0.0, height_mm=0.0, polylines_mm=[])

    # Compute bbox in px over all paths
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    for p in paths:
        bx = p.bbox()
        if bx is None:
            continue
        x0, x1, y0, y1 = bx  # (xmin, xmax, ymin, ymax)
        xmin = min(xmin, x0); xmax = max(xmax, x1)
        ymin = min(ymin, y0); ymax = max(ymax, y1)

    if not math.isfinite(xmin):
        return ShapedRun(text="", width_mm=0.0, height_mm=0.0, polylines_mm=[])

    width_mm  = (xmax - xmin) / PX_PER_MM
    height_mm = (ymax - ymin) / PX_PER_MM

    polylines: List[List[Tuple[float, float]]] = []
    step_px = sample_step_mm * PX_PER_MM

    for p in paths:
        try:
            subs = p.continuous_subpaths()   # svgpathtools >= 1.6
        except AttributeError:
            subs = p.as_subpaths()           # fallback

        for sp in subs:
            length_px = sp.length()
            if length_px <= 0:
                continue

            n = max(2, int(math.ceil(length_px / step_px)))
            pts = []
            for i in range(n):
                t = i / (n - 1)
                z = sp.point(t)  # complex
                x_px, y_px = z.real, z.imag

                # Normalize bbox top-left to (0,0) in mm
                pts.append(((x_px - xmin) / PX_PER_MM, (y_px - ymin) / PX_PER_MM))

            polylines.append(pts)

    return ShapedRun(text="", width_mm=width_mm, height_mm=height_mm, polylines_mm=polylines)


# =========================
# Centerline algorithm (raster skeletonization)
# =========================

def is_closed(seg: List[Tuple[float, float]], close_mm: float) -> bool:
    """
    Heuristic: treat a polyline as a closed loop if endpoints are within close_mm.

    Why needed:
      Outline glyphs come as closed contours (outer boundary + holes).
      We fill closed contours to get solid ink regions before skeletonizing.
    """
    if len(seg) < 3:
        return False
    (x0, y0) = seg[0]
    (x1, y1) = seg[-1]
    return hypot(x0 - x1, y0 - y1) <= close_mm


def _vectorize_skeleton(skel_bool: np.ndarray,
                        px_per_mm: float,
                        min_branch_mm: float) -> List[List[Tuple[float, float]]]:
    """
    Convert a 1-pixel-wide skeleton image into polylines.

    INPUT:
      skel_bool: boolean array where True pixels form the skeleton lines.

    OUTPUT:
      list of polylines, each a list of (x_mm, y_mm).

    Key idea:
      Skeleton pixels form a graph:
        - nodes: pixels with degree != 2 (endpoints, junctions)
        - edges: chains of pixels between nodes
      We "walk" from nodes along edges to get maximal stroke paths.

    Steps:
      1) Build a set of all skeleton pixels and compute each pixel's neighbor degree.
      2) Seed walks from all pixels with deg != 2 (end/junction) to extract edge-like strokes.
      3) Extract remaining cycles (loops) by starting from any unvisited pixel.
      4) Convert (y,x) pixels -> (x,y) millimeters.
      5) Prune tiny fragments shorter than min_branch_mm.
    """
    pix = set(map(tuple, np.argwhere(skel_bool)))
    if not pix:
        return []

    # 8-connected neighborhood (smoother connectivity for diagonals)
    NBR = [(-1, -1), (-1, 0), (-1, 1),
           ( 0, -1),          ( 0, 1),
           ( 1, -1), ( 1, 0), ( 1, 1)]

    def neighbors(p):
        y, x = p
        for dy, dx in NBR:
            q = (y + dy, x + dx)
            if q in pix:
                yield q

    # Degree = how many skeleton pixels touch this pixel
    deg = {p: sum(1 for _ in neighbors(p)) for p in pix}

    visited = set()
    polylines_px = []

    def grow_from(start):
        """
        Walk forward along the skeleton until:
          - we hit a node/end (no unvisited continuation), or
          - we would revisit pixels.

        When at a junction (2+ choices), pick the neighbor that best continues the
        current direction to reduce zig-zagging.
        """
        path = [start]
        cur = start
        prev = None

        while True:
            nbrs = [q for q in neighbors(cur) if q != prev]
            nxt = None

            if len(nbrs) == 1:
                nxt = nbrs[0]

            elif len(nbrs) >= 2:
                if prev is None:
                    nxt = nbrs[0]
                else:
                    # Direction continuity heuristic using cosine similarity
                    vy, vx = cur[0] - prev[0], cur[1] - prev[1]
                    best_c, best_q = -2.0, None
                    for q in nbrs:
                        wy, wx = q[0] - cur[0], q[1] - cur[1]
                        dot = vx * wx + vy * wy
                        nv = (vx * vx + vy * vy) ** 0.5 or 1.0
                        nw = (wx * wx + wy * wy) ** 0.5 or 1.0
                        c = dot / (nv * nw)
                        if c > best_c:
                            best_c, best_q = c, q
                    nxt = best_q

            if nxt is None or nxt in visited:
                break

            prev, cur = cur, nxt
            path.append(cur)
            visited.add(cur)

        return path

    # (A) First extract paths starting at nodes/ends (degree != 2)
    seeds = [p for p, d in deg.items() if d != 2]
    for s in seeds:
        if s in pix and s not in visited:
            visited.add(s)
            poly = grow_from(s)
            if len(poly) >= 2:
                polylines_px.append(poly)

    # (B) Then extract leftover loops (all degree==2 cycles)
    remaining = [p for p in pix if p not in visited]
    for s in remaining:
        if s not in visited:
            visited.add(s)
            poly = grow_from(s)
            if len(poly) >= 2:
                polylines_px.append(poly)

    # Helper: length of a polyline in mm
    def seg_length_mm(pts_mm):
        L = 0.0
        for i in range(1, len(pts_mm)):
            x0, y0 = pts_mm[i - 1]
            x1, y1 = pts_mm[i]
            L += hypot(x1 - x0, y1 - y0)
        return L

    # Convert pixels -> mm and prune tiny stubs
    out = []
    for poly in polylines_px:
        pts_mm = [(x / px_per_mm, y / px_per_mm) for (y, x) in poly]
        if seg_length_mm(pts_mm) >= min_branch_mm:
            out.append(pts_mm)

    return out


def centerlines_from_outlines(
    polylines_mm: List[List[Tuple[float, float]]],
    px_per_mm: int = 38,
    close_tol_mm: float = 0.10,
    min_branch_mm: float = 0.28,
    do_close_gaps: bool = False,
    return_bw: bool = False,
):
    """
    OUTLINES (filled shapes) -> CENTERLINES (single stroke polylines)

    This is the "interesting algorithm" in one function.

    1) Rasterize outline polylines into a filled binary mask image.
       - Closed loops are filled as polygons.
       - We XOR fills so holes knock out correctly (outer vs inner contours).
    2) Optionally close tiny gaps in the mask (morphological closing).
    3) Skeletonize the filled mask -> 1-pixel-wide skeleton.
    4) Vectorize skeleton pixels back into polylines.

    NOTE: This will not produce typographically perfect "true medial axes"
          in all cases, but it's very effective and robust for plotter-friendly
          “single stroke” text.
    """
    if not polylines_mm:
        return []

    maxx = max(x for seg in polylines_mm for (x, _) in seg)
    maxy = max(y for seg in polylines_mm for (_, y) in seg)

    W = max(2, int(np.ceil(maxx * px_per_mm)))
    H = max(2, int(np.ceil(maxy * px_per_mm)))

    # 1-bit mask (mode "1")
    mask = Image.new("1", (W, H), 0)
    
    # (A) Fill closed polygons using XOR, to respect holes
    for seg in polylines_mm:
        if len(seg) < 2:
            continue
        pts_px = [(int(round(x * px_per_mm)), int(round(y * px_per_mm))) for (x, y) in seg]
        if is_closed(seg, close_tol_mm):
            poly = Image.new("1", (W, H), 0)
            ImageDraw.Draw(poly).polygon(pts_px, outline=1, fill=1)
            mask = ImageChops.logical_xor(mask, poly)

    # (B) Draw open polylines (rare in glyph outlines, but safe)
    draw = ImageDraw.Draw(mask)
    for seg in polylines_mm:
        if len(seg) < 2:
            continue
        if not is_closed(seg, close_tol_mm):
            pts_px = [(int(round(x * px_per_mm)), int(round(y * px_per_mm))) for (x, y) in seg]
            draw.line(pts_px, fill=1, width=1)

    bw = np.array(mask, dtype=bool)

    if do_close_gaps:
        bw = binary_closing(bw, square(3))

    # Skeletonize to 1px width
    skel = skeletonize(bw, method='lee')

    # Vectorize skeleton pixels -> polylines in mm
    centerlines = _vectorize_skeleton(
        skel, px_per_mm=float(px_per_mm), min_branch_mm=float(min_branch_mm)
    )

    if return_bw:
        return bw, centerlines
    return centerlines




# =========================
# Export polylines -> SVG
# =========================

def polylines_to_svg(polylines_mm: List[List[Tuple[float, float]]],
                     out_svg_path: str,
                     margin_mm: float = 2.0,
                     stroke_width_mm: float = 0.3) -> None:
    """
    Write a minimal SVG containing one <path> per polyline.

    We generate:
      M x y L x y L ...
    All in mm coordinates.
    """
    if not polylines_mm:
        # still write an SVG
        W = H = 10.0
    else:
        maxx = max(x for seg in polylines_mm for (x, _) in seg) + margin_mm
        maxy = max(y for seg in polylines_mm for (_, y) in seg) + margin_mm
        W = max(10.0, maxx)
        H = max(10.0, maxy)

    def path_d(seg):
        if not seg:
            return ""
        cmds = [f"M {seg[0][0]:.3f} {seg[0][1]:.3f}"]
        for (x, y) in seg[1:]:
            cmds.append(f"L {x:.3f} {y:.3f}")
        return " ".join(cmds)

    paths_xml = []
    for seg in polylines_mm:
        if len(seg) < 2:
            continue
        d = path_d(seg)
        paths_xml.append(
            f'<path d="{d}" fill="none" stroke="black" stroke-width="{stroke_width_mm:.3f}" '
            f'stroke-linecap="round" stroke-linejoin="round" />'
        )

    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{W:.3f}mm" height="{H:.3f}mm"
     viewBox="0 0 {W:.3f} {H:.3f}">
  {"".join(paths_xml)}
</svg>
'''
    with open(out_svg_path, "w", encoding="utf-8") as f:
        f.write(svg)


# =========================
# High-level "convert text -> centerline SVG"
# =========================

def text_to_centerline_polylines(
    text: str,
    font_family: str,
    font_size_mm: float,
    sample_step_mm: float,
    skel_px_per_mm: int,
    skel_close_mm: float,
    skel_min_branch_mm: float,
    skel_close_gaps: bool,
    mask_method: str,
) -> Tuple[np.ndarray, List[List[Tuple[float, float]]]]:
    """
    Convenience wrapper:
      text -> outline svg -> sampled outline polylines -> centerline polylines
    """
    outline_svg = inkscape_text_to_paths(text, font_family, font_size_mm)
    try:
        if mask_method == "Inkscape Raster":
            # Inkscape itself rasterizes the filled outlines (handles overlaps + holes correctly)
            bw = rasterize_outline_svg_to_bw(outline_svg, px_per_mm=skel_px_per_mm)

            if skel_close_gaps:
                bw = binary_closing(bw, square(3))

            skel = skeletonize(bw, method="lee")
            centerlines = _vectorize_skeleton(
                skel, px_per_mm=float(skel_px_per_mm), min_branch_mm=float(skel_min_branch_mm)
            )
            return bw, centerlines

        else:
            # Current approach: sample paths -> XOR-filled mask inside centerlines_from_outlines()
            shaped = load_and_sample(outline_svg, sample_step_mm)

            bw, centerlines = centerlines_from_outlines(
                shaped.polylines_mm,
                px_per_mm=skel_px_per_mm,
                close_tol_mm=skel_close_mm,
                min_branch_mm=skel_min_branch_mm,
                do_close_gaps=skel_close_gaps,
                return_bw=True,
            )
            return bw, centerlines

    finally:
        try:
            os.remove(outline_svg)
        except Exception:
            pass




class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # Load presets from external JSON
        try:
            self.language_presets = load_language_presets()
        except Exception as e:
            # Don't crash the whole app; fall back and show why
            self.language_presets = {"Default": {}}
            messagebox.showwarning("Preset load failed", f"Could not load {PRESETS_FILENAME}:\n\n{e}")

        def _tk_exception_handler(exc, val, tb):
            import traceback
            err = "".join(traceback.format_exception(exc, val, tb))
            print(err)  # shows in terminal if launched from terminal
            messagebox.showerror("Unhandled exception", err)

        self.report_callback_exception = _tk_exception_handler

        self.title("Text → Centerline SVG (Inkscape + Skeletonize)")


        self.language_preset = tk.StringVar(value="Default")
        self.mask_method = tk.StringVar(value="XOR")
        self.font_family = tk.StringVar(value="Noto Sans CJK SC")
        self.font_size_mm = tk.DoubleVar(value=8.0)
        self.sample_step_mm = tk.DoubleVar(value=1.0)

        self.skel_px_per_mm = tk.IntVar(value=38)
        self.skel_close_mm = tk.DoubleVar(value=0.10)
        self.skel_min_branch_mm = tk.DoubleVar(value=0.28)
        self.skel_close_gaps = tk.BooleanVar(value=True)

        self._build()

    def apply_language_preset(self, preset_name: str):
        preset = self.language_presets.get(preset_name, {})
        if not isinstance(preset, dict) or not preset:
            return

        # Only overwrite keys that exist in the preset dict
        if "mask_method" in preset:
            self.mask_method.set(str(preset["mask_method"]))
        if "font_family" in preset:
            self.font_family.set(str(preset["font_family"]))
        if "font_size_mm" in preset:
            self.font_size_mm.set(float(preset["font_size_mm"]))
        if "sample_step_mm" in preset:
            self.sample_step_mm.set(float(preset["sample_step_mm"]))

        if "skel_px_per_mm" in preset:
            self.skel_px_per_mm.set(int(preset["skel_px_per_mm"]))
        if "skel_close_mm" in preset:
            self.skel_close_mm.set(float(preset["skel_close_mm"]))
        if "skel_min_branch_mm" in preset:
            self.skel_min_branch_mm.set(float(preset["skel_min_branch_mm"]))
        if "skel_close_gaps" in preset:
            self.skel_close_gaps.set(bool(preset["skel_close_gaps"]))

    def _on_preset_selected(self, event=None):
        name = self.language_preset.get()
        self.apply_language_preset(name)
        self.status.config(text=f"Preset applied: {name}")


    def _build(self):
        frm = ttk.Frame(self, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        ttk.Label(frm, text="Type text:").grid(row=0, column=0, sticky="w")
        self.txt = tk.Text(frm, width=60, height=6)
        self.txt.grid(row=1, column=0, columnspan=4, sticky="nsew", pady=(4, 10))
        frm.rowconfigure(1, weight=1)
        frm.columnconfigure(1, weight=1)

        # Controls
        def add_row(r, label, widget):
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", pady=2)
            widget.grid(row=r, column=1, sticky="ew", pady=2)

        ttk.Label(frm, text="Language preset:").grid(row=2, column=0, sticky="w", pady=2)

        preset_box = ttk.Combobox(
            frm,
            textvariable=self.language_preset,
            values=list(self.language_presets.keys()),
            state="readonly",
            width=28,
        )
        preset_box.grid(row=2, column=1, sticky="ew", pady=2)
        preset_box.bind("<<ComboboxSelected>>", self._on_preset_selected)


        ttk.Separator(frm).grid(row=3, column=0, columnspan=4, sticky="ew", pady=8)


        ttk.Label(frm, text="Mask method:").grid(row=4, column=0, sticky="w", pady=2)

        mask_box = ttk.Combobox(
            frm,
            textvariable=self.mask_method,
            values=MASK_METHODS,
            state="readonly",
            width=28,
        )
        mask_box.grid(row=4, column=1, sticky="ew", pady=2)


        add_row(5, "Font family:", ttk.Entry(frm, textvariable=self.font_family))
        add_row(6, "Font size (mm):", ttk.Entry(frm, textvariable=self.font_size_mm))
        add_row(7, "Sample step (mm):", ttk.Entry(frm, textvariable=self.sample_step_mm))



        ttk.Separator(frm).grid(row=8, column=0, columnspan=4, sticky="ew", pady=8)

        add_row(9, "Skeleton px/mm:", ttk.Entry(frm, textvariable=self.skel_px_per_mm))
        add_row(10, "Closed-loop tol (mm):", ttk.Entry(frm, textvariable=self.skel_close_mm))
        add_row(11, "Min branch length (mm):", ttk.Entry(frm, textvariable=self.skel_min_branch_mm))
        ttk.Checkbutton(frm, text="Close tiny gaps before skeletonize",
                        variable=self.skel_close_gaps).grid(row=12, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Separator(frm).grid(row=13, column=0, columnspan=4, sticky="ew", pady=8)

        ttk.Label(frm, text="Mask preview (bw):").grid(row=14, column=0, sticky="w", pady=(8, 2))
        self.preview = ttk.Label(frm)
        self.preview.grid(row=15, column=0, columnspan=4, sticky="w", pady=(0, 6))
        self._preview_imgtk = None  # keep a reference so Tk doesn't garbage-collect

        btn = ttk.Button(frm, text="Generate SVG…", command=self.on_generate)
        btn.grid(row=16, column=0, sticky="w")

        self.status = ttk.Label(frm, text="Ready.")
        self.status.grid(row=16, column=1, columnspan=3, sticky="w")

    def on_generate(self):
        text = self.txt.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("No text", "Type something first.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Save centerline SVG",
            defaultextension=".svg",
            filetypes=[("SVG files", "*.svg")]
        )
        if not out_path:
            return

        try:
            file_exists_or_raise(INKSCAPE_EXE, f"Inkscape not found at: {INKSCAPE_EXE}")

            self.status.config(text="Converting… (generating mask preview)")
            self.update_idletasks()


            bw, polylines = text_to_centerline_polylines(
                text=text,
                font_family=self.font_family.get(),
                font_size_mm=float(self.font_size_mm.get()),
                sample_step_mm=float(self.sample_step_mm.get()),
                skel_px_per_mm=int(self.skel_px_per_mm.get()),
                skel_close_mm=float(self.skel_close_mm.get()),
                skel_min_branch_mm=float(self.skel_min_branch_mm.get()),
                skel_close_gaps=bool(self.skel_close_gaps.get()),
                mask_method=self.mask_method.get(),
            )


            # --- show bw preview immediately ---
            # Convert to 8-bit grayscale image
            img = Image.fromarray((bw.astype(np.uint8) * 255), mode="L")

            # Scale up for visibility (nearest-neighbor keeps pixels crisp)
            SCALE = 2
            img = img.resize((img.size[0] * SCALE, img.size[1] * SCALE), Image.NEAREST)

            # Optional: cap display size so it doesn’t get huge
            MAX_W, MAX_H = 900, 250
            if img.size[0] > MAX_W or img.size[1] > MAX_H:
                s = min(MAX_W / img.size[0], MAX_H / img.size[1])
                img = img.resize((int(img.size[0] * s), int(img.size[1] * s)), Image.NEAREST)

            self._preview_imgtk = ImageTk.PhotoImage(img)
            self.preview.config(image=self._preview_imgtk)
            self.update_idletasks()

            # --- now write SVG ---
            polylines_to_svg(polylines, out_path)


            self.status.config(text=f"Saved: {out_path}")
            messagebox.showinfo("Done", f"Saved centerline SVG:\n{out_path}")

        except Exception as e:
            self.status.config(text="Error.")
            messagebox.showerror("Error", str(e))


def main():
    app = App()
    app.minsize(700, 420)
    app.mainloop()


if __name__ == "__main__":
    main()
