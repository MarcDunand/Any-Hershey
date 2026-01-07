#!/usr/bin/env python3
"""
text_to_centerline_svg.py

Generates hershey font svg for inputted text
Works for any language, fine-tuned for a short list of common languages

Pipeline:
  1) Use Inkscape headless to convert TEXT to OUTLINE PATHS (font outlines).
  2) Either:
      A) Rasterize filled outlines via Inkscape -> skeletonize -> vectorize, or
      B) Sample outline paths -> XOR-fill -> skeletonize -> vectorize
  3) Export resulting polylines as SVG paths (stroke only, no fill).

Requires:
  Inkscape installed (we call it headlessly).
"""

import os
import math
import time
import json
import tempfile
import subprocess
import shutil
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
from math import hypot


import numpy as np
from PIL import Image, ImageDraw, ImageChops
from skimage.morphology import skeletonize, binary_closing, square
from svgpathtools import svg2paths2
from xml.sax.saxutils import escape as xml_escape


# UI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox



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


# CONFIG

# Optional override: set this if Inkscape is in a non-standard location.
# If None, we will auto-detect.
INKSCAPE_EXE: str | None = None

# Inkscape uses 96 px per inch
PX_PER_MM = 96.0 / 25.4

# Filename containing preset values for text in a specific alphabet
PRESETS_FILENAME = "language_presets.json"

# Methods used to create the rasterized bitmask of characters
MASK_METHODS = ["Inkscape Raster", "XOR"]

# Default vpype parameters
VP_LINEMERGE_TOL_MM = 0.10
VP_LINESIMPLIFY_TOL_MM = 0.05

DEBUG = False

if DEBUG:
    from PIL import ImageTk
else:
    ImageTk = None

# =========================
# Utilities
# =========================

def file_exists_or_raise(path: str, msg: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(msg)


def find_inkscape_executable() -> str:
    """Return a path/command we can use to invoke Inkscape.

    Resolution order:
      1) INKSCAPE_EXE if set (and exists)
      2) inkscape found on PATH
      3) common install locations (Windows/macOS)
    """
    if INKSCAPE_EXE:
        if os.path.exists(INKSCAPE_EXE):
            return INKSCAPE_EXE
        raise FileNotFoundError(f"Inkscape not found at configured path: {INKSCAPE_EXE}")

    # 1) PATH (works well when user installed normally or via package manager)
    found = shutil.which("inkscape")
    if found:
        return found

    # 2) Common install locations
    candidates: list[str] = []

    if os.name == "nt":
        # Typical per-machine install
        candidates += [
            r"C:\Program Files\Inkscape\bin\inkscape.exe",
            r"C:\Program Files\Inkscape\inkscape.exe",
            r"C:\Program Files (x86)\Inkscape\bin\inkscape.exe",
            r"C:\Program Files (x86)\Inkscape\inkscape.exe",
        ]

        # Microsoft Store installs sometimes land under WindowsApps (often inaccessible),
        # but we can at least try PATH first (above). We avoid hardcoding WindowsApps.

    elif sys.platform == "darwin":
        # Standard macOS app bundle
        candidates += [
            "/Applications/Inkscape.app/Contents/MacOS/inkscape",
            str(Path.home() / "Applications/Inkscape.app/Contents/MacOS/inkscape"),
        ]
    else:
        # Linux: PATH is usually enough; keep a couple common paths anyway
        candidates += ["/usr/bin/inkscape", "/usr/local/bin/inkscape"]

    for c in candidates:
        if os.path.exists(c):
            return c

    raise FileNotFoundError(
        "Inkscape was not found.\n\n"
        "Install Inkscape (v1.0+ recommended), or set INKSCAPE_EXE in the CONFIG section.\n\n"
        "Tried:\n- inkscape on PATH\n- common install locations"
    )


def vpype_optimize_svg(
    input_svg_path: str,
    output_svg_path: str,
    *,
    linemerge_tol_mm: float = VP_LINEMERGE_TOL_MM,
    linesimplify_tol_mm: float = VP_LINESIMPLIFY_TOL_MM,
) -> None:
    """Run vpype 'Approach A' IN-PROCESS via vpype_cli.execute()."""

    try:
        from vpype_cli import execute
    except Exception as e:
        raise RuntimeError(
            "vpype is not available in this Python environment.\n\n"
            "If you're running from source, install it in this venv:\n"
            "  pip install vpype\n\n"
            f"Import error: {e}"
        )

    # vpype parses a CLI-like pipeline string. Quote paths in case they contain spaces.
    in_q = '"' + input_svg_path.replace('"', '\\"') + '"'
    out_q = '"' + output_svg_path.replace('"', '\\"') + '"'

    pipeline = (
        f"read {in_q} "
        f"linemerge --tolerance {linemerge_tol_mm}mm "
        f"linesort "
        f"linesimplify --tolerance {linesimplify_tol_mm}mm "
        f"write --restore-attribs {out_q}"
    )

    try:
        execute(pipeline)
    except Exception as e:
        # execute() raises normal Python exceptions (not subprocess return codes)
        raise RuntimeError(
            "vpype optimization failed.\n\n"
            f"Pipeline:\n{pipeline}\n\n"
            f"Error:\n{e}"
        )

def resource_path(rel_path: str) -> str:
    """
    Return an absolute path to a resource file.

    Works for:
    - running from source
    - PyInstaller onefile/onedir (sys._MEIPASS)
    - app bundles where resources are copied next to the executable
    """
    # PyInstaller onefile sets sys._MEIPASS to the temp extraction dir
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return os.path.join(base, rel_path)

    # When packaged, you often want resources next to the executable
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    candidate = os.path.join(exe_dir, rel_path)
    if os.path.exists(candidate):
        return candidate

    # Source run: directory containing this file
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)





# =========================
# Language presets (JSON)
# =========================

def load_language_presets() -> dict:
    path = resource_path(PRESETS_FILENAME)

    if not os.path.exists(path):
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
    Create an SVG containing multiline <text> using <tspan> so that newlines render.

    NOTE: This SVG still depends on fonts, so we immediately run it through
    Inkscape 'object-to-path' to bake it into geometric outlines.
    """
    font_px = font_size_mm * PX_PER_MM

    # Line spacing: 1.2em is a safe default for most fonts/scripts
    line_height_px = font_px * 1.2

    # Keep empty lines (splitlines() drops trailing empty if text ends with \n, so handle that)
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    # Escape XML special chars so input like "&" doesn't break the SVG
    lines = [xml_escape(line) for line in lines]

    # Build tspans; dy=0 for first line, then dy=line_height for subsequent lines
    tspans = []
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else line_height_px
        # Even empty lines should advance; use a non-breaking space so the tspan exists
        safe_line = line if line != "" else "&#160;"  # NBSP
        tspans.append(f'<tspan x="0" dy="{dy:.2f}">{safe_line}</tspan>')

    tspans_xml = "\n    ".join(tspans)

    # Large canvas to avoid clipping; baseline at y=font_px for first line.
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="1000mm" height="300mm"
     viewBox="0 0 {1000*PX_PER_MM:.2f} {300*PX_PER_MM:.2f}">
  <text x="0" y="{font_px:.2f}" font-family="{font_family}"
        font-size="{font_px:.2f}" xml:space="preserve">
    {tspans_xml}
  </text>
</svg>'''



def inkscape_text_to_paths(text: str, font_family: str, font_size_mm: float) -> str:
    """
    Run Inkscape headless to convert text -> outline paths.
    Returns a filename to a temp SVG containing only paths.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        in_svg = os.path.join(tmpdir, "in.svg")
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

        inkscape_cmd = find_inkscape_executable()
        proc = subprocess.run(
            [inkscape_cmd, in_svg, "--actions", actions],
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

    final_svg = os.path.join(tempfile.gettempdir(), f"outline_{int(time.time() * 1000)}.svg")
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

        inkscape_cmd = find_inkscape_executable()
        proc = subprocess.run(
            [inkscape_cmd, outline_svg, "--actions", actions],
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
    return arr < 128


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

    Important: we preserve document order and subpath order so the result
    is stable and “glyph-ish” before centerline conversion.
    """
    paths, _, _ = svg2paths2(svg_file)
    if not paths:
        return ShapedRun(text="", width_mm=0.0, height_mm=0.0, polylines_mm=[])

    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    for p in paths:
        bx = p.bbox()
        if bx is None:
            continue
        x0, x1, y0, y1 = bx  # (xmin, xmax, ymin, ymax)
        xmin = min(xmin, x0)
        xmax = max(xmax, x1)
        ymin = min(ymin, y0)
        ymax = max(ymax, y1)

    if not math.isfinite(xmin):
        return ShapedRun(text="", width_mm=0.0, height_mm=0.0, polylines_mm=[])

    width_mm = (xmax - xmin) / PX_PER_MM
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
    """Treat a polyline as a closed loop if endpoints are within close_mm."""
    if len(seg) < 3:
        return False
    x0, y0 = seg[0]
    x1, y1 = seg[-1]
    return hypot(x0 - x1, y0 - y1) <= close_mm


def _vectorize_skeleton(skel_bool: np.ndarray, px_per_mm: float) -> List[List[Tuple[float, float]]]:
    """
    Convert a 1-pixel-wide skeleton image into polylines.

    - Skeleton pixels form a graph
    - We walk from nodes (degree != 2) to extract strokes
    - Then extract leftover loops (pure cycles)
    - Convert (y,x) pixels -> (x,y) in millimeters
    """
    pix = set(map(tuple, np.argwhere(skel_bool)))
    if not pix:
        return []

    # 8-connected neighborhood (handles diagonals)
    NBR = [(-1, -1), (-1, 0), (-1, 1),
           (0, -1),           (0, 1),
           (1, -1),  (1, 0),  (1, 1)]

    def neighbors(p):
        y, x = p
        for dy, dx in NBR:
            q = (y + dy, x + dx)
            if q in pix:
                yield q

    deg = {p: sum(1 for _ in neighbors(p)) for p in pix}

    visited = set()
    polylines_px = []

    def grow_from(start):
        """
        Walk forward along the skeleton until we can't extend without revisiting.
        At junctions, pick the neighbor that best continues direction to reduce zig-zagging.
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

    # (A) Extract paths starting at nodes/ends (degree != 2)
    for s, d in deg.items():
        if d != 2 and s not in visited:
            visited.add(s)
            poly = grow_from(s)
            if len(poly) >= 2:
                polylines_px.append(poly)

    # (B) Extract leftover loops (all degree==2 cycles)
    for s in list(pix):
        if s not in visited:
            visited.add(s)
            poly = grow_from(s)
            if len(poly) >= 2:
                polylines_px.append(poly)

    # Convert pixels -> mm
    out = []
    for poly in polylines_px:
        out.append([(x / px_per_mm, y / px_per_mm) for (y, x) in poly])
    return out


def centerlines_from_outlines(
    polylines_mm: List[List[Tuple[float, float]]],
    px_per_mm: int = 38,
    close_tol_mm: float = 0.10,
    do_close_gaps: bool = False,
    return_bw: bool = False,
):
    """
    OUTLINES (filled shapes) -> CENTERLINES (single stroke polylines)

    1) Rasterize outline polylines into a filled binary mask image (XOR fill to respect holes)
    2) Optionally close tiny gaps in the mask (morphological closing)
    3) Skeletonize the filled mask -> 1-pixel-wide skeleton
    4) Vectorize skeleton pixels back into polylines
    """
    if not polylines_mm:
        return [] if not return_bw else (np.zeros((1, 1), dtype=bool), [])

    maxx = max(x for seg in polylines_mm for (x, _) in seg)
    maxy = max(y for seg in polylines_mm for (_, y) in seg)

    W = max(2, int(np.ceil(maxx * px_per_mm)))
    H = max(2, int(np.ceil(maxy * px_per_mm)))

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

    skel = skeletonize(bw, method="lee")
    centerlines = _vectorize_skeleton(skel, px_per_mm=float(px_per_mm))

    return (bw, centerlines) if return_bw else centerlines


# =========================
# Export polylines -> SVG
# =========================

def polylines_to_svg(
    polylines_mm: List[List[Tuple[float, float]]],
    out_svg_path: str,
    margin_mm: float = 2.0,
    stroke_width_mm: float = 0.3
) -> None:
    """
    Write a minimal SVG containing one <path> per polyline.
    """
    if not polylines_mm:
        W = H = 10.0
    else:
        maxx = max(x for seg in polylines_mm for (x, _) in seg) + margin_mm
        maxy = max(y for seg in polylines_mm for (_, y) in seg) + margin_mm
        W = max(10.0, maxx)
        H = max(10.0, maxy)

    def path_d(seg):
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
    skel_close_gaps: bool,
    mask_method: str,
) -> Tuple[np.ndarray, List[List[Tuple[float, float]]]]:
    """
    Convenience wrapper:
      text -> outline svg -> (mask) -> centerline polylines
    """
    outline_svg = inkscape_text_to_paths(text, font_family, font_size_mm)
    try:
        if mask_method == "Inkscape Raster":
            bw = rasterize_outline_svg_to_bw(outline_svg, px_per_mm=skel_px_per_mm)

            if skel_close_gaps:
                bw = binary_closing(bw, square(3))

            skel = skeletonize(bw, method="lee")
            centerlines = _vectorize_skeleton(skel, px_per_mm=float(skel_px_per_mm))
            return bw, centerlines

        shaped = load_and_sample(outline_svg, sample_step_mm)
        bw, centerlines = centerlines_from_outlines(
            shaped.polylines_mm,
            px_per_mm=skel_px_per_mm,
            close_tol_mm=skel_close_mm,
            do_close_gaps=skel_close_gaps,
            return_bw=True,
        )
        return bw, centerlines

    finally:
        try:
            os.remove(outline_svg)
        except Exception:
            pass


# =========================
# UI App
# =========================

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # Load presets from external JSON
        try:
            self.language_presets = load_language_presets()
        except Exception as e:
            self.language_presets = {"Default": {}}
            messagebox.showwarning("Preset load failed", f"Could not load {PRESETS_FILENAME}:\n\n{e}")

        def _tk_exception_handler(exc, val, tb):
            import traceback
            err = "".join(traceback.format_exception(exc, val, tb))
            messagebox.showerror("Unhandled exception", err)

        self.report_callback_exception = _tk_exception_handler

        self.title("Text → Centerline SVG (Inkscape + Skeletonize)")

        self.language_preset = tk.StringVar(value="Default")
        self.mask_method = tk.StringVar(value="Inkscape Raster")
        self.font_family = tk.StringVar(value="Noto Sans CJK SC")
        self.font_size_mm = tk.DoubleVar(value=8.0)
        self.sample_step_mm = tk.DoubleVar(value=1.0)

        self.skel_px_per_mm = tk.IntVar(value=38)
        self.skel_close_mm = tk.DoubleVar(value=0.10)
        self.skel_close_gaps = tk.BooleanVar(value=True)

        # vpype post-processing knobs (Approach A)
        self.vp_linemerge_tol_mm = tk.DoubleVar(value=VP_LINEMERGE_TOL_MM)
        self.vp_linesimplify_tol_mm = tk.DoubleVar(value=VP_LINESIMPLIFY_TOL_MM)


        self._build()

    def apply_language_preset(self, preset_name: str) -> None:
        preset = self.language_presets.get(preset_name, {})
        if not isinstance(preset, dict) or not preset:
            return

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
        if "skel_close_gaps" in preset:
            self.skel_close_gaps.set(bool(preset["skel_close_gaps"]))

        if "vp_linemerge_tol_mm" in preset:
            self.vp_linemerge_tol_mm.set(float(preset["vp_linemerge_tol_mm"]))
        if "vp_linesimplify_tol_mm" in preset:
            self.vp_linesimplify_tol_mm.set(float(preset["vp_linesimplify_tol_mm"]))



    def _on_preset_selected(self, event=None) -> None:
        name = self.language_preset.get()
        self.apply_language_preset(name)
        self.status.config(text=f"Preset applied: {name}")

    def _build(self) -> None:
        frm = ttk.Frame(self, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        ttk.Label(frm, text="Type text:").grid(row=0, column=0, sticky="w")
        self.txt = tk.Text(frm, width=60, height=6)
        self.txt.grid(row=1, column=0, columnspan=4, sticky="nsew", pady=(4, 10))
        frm.rowconfigure(1, weight=1)
        frm.columnconfigure(1, weight=1)

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
        ttk.Checkbutton(
            frm,
            text="Close tiny gaps before skeletonize",
            variable=self.skel_close_gaps
        ).grid(row=11, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Separator(frm).grid(row=12, column=0, columnspan=4, sticky="ew", pady=8)

        # vpype post-processing knobs
        add_row(13, "vpype linemerge tol (mm):", ttk.Entry(frm, textvariable=self.vp_linemerge_tol_mm))
        add_row(14, "vpype linesimplify tol (mm):", ttk.Entry(frm, textvariable=self.vp_linesimplify_tol_mm))

        ttk.Separator(frm).grid(row=15, column=0, columnspan=4, sticky="ew", pady=8)


        self._preview_imgtk = None  # keep a reference so Tk doesn't garbage-collect

        if DEBUG:
            ttk.Label(frm, text="Mask preview (bw):").grid(row=16, column=0, sticky="w", pady=(8, 2))
            self.preview = ttk.Label(frm)
            self.preview.grid(row=17, column=0, columnspan=4, sticky="w", pady=(0, 6))
        else:
            # Still create the widget so on_generate() can safely call self.preview.config(...)
            self.preview = ttk.Label(frm)


        btn = ttk.Button(frm, text="Generate SVG…", command=self.on_generate)
        btn.grid(row=18, column=0, sticky="w")

        self.status = ttk.Label(frm, text="Ready.")
        self.status.grid(row=18, column=1, columnspan=3, sticky="w")

    def on_generate(self) -> None:
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
            # Ensure Inkscape is available (auto-detect)
            _ = find_inkscape_executable()

            self.status.config(text="Converting…")
            self.update_idletasks()

            bw, polylines = text_to_centerline_polylines(
                text=text,
                font_family=self.font_family.get(),
                font_size_mm=float(self.font_size_mm.get()),
                sample_step_mm=float(self.sample_step_mm.get()),
                skel_px_per_mm=int(self.skel_px_per_mm.get()),
                skel_close_mm=float(self.skel_close_mm.get()),
                skel_close_gaps=bool(self.skel_close_gaps.get()),
                mask_method=self.mask_method.get(),
            )

            if DEBUG:
                # Mask preview
                img = Image.fromarray((bw.astype(np.uint8) * 255), mode="L")

                SCALE = 2
                img = img.resize((img.size[0] * SCALE, img.size[1] * SCALE), Image.NEAREST)

                MAX_W, MAX_H = 900, 250
                if img.size[0] > MAX_W or img.size[1] > MAX_H:
                    s = min(MAX_W / img.size[0], MAX_H / img.size[1])
                    img = img.resize((int(img.size[0] * s), int(img.size[1] * s)), Image.NEAREST)

                self._preview_imgtk = ImageTk.PhotoImage(img)
                self.preview.config(image=self._preview_imgtk)
                self.update_idletasks()
            else:
                # Ensure preview is blank when debug is off
                self.preview.config(image="")
                self._preview_imgtk = None

            # --- write raw SVG to a temp file, then vpype-optimize into the chosen output path ---
            raw_svg = os.path.join(tempfile.gettempdir(), f"raw_{int(time.time() * 1000)}.svg")
            polylines_to_svg(polylines, raw_svg)

            try:
                lm = max(0.0, float(self.vp_linemerge_tol_mm.get()))
                ls = max(0.0, float(self.vp_linesimplify_tol_mm.get()))

                vpype_optimize_svg(
                    raw_svg,
                    out_path,
                    linemerge_tol_mm=lm,
                    linesimplify_tol_mm=ls,
                )
            except Exception as vp_err:
                # Fall back to raw output if vpype fails for any reason.
                shutil.copyfile(raw_svg, out_path)
                messagebox.showwarning(
                    "vpype optimization failed",
                    "Saved the unoptimized SVG instead.\n\n"
                    f"Details:\n{vp_err}"
                )
            finally:
                try:
                    os.remove(raw_svg)
                except Exception:
                    pass

            self.status.config(text=f"Saved: {out_path}")

        except Exception as e:
            self.status.config(text="Error.")
            messagebox.showerror("Error", str(e))


def main() -> None:
    app = App()
    app.minsize(700, 420)
    app.mainloop()


if __name__ == "__main__":
    main()
