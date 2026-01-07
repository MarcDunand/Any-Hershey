#!/usr/bin/env python3
"""
test_vpype.py

Quick smoke test to verify vpype works end-to-end in *this* Python environment.
Creates a tiny SVG, runs the same vpype pipeline your app uses, and verifies output.

Run (PowerShell, venv activated):
    python .\test_vpype.py
"""

from __future__ import annotations

import os
import sys
import tempfile


def main() -> int:
    try:
        from vpype_cli import execute
    except Exception as e:
        print("FAIL: could not import vpype_cli.execute")
        print("Error:", repr(e))
        return 2

    svg = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="10mm" height="10mm" viewBox="0 0 10 10">
  <path d="M 1 1 L 9 9" fill="none" stroke="black" stroke-width="0.3"
        stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M 1 9 L 9 1" fill="none" stroke="black" stroke-width="0.3"
        stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

    pipeline = (
        'read "{inp}" '
        "linemerge --tolerance 0.1mm "
        "linesort "
        "linesimplify --tolerance 0.05mm "
        'write "{out}"'
    )

    with tempfile.TemporaryDirectory() as d:
        inp = os.path.join(d, "in.svg")
        out = os.path.join(d, "out.svg")

        with open(inp, "w", encoding="utf-8") as f:
            f.write(svg)

        try:
            execute(pipeline.format(inp=inp, out=out))
        except Exception as e:
            print("FAIL: vpype execute() raised an exception")
            print("Pipeline:", pipeline.format(inp=inp, out=out))
            print("Error:", repr(e))
            return 3

        if not os.path.exists(out):
            print("FAIL: vpype reported success but output file was not created:", out)
            return 4

        size = os.path.getsize(out)
        if size <= 0:
            print("FAIL: output file exists but is empty:", out)
            return 5

        print("OK: vpype pipeline executed successfully.")
        print("Output:", out)
        print("Bytes:", size)

    # Optional: show shapely presence (useful for packaging expectations)
    try:
        import shapely  # type: ignore

        print("INFO: shapely present:", getattr(shapely, "__version__", "unknown"))
    except Exception:
        print("INFO: shapely not importable (that can be fine too).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
