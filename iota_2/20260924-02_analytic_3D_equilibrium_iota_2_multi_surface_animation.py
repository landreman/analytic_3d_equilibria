#!/usr/bin/env python
"""Render a seamless orbit of the iota=2 equilibrium as an animated GIF.

Run with the project virtual environment, for example:
    venv/bin/python iota_2/animation/20260924-02_analytic_3D_equilibrium_iota_2_multi_surface_animation.py

The GIF is written beside this script. Use --frames and --frame-duration-ms to
control the orbit's smoothness and duration; GIF timing is rounded to 10 ms.
Dependencies: NumPy, PyVista, and imageio.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "figures"))
from analytic_3d_equilibrium_iota_2 import axis, surface


DEFAULT_OUTPUT = HERE / f"{Path(__file__).stem}.gif"
EPSILON = 0.5
OUTER_PSI = 1 / 64
START_AZIMUTH = np.deg2rad(-60.0)
ELEVATION = np.deg2rad(26.0)
IMAGE_WIDTH = 800
IMAGE_PADDING = 14
FOOTER_HEIGHT = 56
# These are the same three flux labels used in the static cross-section figures.
SURFACE_STYLES = (
    (1 / 9, "#367da5", 0.5),
    (4 / 9, "#63b1bd", 0.5),
    (1, "#9bcfda", 0.5),
)


def orbit_framing(points: np.ndarray) -> tuple[tuple[int, int], float, float]:
    """Use one fixed camera crop that contains the geometry at every angle.

    Each vertex's cylindrical radius bounds its horizontal projection. Its
    vertical extrema over a complete orbit are z*cos(elevation) plus or minus
    radius*sin(elevation). A single crop keeps the origin stationary on screen.
    """
    radius = np.hypot(points[:, 0], points[:, 1])
    horizontal_bound = radius.max()
    vertical_low = np.min(points[:, 2] * np.cos(ELEVATION)
                          - radius * np.sin(ELEVATION))
    vertical_high = np.max(points[:, 2] * np.cos(ELEVATION)
                           + radius * np.sin(ELEVATION))
    pixels_per_unit = (IMAGE_WIDTH - 2 * IMAGE_PADDING) / (2 * horizontal_bound)
    height = (int(np.ceil((vertical_high - vertical_low) * pixels_per_unit))
              + 2 * IMAGE_PADDING + FOOTER_HEIGHT)
    parallel_scale = height / (2 * pixels_per_unit)
    window_center_y = ((vertical_low + vertical_high) * pixels_per_unit
                       - FOOTER_HEIGHT) / height
    return (IMAGE_WIDTH, height), parallel_scale, window_center_y


def make_animation(frames: int, frame_duration_ms: int, output: Path) -> None:
    import pyvista as pv

    beta = np.linspace(0, 2 * np.pi, 145)
    t = np.linspace(0, 2 * np.pi, 385)
    surfaces = []
    for fraction, color, opacity in SURFACE_STYLES:
        xyz = surface(OUTER_PSI * fraction, beta[:, None], t[None, :], EPSILON)
        mesh = pv.StructuredGrid(xyz[..., 0], xyz[..., 1], xyz[..., 2])
        surfaces.append((mesh, color, opacity))

    t_line = np.linspace(0, 2 * np.pi, 1600, endpoint=False)
    tubes = []
    for beta0, color in ((0.25, "#c54f3a"), (3.39, "#ffb413")):
        points = surface(OUTER_PSI, beta0, t_line, EPSILON)
        tubes.append((pv.lines_from_points(points, close=True).tube(
            radius=0.007, n_sides=16), color))
    center = axis(t_line, EPSILON)
    axis_tube = pv.lines_from_points(center, close=True).tube(
        radius=0.010, n_sides=16)
    all_points = np.concatenate([mesh.points for mesh, _, _ in surfaces]
                                + [axis_tube.points]
                                + [tube.points for tube, _ in tubes])
    angles = START_AZIMUTH + 2 * np.pi * np.arange(frames) / frames
    window_size, parallel_scale, window_center_y = orbit_framing(all_points)

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    try:
        plotter.set_background("white")
        for mesh, color, opacity in surfaces:
            plotter.add_mesh(mesh, color=color, opacity=opacity,
                             smooth_shading=True, show_edges=False,
                             ambient=0.35, diffuse=0.6, specular=0.2)
        for tube, color in tubes:
            plotter.add_mesh(tube, color=color, smooth_shading=True)
        plotter.add_mesh(axis_tube, color="#242424", smooth_shading=True)
        for x, label, color in ((18, "Flux surfaces", "#478ca4"),
                                (int(IMAGE_WIDTH * 0.31), "Magnetic axis", "#242424"),
                                (int(IMAGE_WIDTH * 0.65), "Field lines", "#c54f3a")):
            plotter.add_text(label, position=(x, 15), font_size=15, color=color)

        distance = 3 * np.ptp(all_points, axis=0).max()
        horizontal = distance * np.cos(ELEVATION)
        camera_height = distance * np.sin(ELEVATION)
        plotter.camera_position = (
            (horizontal * np.cos(START_AZIMUTH),
             horizontal * np.sin(START_AZIMUTH), camera_height),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = parallel_scale
        plotter.camera.window_center = (0, window_center_y)
        plotter.show(interactive=False, auto_close=False)
        # ImageIO truncates GIF durations to 10 ms units. The small offset
        # avoids a floating-point value just below the requested duration.
        plotter.open_gif(output, loop=0, fps=1000 / (frame_duration_ms + 0.1))
        # Omitting the duplicate endpoint makes the final-to-first step smooth.
        for angle in angles:
            plotter.camera_position = (
                (horizontal * np.cos(angle), horizontal * np.sin(angle), camera_height),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            )
            plotter.camera.parallel_projection = True
            plotter.camera.parallel_scale = parallel_scale
            plotter.camera.window_center = (0, window_center_y)
            plotter.reset_camera_clipping_range()
            plotter.write_frame()
    finally:
        plotter.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=120,
                        help="frames in one complete orbit (default: 120)")
    parser.add_argument("--frame-duration-ms", type=int, default=60,
                        help="time per frame in milliseconds (default: 60)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"output GIF path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args(argv)
    if args.frames < 3:
        parser.error("--frames must be at least 3")
    if args.frame_duration_ms < 10 or args.frame_duration_ms % 10:
        parser.error("--frame-duration-ms must be a multiple of 10 and at least 10")
    if args.output.suffix.lower() != ".gif":
        parser.error("--output must end in .gif")
    make_animation(args.frames, args.frame_duration_ms, args.output)
    print(f"Wrote {args.frames}-frame, indefinitely looping GIF to {args.output}")


if __name__ == "__main__":
    main()
