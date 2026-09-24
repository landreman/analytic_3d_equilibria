"""Render a seamless, indefinitely looping PyVista orbit of the analytic equilibrium.

Run from the project root with the project virtual environment, for example:
    venv/bin/python sheared_iota/animation/20260924-01_analytic_3D_equilibrium_sheared_iota_multi_surface_animation.py
    venv/bin/python sheared_iota/animation/20260924-01_analytic_3D_equilibrium_sheared_iota_multi_surface_animation.py --frames 90 --frame-duration-ms 60

The GIF is written beside this script. Use --frames and --frame-duration-ms to
control the orbit's smoothness and duration; GIF timing is rounded to 10 ms.
Dependencies: NumPy, SciPy, PyVista, and imageio.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / f"{Path(__file__).stem}.gif"
START_AZIMUTH = np.deg2rad(-60.0)
ELEVATION = np.deg2rad(26.0)
IMAGE_WIDTH = 800
IMAGE_PADDING = 18
FOOTER_HEIGHT = 62
# Each level is a fraction of the case's outer flux-surface radius k_b.
# Lower opacity on the outer surfaces keeps the nested interiors visible.
# size, color, opacity
SURFACE_STYLES = (
    (0.35, "#367da5", 0.5),
    (0.65, "#63b1bd", 0.5),
    (1.00, "#9bcfda", 0.5),
)


@dataclass(frozen=True)
class Case:
    name: str
    epsilon: float
    S: float
    k_b: float
    lam: float

    @property
    def axis_ratio(self) -> float:
        h = np.hypot(2 * self.S, self.epsilon)
        return float(np.sqrt((h + self.epsilon) / (h - self.epsilon)))

    def validate(self) -> None:
        if not (self.epsilon > 0 and self.lam > 0 and 0 < self.k_b < 1
                and self.S > np.arcsin(self.k_b)):
            raise ValueError(f"Case {self.name} violates the analytic domain")
        if not self.axis_ratio < 2:
            raise ValueError(f"Case {self.name} has axis Rmax/Rmin >= 2")


CASES = (
    Case("A", 1.08, 3.0, 0.70, 3.5),
    Case("B", 4.0, 3.5, 0.70, 3.5),
    Case("C", 5.6, 4.0, 0.80, 2.0),
)


def semiaxes(sigma: np.ndarray, case: Case) -> tuple[np.ndarray, np.ndarray]:
    """Confocal semiaxes, evaluating the smaller one without cancellation."""
    h = np.hypot(2 * sigma, case.epsilon)
    b = np.sqrt((h + case.epsilon) / 2)
    a = sigma / b
    return a, b


def axis(t: np.ndarray, case: Case) -> np.ndarray:
    a, b = semiaxes(np.asarray(case.S), case)
    t = np.asarray(t)
    return np.stack((a * np.cos(t), b * np.sin(t), np.zeros_like(t)), axis=-1)


def surface(k: float, chi: np.ndarray, t: np.ndarray, case: Case) -> np.ndarray:
    """Exact (P,Y,t) map on psi=k**2/2, using the paper's sign convention."""
    chi, t = np.broadcast_arrays(chi, t)
    P, Y = -k * np.cos(chi), k * np.sin(chi)
    nu = case.epsilon / 2 * np.sin(2 * t)
    sigma = (case.S + np.arctan(np.tanh(nu) * Y / np.sqrt(1 - Y**2))
             - np.arcsin(P / np.sqrt(np.cosh(nu)**2 - Y**2)))
    a, b = semiaxes(sigma, case)
    return np.stack((a * np.cos(t), b * np.sin(t),
                     -np.arcsin(Y) / case.lam), axis=-1)


def field_line(k: float, chi0: float, t: np.ndarray, case: Case) -> np.ndarray:
    """Integrate dchi/dt from the analytic MHD field-line equation."""
    def rhs(tt: float, chi: np.ndarray) -> np.ndarray:
        P, Y = -k * np.cos(chi[0]), k * np.sin(chi[0])
        nu = case.epsilon / 2 * np.sin(2 * tt)
        sigma = (case.S + np.arctan(np.tanh(nu) * Y / np.sqrt(1 - Y**2))
                 - np.arcsin(P / np.sqrt(np.cosh(nu)**2 - Y**2)))
        G = (np.hypot(2 * sigma, case.epsilon)
             + case.epsilon * np.cos(2 * tt)) / 2
        return np.array([2 * G * np.sqrt(1 - Y**2)
                         / np.sqrt(np.cosh(nu)**2 - k**2)])

    solution = solve_ivp(rhs, (float(t[0]), float(t[-1])), [chi0],
                         t_eval=t, rtol=2e-10, atol=2e-12)
    if not solution.success:
        raise RuntimeError(solution.message)
    return surface(k, solution.y[0], t, case)


def orbit_framing(points: np.ndarray) -> tuple[tuple[int, int], float, float]:
    """Fit every camera angle at one scale, leaving a narrow bottom label band.

    The horizontal bound is the greatest cylindrical radius. Maximizing the
    vertical projection of each vertex over all azimuths gives an exact bound
    for the entire orbit, so the image never clips as the camera moves.
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
    # A projection shift lifts the object above the footer while the camera
    # continues to point at the origin throughout the orbit.
    window_center_y = ((vertical_low + vertical_high) * pixels_per_unit
                       - FOOTER_HEIGHT) / height
    return (IMAGE_WIDTH, height), parallel_scale, window_center_y


def make_animation(case: Case, frames: int, frame_duration_ms: int,
                   output: Path) -> None:
    import pyvista as pv

    chi = np.linspace(0, 2 * np.pi, 145)
    t = np.linspace(0, 2 * np.pi, 385)
    surfaces = []
    for fraction, color, opacity in SURFACE_STYLES:
        xyz = surface(fraction * case.k_b, chi[:, None], t[None, :], case)
        mesh = pv.StructuredGrid(xyz[..., 0], xyz[..., 1], xyz[..., 2])
        surfaces.append((mesh, color, opacity))
    t_line = np.linspace(0, 4 * np.pi, 2400)
    tubes = []
    for chi0, color in ((0.25, "#c54f3a"), (3.39, "#ffb413")):
        points = field_line(case.k_b, chi0, t_line, case)
        tubes.append((pv.lines_from_points(points).tube(radius=0.009, n_sides=16), color))
    center = axis(np.linspace(0, 2 * np.pi, 1000, endpoint=False), case)
    axis_tube = pv.lines_from_points(center, close=True).tube(radius=0.012, n_sides=16)
    all_points = np.concatenate([mesh.points for mesh, _, _ in surfaces]
                                + [axis_tube.points]
                                + [tube.points for tube, _ in tubes])
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
                                (int(IMAGE_WIDTH * 0.65), "Field-line segments", "#c54f3a")):
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
        plotter.open_gif(output, loop=0, fps=1000 / frame_duration_ms)
        # Omit the duplicate 2*pi endpoint: the last-to-first step then has
        # the same angular size as every other frame transition.
        for angle in START_AZIMUTH + 2 * np.pi * np.arange(frames) / frames:
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
    parser.add_argument("--configuration", "--config", type=str.upper,
                        choices=[case.name for case in CASES], default="B",
                        help="3-D case: A, B, or C (default: B)")
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
    case = next(case for case in CASES if case.name == args.configuration)
    case.validate()
    make_animation(case, args.frames, args.frame_duration_ms, args.output)
    print(f"Wrote {args.frames}-frame, indefinitely looping GIF to {args.output}")


if __name__ == "__main__":
    main()
