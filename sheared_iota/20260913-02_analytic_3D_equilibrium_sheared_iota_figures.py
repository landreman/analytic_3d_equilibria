"""Figures for the sheared-transform analytic equilibrium in the companion paper.

Run with the project virtual environment, for example from the project root:
    venv/bin/python sheared_iota/20260913-02_analytic_3D_equilibrium_sheared_iota_figures.py
    venv/bin/python sheared_iota/20260913-02_analytic_3D_equilibrium_sheared_iota_figures.py --interactive --configuration B

The three section plots use exact intersections with fixed physical azimuth,
not confocal-angle cuts. They show the cylindrical radius R and the vertical
coordinate Z. The PyVista 3-D figure defaults to case B and
shows two nonclosed analytic field-line segments at true Cartesian scale.
--configuration A|B|C selects the 3-D case; all three section plots are saved.
--interactive opens the PyVista viewer after saving the PNG and PDF figures.
The selected 3-D case replaces the existing *_geometry.png and *_geometry.pdf.
Dependencies: NumPy, SciPy, Matplotlib, and PyVista.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.integrate import solve_ivp


HERE = Path(__file__).resolve().parent
PREFIX = "20260913-02_analytic_3D_equilibrium_sheared_iota"
PHIS = (0.0, np.pi / 4, np.pi / 2)
PHI_NAMES = (r"$\phi=0$", r"$\phi=\pi/4$", r"$\phi=\pi/2$")
PHI_COLORS = ("red", "green", "blue")
FLUX_FRACTIONS = ((1.0, "-"), (4 / 9, "--"), (1 / 9, ":"))


@dataclass(frozen=True)
class Case:
    name: str
    epsilon: float
    S: float
    k_b: float
    lam: float

    @property
    def delta(self) -> float:
        return self.k_b**2 / 2

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


# Case A has a nearly circular axis and moderate cross-section elongation;
# cases B and C retain the more strongly rotating elongation. Finite-boundary
# vacuum transform is a separate harmonic-field problem, not solved here.
CASES = (
    Case("A", 1.08, 3.0, 0.70, 3.5),
    # Case("B", 4.0, 3.5, 0.70, 4.25),
    Case("B", 4.0, 3.5, 0.70, 3.5),
    # Case("C", 5.6, 4.0, 0.80, 4.8),
    Case("C", 5.6, 4.0, 0.80, 2.0),
)

plt.rcParams.update({"font.size": 15, "font.family": "serif",
                     "mathtext.fontset": "cm", "pdf.fonttype": 42})


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


def axis_radius(phi: float, case: Case) -> float:
    a, b = semiaxes(np.asarray(case.S), case)
    return float(1 / np.sqrt(np.cos(phi)**2 / a**2 + np.sin(phi)**2 / b**2))


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


def cross_section(k: float, phi: float, chi: np.ndarray, case: Case) -> np.ndarray:
    """Invert the exact map at fixed cylindrical phi by safeguarded bisection."""
    chi = np.asarray(chi)
    P, Y = -k * np.cos(chi), k * np.sin(chi)
    root = np.sqrt(1 - Y**2)
    half_width = np.arcsin(k)
    lower = np.full_like(chi, case.S - half_width)
    upper = np.full_like(chi, case.S + half_width)

    def residual(sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = semiaxes(sigma, case)
        t = np.arctan2(a * np.sin(phi), b * np.cos(phi))
        nu = case.epsilon / 2 * np.sin(2 * t)
        value = (sigma - case.S
                 - np.arctan(np.tanh(nu) * Y / root)
                 + np.arcsin(P / np.sqrt(np.cosh(nu)**2 - Y**2)))
        return value, t

    for _ in range(54):
        middle = (lower + upper) / 2
        value, _ = residual(middle)
        lower = np.where(value < 0, middle, lower)
        upper = np.where(value >= 0, middle, upper)
    sigma = (lower + upper) / 2
    value, t = residual(sigma)
    if np.max(np.abs(value)) > 2e-13:
        raise RuntimeError("Fixed-azimuth coordinate inversion failed")
    xyz = surface(k, chi, t, case)
    actual_phi = np.arctan2(xyz[..., 1], xyz[..., 0])
    if np.max(np.abs(np.angle(np.exp(1j * (actual_phi - phi))))) > 2e-13:
        raise RuntimeError("Cross-section does not lie in its toroidal plane")
    return np.column_stack((np.hypot(xyz[..., 0], xyz[..., 1]), xyz[..., 2]))


def section_limits(chi: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """One common equal-metric viewing window for all three section figures."""
    points = np.concatenate([
        cross_section(case.k_b, phi, chi, case)
        for case in CASES for phi in PHIS
    ])
    r_min, r_max = points[:, 0].min() - 0.04, points[:, 0].max() + 0.04
    z_extent = max(abs(points[:, 1].min()), abs(points[:, 1].max())) + 0.025
    return ((r_min, r_max), (-z_extent, z_extent))


def green_section_elongation(case: Case, chi: np.ndarray) -> float:
    """Major/minor spread of the boundary cut at phi=pi/4."""
    rz = cross_section(case.k_b, np.pi / 4, chi, case)
    eigenvalues = np.linalg.eigvalsh(np.cov(rz.T))
    return float(np.sqrt(eigenvalues[1] / eigenvalues[0]))


def make_sections(case: Case, chi: np.ndarray,
                  limits: tuple[tuple[float, float], tuple[float, float]]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for fraction, linestyle in FLUX_FRACTIONS:
        for phi, name, color in zip(PHIS, PHI_NAMES, PHI_COLORS):
            rz = cross_section(case.k_b * np.sqrt(fraction), phi, chi, case)
            ax.plot(*rz.T, color=color, linestyle=linestyle, linewidth=1.65,
                    label=name if fraction == 1 else None)
    for phi, color in zip(PHIS, PHI_COLORS):
        ax.plot(axis_radius(phi, case), 0, ".", color=color,
                markersize=7, zorder=5)
    ax.set(xlabel=r"$R$", ylabel=r"$Z$",
           xlim=limits[0], ylim=limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.05), fontsize=12, frameon=False,
               handlelength=1.5, columnspacing=1.8)
    ax.set_title(rf"{case.name}: $\epsilon={case.epsilon:g},\ S={case.S:g},\ "
                 rf"\delta={case.delta:g},\ \lambda={case.lam:g}$", fontsize=13)
    ax.tick_params(labelsize=12)
    fig.tight_layout(rect=(0, 0.12, 1, 1), pad=0.35)
    path = HERE / f"{PREFIX}_cross_sections_case_{case.name}"
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


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


def geometry_camera(points: np.ndarray) -> dict:
    """Fit the projected vertices, including tubes, with a 16-pixel margin.

    The image height follows the projected geometry's aspect ratio. A compact
    header is reserved above it. Parallel projection preserves equal physical
    scales, and fitting actual vertices avoids empty bounding-box corners.
    """
    width, padding, header = 1400, 16, 112
    elevation, azimuth = np.deg2rad([26, -60])
    direction = np.array([np.cos(elevation) * np.cos(azimuth),
                          np.cos(elevation) * np.sin(azimuth),
                          np.sin(elevation)])
    right = np.cross([0.0, 0.0, 1.0], direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    basis = np.column_stack((right, up, direction))
    projected = points @ basis
    lower, upper = projected.min(axis=0), projected.max(axis=0)
    span = upper - lower
    pixels_per_unit = (width - 2 * padding) / span[0]
    height = int(np.ceil(span[1] * pixels_per_unit)) + 2 * padding + header
    center = basis @ ((lower + upper) / 2)
    # Raising the focal point places the object below the header on screen.
    focal = center + up * header / (2 * pixels_per_unit)
    position = focal + direction * (3 * np.linalg.norm(span))
    return dict(window_size=(width, height), position=position, focal=focal,
                up=up, parallel_scale=height / (2 * pixels_per_unit),
                padding=padding, header=header)


def check_geometry_framing(plotter, points: np.ndarray, fit: dict) -> None:
    """Check VTK's actual projection, including the near/far clipping planes."""
    width, height = fit["window_size"]
    matrix = plotter.camera.GetCompositeProjectionTransformMatrix(width / height, 0, 1)
    matrix = np.array([[matrix.GetElement(i, j) for j in range(4)] for i in range(4)])
    projected = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    ndc = projected[:, :3] / projected[:, 3, None]
    pixels = (ndc[:, :2] + 1) * np.array([width, height]) / 2
    lo, hi = pixels.min(axis=0), pixels.max(axis=0)
    pad = fit["padding"] - 1  # allow pixel rounding when choosing image height
    if (np.any(lo < pad) or hi[0] > width - pad
            or hi[1] > height - fit["header"] - pad
            or ndc[:, 2].min() < 0 or ndc[:, 2].max() > 1):
        raise RuntimeError(f"3-D geometry would be clipped: pixel bounds {lo}, {hi}")


def make_geometry(case: Case, interactive: bool = False) -> None:
    """Save the fitted PyVista scene, then optionally enter its interactive viewer."""
    import pyvista as pv

    chi = np.linspace(0, 2 * np.pi, 145)
    t = np.linspace(0, 2 * np.pi, 385)
    xyz = surface(case.k_b, chi[:, None], t[None, :], case)
    mesh = pv.StructuredGrid(xyz[..., 0], xyz[..., 1], xyz[..., 2])
    t_line = np.linspace(0, 4 * np.pi, 2400)
    tubes = []
    for chi0, color in ((0.25, "#c54f3a"), (3.39, "#ffb413")):
        points = field_line(case.k_b, chi0, t_line, case)
        tubes.append((pv.lines_from_points(points).tube(radius=0.009, n_sides=16), color))
    center = axis(np.linspace(0, 2 * np.pi, 1000, endpoint=False), case)
    axis_tube = pv.lines_from_points(center, close=True).tube(radius=0.012, n_sides=16)
    all_points = np.concatenate([mesh.points, axis_tube.points]
                                + [tube.points for tube, _ in tubes])
    fit = geometry_camera(all_points)
    plotter = pv.Plotter(off_screen=not interactive, window_size=fit["window_size"])
    try:
        plotter.set_background("white")
        plotter.add_mesh(mesh, color="#77afc1", opacity=0.60,
                         smooth_shading=True, show_edges=False,
                         ambient=0.35, diffuse=0.6, specular=0.2)
        for tube, color in tubes:
            plotter.add_mesh(tube, color=color, smooth_shading=True)
        plotter.add_mesh(axis_tube, color="#242424", smooth_shading=True)
        width, height = fit["window_size"]
        text_font_size = 22
        title = plotter.add_text(
            f"Case {case.name}: epsilon={case.epsilon:g}, S={case.S:g}, "
            f"delta={case.delta:g}, lambda={case.lam:g}",
            position=(16, height - 8), font_size=text_font_size, color="#242424")
        title.GetTextProperty().SetVerticalJustificationToTop()
        for x, text, color in ((16, "Flux surface", "#77afc1"),
                               (int(width * 0.30), "Magnetic axis", "#242424"),
                               (int(width * 0.62), "Field-line segments", "#c54f3a")):
            label = plotter.add_text(text, position=(x, height - 62),
                                     font_size=text_font_size, color=color)
            label.GetTextProperty().SetVerticalJustificationToTop()
        plotter.camera_position = (fit["position"], fit["focal"], fit["up"])
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = fit["parallel_scale"]
        plotter.reset_camera_clipping_range()
        plotter.show(interactive=False, auto_close=False)
        check_geometry_framing(plotter, all_points, fit)
        path = HERE / f"{PREFIX}_geometry"
        plotter.screenshot(path.with_suffix(".png"))
        plotter.save_graphic(path.with_suffix(".pdf"))
        if interactive:
            plotter.show(interactive=True, auto_close=False)
    finally:
        plotter.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interactive", action="store_true",
                        help="open the PyVista geometry after saving the figures")
    parser.add_argument("--configuration", "--config", type=str.upper,
                        choices=[case.name for case in CASES],
                        default="B", help="3-D case: A, B, or C (default: B)")
    args = parser.parse_args(argv)
    for case in CASES:
        case.validate()
    chi = np.linspace(0, 2 * np.pi, 1001)
    limits = section_limits(chi)
    for case in CASES:
        make_sections(case, chi, limits)
    elongations = {case.name: green_section_elongation(case, chi) for case in CASES}
    geometry_case = (next(case for case in CASES if case.name == args.configuration)
                     if args.configuration else
                     min(CASES, key=lambda case: elongations[case.name]))
    make_geometry(geometry_case, interactive=args.interactive)
    for case in CASES:
        print(f"Case {case.name}: axis Rmax/Rmin={case.axis_ratio:.6f}, "
              f"phi=pi/4 elongation={elongations[case.name]:.3f}")
    print(f"3-D configuration: {geometry_case.name}")
    print(f"Figures written to {HERE}")


if __name__ == "__main__":
    main()
