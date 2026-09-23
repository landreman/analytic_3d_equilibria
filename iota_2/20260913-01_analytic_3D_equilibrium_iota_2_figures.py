"""Render the analytic equilibrium's 3D geometry and toroidal cross-sections.

Run from any working directory with the project virtual environment.
Dependencies: NumPy, Matplotlib, and PyVista. No external data are used.
Pass --interactive to view the PyVista geometry after saving the figures.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pyvista as pv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analytic_3d_equilibrium_iota_2 import axis, embedding, surface

GEOMETRY_EPSILON = 0.5
SECTION_EPSILONS = (0.25, 0.5, 0.7)
DELTA = 1 / 64
PREFIX = "20260913-01_analytic_3D_equilibrium_iota_2"
OUT = HERE
PHIS = (0, np.pi / 4, np.pi / 2)
PHI_NAMES = (r"$\phi=0$", r"$\phi=\pi/4$", r"$\phi=\pi/2$")
PHI_COLORS = ("red", "green", "blue")
FLUX_FRACTIONS = ((1, "-"), (4 / 9, "--"), (1 / 9, ":"))
PYVISTA_FONT_SIZE = 20

plt.rcParams.update({"font.size": 16, "font.family": "serif",
                     "mathtext.fontset": "cm", "pdf.fonttype": 42})


def cross_section(psi: float, phi: float, beta: np.ndarray,
                  epsilon: float) -> np.ndarray:
    """Exact intersection of a flux surface with a physical toroidal plane.

    Invert the linear map between (cos t, sin t) and (x, y) to obtain the
    parameter t whose azimuth is phi. This does not integrate field lines.
    """
    a, b = np.sqrt(1 + epsilon), np.sqrt(1 - epsilon)
    u, v = -epsilon / 2 + np.sqrt(psi) * np.cos(beta), np.sqrt(psi) * np.sin(beta)
    ell = np.sqrt((1 + np.sqrt(1 - 4 * (u * u + v * v))) / 2)
    m11, m12 = a * (ell + u / ell), a * v / ell
    m21, m22 = b * v / ell, b * (ell - u / ell)
    t = np.arctan2(m11 * np.sin(phi) - m21 * np.cos(phi),
                   m22 * np.cos(phi) - m12 * np.sin(phi))
    xyz = embedding(u, v, t, epsilon)
    actual = np.arctan2(xyz[:, 1], xyz[:, 0])
    assert np.max(np.abs(np.angle(np.exp(1j * (actual - phi))))) < 1e-12
    return np.column_stack((np.hypot(xyz[:, 0], xyz[:, 1]), xyz[:, 2]))


def make_geometry(interactive: bool = False) -> None:
    """Render the surface and closed field lines with PyVista."""
    beta = np.linspace(0, 2 * np.pi, 145)
    t = np.linspace(0, 2 * np.pi, 385)
    xyz = surface(DELTA, beta[:, None], t[None, :], GEOMETRY_EPSILON)
    mesh = pv.StructuredGrid(xyz[..., 0], xyz[..., 1], xyz[..., 2])

    plotter = pv.Plotter(off_screen=not interactive, window_size=(1200, 900))
    try:
        plotter.theme.font.size = PYVISTA_FONT_SIZE
        plotter.set_background("white")
        plotter.add_mesh(mesh, color="#77afc1", opacity=0.60,
                         smooth_shading=True, show_edges=False,
                         ambient=0.35, diffuse=0.6, specular=0.2)
        t_line = np.linspace(0, 2 * np.pi, 1600, endpoint=False)
        for beta0, color in ((0.25, "#c54f3a"), (3.39, "#ffb413")):
            points = surface(DELTA, beta0, t_line, GEOMETRY_EPSILON)
            tube = pv.lines_from_points(points, close=True).tube(radius=0.007,
                                                                  n_sides=16)
            plotter.add_mesh(tube, color=color, smooth_shading=True)
        center = axis(t_line, GEOMETRY_EPSILON)
        axis_tube = pv.lines_from_points(center, close=True).tube(radius=0.010,
                                                                  n_sides=16)
        plotter.add_mesh(axis_tube, color="#242424", smooth_shading=True)
        plotter.add_axes(color="#333333", label_size=(0.18, 0.08),
                 viewport=(0.02, 0.02, 0.17, 0.22))
        plotter.add_text("epsilon = 0.5,  psi = 1/64", position="upper_left",
                 font_size=PYVISTA_FONT_SIZE, color="#242424")
        legend = plotter.add_legend([("Flux surface", "#77afc1"),
                                     ("Magnetic axis", "#242424"),
                                     ("Field lines", "#c54f3a")],
                                    size=(0.29, 0.22), bcolor="white")
        # The default upper-right position clips the longest label.
        legend.SetPosition(0.58, 0.73)
        plotter.camera_position = [(2.5, -3.1, 2.0), (0, 0, 0), (0, 0, 1)]
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = 1.13
        plotter.show(interactive=False, auto_close=False)
        plotter.screenshot(OUT / f"{PREFIX}_geometry.png")
        plotter.save_graphic(OUT / f"{PREFIX}_geometry.pdf")
        if interactive:
            plotter.show(interactive=True, auto_close=False)
    finally:
        plotter.close()


def section_limits(beta: np.ndarray) -> tuple[tuple[float, float],
                                              tuple[float, float]]:
    """Use identical R and Z scales for the three epsilon figures."""
    sections = [cross_section(DELTA, phi, beta, epsilon)
                for epsilon in SECTION_EPSILONS for phi in PHIS]
    points = np.concatenate(sections)
    r_min, z_min = np.min(points, axis=0)
    r_max, z_max = np.max(points, axis=0)
    return ((r_min - 0.04, r_max + 0.04),
            (z_min - 0.04, z_max + 0.04))


def make_sections(epsilon: float, beta: np.ndarray,
                  limits: tuple[tuple[float, float], tuple[float, float]]) -> None:
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    for fraction, ls in FLUX_FRACTIONS:
        for phi, name, color in zip(PHIS, PHI_NAMES, PHI_COLORS):
            rz = cross_section(DELTA * fraction, phi, beta, epsilon)
            ax.plot(*rz.T, color=color, linestyle=ls, linewidth=1.6,
                    label=name if fraction == 1 else None)
    axis_radius = np.sqrt(1 - epsilon * epsilon)
    for phi, color, marker_size in zip(PHIS, PHI_COLORS, (7, 5, 3.5)):
        # At phi=0 and phi=pi/2 the axis points coincide in (R, Z).
        # Different marker sizes keep the red and blue dots both visible.
        ax.plot(axis_radius, epsilon / 2 * np.sin(2 * phi), color=color,
                marker="o", markersize=marker_size, linestyle="none")
    ax.set(xlabel=r"$R$", ylabel=r"$Z$", xlim=limits[0], ylim=limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.legend(loc="lower left", fontsize=13, frameon=False, handlelength=1.6,
              borderaxespad=0.15, labelspacing=0.2)
    ax.set_title(rf"$\epsilon={epsilon:g}$", fontsize=16)
    ax.tick_params(labelsize=14)
    fig.tight_layout(pad=0.3)
    suffix = f"{PREFIX}_cross_sections_epsilon_{epsilon:.2f}".replace(".", "p")
    fig.savefig(OUT / f"{suffix}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{suffix}.png", dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactive", action="store_true",
                        help="open the PyVista geometry for interactive viewing")
    args = parser.parse_args()
    beta = np.linspace(0, 2 * np.pi, 1001)
    limits = section_limits(beta)
    for epsilon in SECTION_EPSILONS:
        make_sections(epsilon, beta, limits)
    make_geometry(interactive=args.interactive)
    print(f"Figures written to {OUT}")
