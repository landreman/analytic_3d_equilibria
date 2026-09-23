#!/usr/bin/env python
"""Solve the analytic 3-D MHD equilibrium, or its separate vacuum problem, in DESC.

Field and poloidal convention: analytic_3d_equilibrium_positive_iota.tex.
Tested with desc-opt 0.17.3 and JAX 0.9.2, using 64-bit arithmetic.

Defaults use configuration A from
20260913-02_analytic_3D_equilibrium_sheared_iota_figures.py:
epsilon=1.08, S=3, k_b=0.70, axial lambda=3.5, kappa=1.
This smoother geometry permits substantially lower Fourier resolution than the
earlier epsilon=5, S=pi/2 configuration. Its finite-boundary vacuum transform
is a separate problem and is not inferred from the analytic MHD transform.

Run from the project root with the supplied environment:
    venv/bin/python sheared_iota/THIS_SCRIPT.py
    venv/bin/python sheared_iota/THIS_SCRIPT.py --vacuum
Use --help for resolution, tolerance, parameter, and output options.

Coordinates are in meters and B in tesla. Boundary pressure is zero. The paper
normalizes mu_0=1, so its pressure must be divided by mu_0. Its psi=k**2/2 is
NOT DESC's toroidal-flux coordinate: rho=sqrt(Psi(k)/Psi(k_b)). DESC's
poloidal theta=chi increases toward decreasing z, matching the paper's
positive-iota convention.

MHD: the analytic geometry and field initialize R, Z, and DESC's lambda
(unrelated to the axial wavenumber). Force balance is solved with fixed boundary,
pressure, toroidal flux, and CURRENT. Iota is free; its value and shear are
checked along with the Cartesian field, current density, pressure, force balance,
and volume-averaged beta on independent grids. Only the final resolution must pass.

Vacuum (--vacuum): keep that same boundary and toroidal flux, prescribe pressure
and enclosed-current profiles identically zero, and minimize all components of
J with CurrentDensity. The analytic MHD tests are skipped. Report the vacuum
iota profile and independently sampled J residual instead. Interior surfaces
and magnetic axis are allowed to move. Vacuum resolution steps use continuation.

The solve uses QuadratureGrid with full angular resolution on every radial
shell. A LinearGrid supplied with Gaussian radial nodes does not supply the
corresponding Gaussian weights. Independent diagnostic grids are LinearGrids.

Default resolution is L=M=10, N=20, with a 12 x 41 x 81 solve grid per field
period. The memory-efficient matrix-free backend minimizes DESC's unchanged
residual using exact JVP/VJP derivatives and eliminates the fixed-boundary
constraints algebraically. Its MHD gtol=1e-5 is specific to the scaled
coefficients; physical checks, including force balance, determine acceptance.
Vacuum and standard DESC backends default to gtol=1e-8. Use --optimizer lsq-exact
to select DESC's dense solver, with substantially greater memory requirements.

Outputs: <script-stem>.h5 and <script-stem>_checks.json; --vacuum appends
_vacuum to both stems. A failed MHD solve or physical check returns a nonzero
exit status, and its saved equilibrium is only a diagnostic approximation.
Solver status and all physical error tolerances are recorded in the JSON.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import leggauss
from scipy.constants import mu_0
from scipy.optimize import brentq, least_squares
from scipy.sparse.linalg import LinearOperator

import desc
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid, QuadratureGrid
from desc.profiles import PowerSeriesProfile
from desc.transform import Transform
from desc.objectives import CurrentDensity, ForceBalance, ObjectiveFunction

jax.config.update("jax_enable_x64", True)
EPS, S, KAPPA, KB = 1.08, 3.0, 1.0, 0.70
LAM = 3.5
NFP = 2
PSI_B = KB**2 / 2  # paper's flux label; NOT toroidal flux in Wb
P_AXIS = KAPPA**2 * PSI_B / (mu_0 * LAM**2)
# Independent physical error checks below determine whether the solve agrees
# with the analytic MHD solution.
SOLVER_TOLERANCES = {"ftol": 1e-8, "xtol": 1e-10, "gtol": 1e-8}
# This gradient is with respect to the algebraically scaled, eliminated
# coefficients below; its magnitude is not comparable to DESC's dense-solver
# gradient. Independent physical checks, including force balance, are decisive.
MHD_MATRIX_FREE_GTOL = 1e-5


def h(sigma):
    return np.sqrt(4 * sigma**2 + EPS**2)


def paper_beta_quadratures(n_xi=96, n_t=512):
    """Section 3.4 finite-radius volume averages for zero boundary pressure.

    Gauss integration in alpha, with xi=asin(k_b)*sin(alpha), removes the
    square-root endpoint behavior of U. The periodic t integral is trapezoidal.
    Pressure is converted from the paper's mu_0=1 units to pascals.
    """
    nodes, weights = leggauss(n_xi)
    d = np.arcsin(KB)
    alpha = (np.pi / 2) * nodes
    xi = (d * np.sin(alpha))[:, None]
    dxi_weights = (d * np.pi / 2 * np.cos(alpha) * weights)[:, None]
    t = (2 * np.pi * np.arange(n_t) / n_t)[None, :]
    nu = EPS / 2 * np.sin(2 * t)
    sin_xi = np.sin(xi)
    f = np.cosh(nu)**2 - sin_xi**2
    u = np.arcsin(np.sqrt(np.clip((KB**2 - sin_xi**2) / f, 0, 1)))
    d_u = u - np.sin(2 * u) / 2
    c0 = -sin_xi * np.cos(xi) / np.sqrt(f)
    c1 = np.cosh(nu) * np.sinh(nu) / np.sqrt(f)

    def integral(value):
        return (2 * np.pi / n_t) * np.sum(dxi_weights * value)

    i_u = integral(u)
    mean_psi = integral(sin_xi**2 * u + f * d_u / 2) / (2 * i_u)
    mean_p2 = integral(c0**2 * (2 * u - d_u) + c1**2 * d_u) / (2 * i_u)
    horizontal_b2_over_kappa2 = integral(f * u / h(S + xi)) / (2 * i_u)
    mean_pressure = KAPPA**2 * (PSI_B - mean_psi) / (mu_0 * LAM**2)
    mean_b2 = KAPPA**2 * (horizontal_b2_over_kappa2 + mean_p2 / LAM**2)
    return {
        "volume_m3": i_u / LAM,
        "mean_psi": mean_psi,
        "mean_P2": mean_p2,
        "horizontal_B2_over_kappa2": horizontal_b2_over_kappa2,
        "mean_pressure_Pa": mean_pressure,
        "mean_B2_T2": mean_b2,
        "beta_volume": 2 * mu_0 * mean_pressure / mean_b2,
    }


def flux_derivatives_over_k(k, n=512):
    """Return Q'(k)/k and A'(k)/k, regular including k=0.

    Q is toroidal flux through t=0; A is poloidal flux with positive chi.
    Integrating the closed magnetic flux two-form gives iota_DESC=A'/Q'.
    The poloidal cut chi=0 yields the second one-dimensional quadrature.
    """
    k = np.atleast_1d(k)[:, None]
    u = np.arange(n) * (2 * np.pi / n)
    c = np.sqrt(1 - k**2 * np.sin(u)**2)
    sigma = S + np.arcsin(k * np.cos(u) / c)
    q = KAPPA * np.pi / LAM * np.mean(1 / (h(sigma) * c), axis=1)
    v = EPS / 2 * np.sin(2 * u)
    sigma = S + np.arcsin(k / np.cosh(v))
    g = (h(sigma) + EPS * np.cos(2 * u)) / 2
    a = 2 * np.pi * KAPPA / LAM * np.mean(
        g / (h(sigma) * np.sqrt(np.cosh(v)**2 - k**2)), axis=1
    )
    return q, a


def toroidal_flux(k, n=24, nang=512):
    x, w = leggauss(n)
    radii = k * (x + 1) / 2
    return k / 2 * np.sum(w * radii * flux_derivatives_over_k(radii, nang)[0])


def analytic_iota(k):
    q, a = flux_derivatives_over_k(k)
    return a / q


def analytic_current(k, n=512):
    """Ampere integral of B.dr / mu_0 along theta=chi at t=0."""
    k = np.atleast_1d(k)[:, None]
    u = np.arange(n) * (2 * np.pi / n)
    p, y_trans = -k * np.cos(u), k * np.sin(u)
    c = np.sqrt(1 - y_trans**2)
    sigma = S - np.arcsin(p / c)
    return 2 * np.pi * KAPPA / mu_0 * np.mean(
        y_trans**2 * (1 - k**2) / (2 * h(sigma) * c**3)
        + p**2 / (LAM**2 * c), axis=1
    )


def exact_field(xyz, xp=np):
    """Positive-iota paper's Cartesian field and psi; xp=jnp permits AD."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    zeta, zbar = x + 1j * y, x - 1j * y
    kk = zbar * xp.sqrt(1 + EPS / zbar**2)  # principal square-root branch
    phase = zeta * kk + xp.pi / 2 - S
    w = 1j * KAPPA * xp.sin(phase) / (2 * kk)
    bxy = xp.exp(-1j * LAM * z) * w
    bz = KAPPA / LAM * xp.real(xp.exp(-1j * LAM * z) * xp.cos(phase))
    b = xp.stack((xp.real(bxy), xp.imag(bxy), bz), axis=-1)
    psi = (xp.sin(LAM * z)**2 + (LAM * bz / KAPPA)**2) / 2
    return b, psi


field_jacobian = jax.jit(jax.vmap(jax.jacfwd(lambda x: exact_field(x, jnp)[0])))
psi_gradient = jax.jit(jax.vmap(jax.grad(lambda x: exact_field(x, jnp)[1])))


def exact_derivatives(xyz):
    db = np.asarray(field_jacobian(jnp.asarray(xyz)))  # dB_i/dx_j
    curl = np.stack((db[:, 2, 1] - db[:, 1, 2],
                     db[:, 0, 2] - db[:, 2, 0],
                     db[:, 1, 0] - db[:, 0, 1]), axis=-1)
    gps = np.asarray(psi_gradient(jnp.asarray(xyz)))
    return db, curl / mu_0, gps


def make_profiles():
    """Fit even profiles, increasing their degree until independently resolved.

    Toroidal flux determines rho; k is not assumed proportional to rho.
    Endpoint constraints enforce axis regularity and exactly zero edge pressure.
    """
    flux = toroidal_flux(KB)

    def invert(y):
        return np.array([
            brentq(lambda s: toroidal_flux(s) / flux - yy,
                   0.0, KB, xtol=5e-16) for yy in y
        ])

    test_y = np.unique(np.concatenate((
        np.linspace(0.0, 1.0, 193),
        (1 - np.cos(np.linspace(0, np.pi, 129))) / 2,
    )))
    test_k = invert(test_y)
    test_current = analytic_current(test_k, n=1024)
    edge_current = analytic_current(KB, n=1024)[0]
    for degree in (6, 8, 10, 12, 16, 20, 24):
        fit_y = (1 - np.cos(np.linspace(0, np.pi, 4 * degree + 1))) / 2
        fit_k = invert(fit_y)
        k2 = Polynomial.fit(fit_y, fit_k**2, degree).convert()
        k2.coef[0] = 0.0
        k2.coef[1:] *= KB**2 / k2(1.0)
        current_poly = Polynomial.fit(
            fit_y, analytic_current(fit_k, n=1024), degree
        ).convert()
        current_poly.coef[0] = 0.0
        current_poly.coef[1:] *= edge_current / current_poly(1.0)
        fitted_k2 = k2(test_y)
        flux_error = max(abs(toroidal_flux(kk) / flux - yy)
                         for kk, yy in zip(np.sqrt(np.maximum(fitted_k2, 0)), test_y))
        current_error = np.max(np.abs(current_poly(test_y) - test_current)) / abs(edge_current)
        if flux_error < 2e-12 and current_error < 2e-12:
            break
    else:
        raise RuntimeError(
            f"Profile fits did not converge: flux={flux_error:.3e}, "
            f"current={current_error:.3e}; use a smaller boundary or higher fit degree"
        )
    pcoef = -KAPPA**2 / (2 * LAM**2 * mu_0) * k2.coef.copy()
    pcoef[0] = P_AXIS
    pressure = PowerSeriesProfile(pcoef, modes=2 * np.arange(len(pcoef)))
    current = PowerSeriesProfile(current_poly.coef,
                                 modes=2 * np.arange(len(current_poly.coef)))
    return flux, k2, pressure, current


def geometry(rho, theta, phi, k2):
    """Exact position/Jacobian, with safeguarded inversion at fixed azimuth.

    The original fixed-point iteration is only rapidly convergent for thin
    boundaries. Here sigma is bracketed by S +/- arcsin(k), an exact geometric
    bound, and bisection avoids any contraction assumption.
    """
    rho, theta, phi = np.broadcast_arrays(rho, theta, phi)
    pk = Polynomial(k2.coef[1:])
    pk_values = pk(rho**2)
    if np.any(pk_values <= 0):
        raise RuntimeError("The fitted flux map is not regular")
    k = rho * np.sqrt(pk_values)
    kr = np.sqrt(pk_values) + rho**2 * pk.deriv()(rho**2) / np.sqrt(pk_values)
    # Paper coordinates: P=-k*cos(chi), Y=k*sin(chi), z=-asin(Y)/lambda.
    p, y_trans = -k * np.cos(theta), k * np.sin(theta)
    c = np.sqrt(1 - y_trans**2)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)

    def at_sigma(sigma):
        hh = h(sigma)
        b = np.sqrt((hh + EPS) / 2)
        a = sigma / b  # a*b=sigma; avoids subtraction of nearly equal numbers
        t = np.arctan2(a * sin_phi, b * cos_phi)
        v = EPS / 2 * np.sin(2 * t)
        mapped_sigma = S + np.arctan(np.tanh(v) * y_trans / c) - np.arcsin(
            p / np.sqrt(np.cosh(v)**2 - y_trans**2))
        return sigma - mapped_sigma, t, a, b

    half_width = np.arcsin(k)
    lower = S - half_width
    upper = S + half_width
    if np.any(lower <= 0):
        raise RuntimeError("Require S > arcsin(k_b) for the analytic torus")
    for _ in range(52):
        sigma = (lower + upper) / 2
        residual, _, _, _ = at_sigma(sigma)
        lower = np.where(residual < 0, sigma, lower)
        upper = np.where(residual >= 0, sigma, upper)
    sigma = (lower + upper) / 2
    residual, t, a, b = at_sigma(sigma)
    if np.max(np.abs(residual)) > 1e-12:
        raise RuntimeError("Analytic coordinate inversion did not converge")
    radius = np.hypot(a * np.cos(t), b * np.sin(t))
    xyz = np.stack((radius * cos_phi, radius * sin_phi, -np.arcsin(y_trans) / LAM), axis=-1)
    v = EPS / 2 * np.sin(2 * t)
    hh = h(sigma)
    g = (hh + EPS * np.cos(2 * t)) / 2
    tt = np.sqrt(np.cosh(v)**2 - k**2)
    s = sigma - S
    sigma_t = EPS * np.cos(2 * t) / tt * (
        -np.sin(s) * np.sinh(v) * c + np.cos(s) * np.cosh(v) * y_trans)
    azimuth_denominator = sigma - v / hh * sigma_t
    if np.any(azimuth_denominator <= 0):
        raise RuntimeError("This poloidal coordinate folds at fixed physical azimuth")
    t_phi = radius**2 / azimuth_denominator
    jac = k * kr * g / (LAM * hh * tt * c) * t_phi
    return xyz, jac


def initialize(L, M, N, flux, k2, pressure, current, initial_guess=None):
    boundary_grid = LinearGrid(rho=1.0, M=2*M+3, N=2*N+4, NFP=NFP)
    xyz, _ = geometry(*boundary_grid.nodes.T, k2)
    coords = np.column_stack((np.hypot(xyz[:, 0], xyz[:, 1]), boundary_grid.nodes[:, 2], xyz[:, 2]))
    # Retain the tensor grid so fitting uses separable Fourier transforms.
    surface = FourierRZToroidalSurface(M=M, N=N, NFP=NFP, sym=True)
    surface.R_lmn = Transform(boundary_grid, surface.R_basis, build=False,
                              build_pinv=True).fit(coords[:, 0])
    surface.Z_lmn = Transform(boundary_grid, surface.Z_basis, build=False,
                              build_pinv=True).fit(coords[:, 2])
    eq = Equilibrium(L=L, M=M, N=N, NFP=NFP, sym=True, surface=surface,
                     pressure=pressure, current=current, Psi=flux,
                     L_grid=2*L, M_grid=2*M, N_grid=2*N, ensure_nested=False)
    if initial_guess is not None:
        eq.set_initial_guess(initial_guess, ensure_nested=False)
        eq.axis = eq.get_axis()
        return eq
    grid = LinearGrid(rho=np.linspace(0.04, 1, 2*L+2), M=2*M+3, N=2*N+4, NFP=NFP)
    nodes = np.asarray(grid.nodes)
    rho, theta, phi = nodes.T
    xyz, jac = geometry(rho, theta, phi, k2)
    radius = np.hypot(xyz[:, 0], xyz[:, 1])
    eq.set_initial_guess(grid, radius, xyz[:, 2], ensure_nested=False)
    eq.axis = eq.get_axis()  # keep the separate axis coefficients consistent
    b, _ = exact_field(xyz)
    bphi = (-b[:, 0] * np.sin(phi) + b[:, 1] * np.cos(phi)) / radius
    k = np.sqrt(k2(rho**2))
    # B^theta=+kappa*sqrt(1-Y^2), with Y=k*sin(theta).
    btheta = KAPPA * np.sqrt(1 - k**2 * np.sin(theta)**2)
    psi_r = flux * rho / np.pi  # derivative of DESC's toroidal flux / (2*pi)
    target_theta = bphi * jac / psi_r - 1
    target_phi = analytic_iota(k) - btheta * jac / psi_r
    # Integrate the two angular derivatives in Fourier space, then fit the
    # scalar lambda with DESC's separable transform. This is the same angular
    # least-squares projection without a huge dense derivative matrix.
    shape = (grid.num_theta, grid.num_rho, grid.num_zeta)
    ft = np.fft.fftn(target_theta.reshape(shape, order="F"), axes=(0, 2))
    fp = np.fft.fftn(target_phi.reshape(shape, order="F"), axes=(0, 2))
    mm = np.fft.fftfreq(grid.num_theta, 1/grid.num_theta)[:, None, None]
    nn = NFP*np.fft.fftfreq(grid.num_zeta, 1/grid.num_zeta)[None, None, :]
    denominator = mm**2 + nn**2
    coefficients = (-1j*mm*ft - 1j*nn*fp)/np.where(denominator, denominator, 1)
    lam_values = np.fft.ifftn(coefficients, axes=(0, 2)).real.ravel(order="F")
    eq.L_lmn = Transform(grid, eq.L_basis, build=False, build_pinv=True).fit(lam_values)
    return eq


def rms(x):
    return np.sqrt(np.mean(np.sum(np.asarray(x)**2, axis=-1)))


def cylindrical_to_cartesian(vector, phi):
    v = np.asarray(vector)
    return np.column_stack((v[:, 0]*np.cos(phi) - v[:, 1]*np.sin(phi),
                            v[:, 0]*np.sin(phi) + v[:, 1]*np.cos(phi), v[:, 2]))


def validate(eq, k2, beta_reference):
    """Independent of solve/fit nodes; includes interior, near-axis, and edge."""
    grid = LinearGrid(rho=np.array([0.035, 0.17, 0.37, 0.61, 0.83, 1.0]),
                      theta=(np.arange(2*eq.M_grid+9)+0.23)*2*np.pi/(2*eq.M_grid+9),
                      zeta=(np.arange(2*eq.N_grid+11)+0.37)*np.pi/(2*eq.N_grid+11), NFP=NFP)
    quantities = ["R", "Z", "B", "J", "grad(p)", "p", "sqrt(g)", "grad(B)"]
    data = eq.compute(quantities, grid=grid)
    phi = grid.nodes[:, 2]
    xyz = np.column_stack((data["R"]*np.cos(phi), data["R"]*np.sin(phi), data["Z"]))
    b = cylindrical_to_cartesian(data["B"], phi)
    j = cylindrical_to_cartesian(data["J"], phi)
    gp = cylindrical_to_cartesian(data["grad(p)"], phi)
    be, psi = exact_field(xyz)
    db, je, gps = exact_derivatives(xyz)
    gpe = -KAPPA**2 / (mu_0 * LAM**2) * gps
    b_error = np.linalg.norm(b-be, axis=1) / np.linalg.norm(be, axis=1)
    expected_psi = k2(grid.nodes[:, 0]**2) / 2
    checks = {
        "B_relative_rms": (rms(b-be) / rms(be), 1e-4),
        "B_relative_max": (np.max(b_error), 1e-3),
        "J_relative_rms": (rms(j-je) / rms(je), 1e-3),
        "pressure_gradient_relative_rms": (rms(gp-gpe) / rms(gpe), 2e-3),
        "force_relative_rms": (rms(np.cross(j, b)-gp) / rms(gpe), 1e-3),
        "force_relative_max": (np.max(np.linalg.norm(np.cross(j,b)-gp, axis=1)) / rms(gpe), 1e-2),
        "surface_psi_max_over_psi_b": (np.max(np.abs(psi-expected_psi)) / PSI_B, 1e-3),
        "pressure_max_over_axis_pressure": (np.max(np.abs(np.asarray(data["p"])-KAPPA**2*(PSI_B-psi)/(mu_0*LAM**2))) / P_AXIS, 1e-3),
        "B_dot_grad_analytic_psi_relative_rms": (np.sqrt(np.mean(np.sum(b*gps, axis=1)**2)) / (rms(be)*rms(gps)), 1e-4),
        # Divergence is a structural DESC identity; report it as a consistency check.
        "div_B_relative_rms": (np.sqrt(np.mean(np.trace(data["grad(B)"], axis1=1, axis2=2)**2)) / rms(db.reshape(-1, 9)), 1e-10),
        "nonpositive_jacobian_count": (np.count_nonzero(np.asarray(data["sqrt(g)"]) <= 0), 0),
        # Unrequested dependency entries may include NaNs for unset kinetic
        # profiles. Test the physical outputs used here, not those placeholders.
        "nonfinite_value_count": (sum(np.count_nonzero(~np.isfinite(np.asarray(data[key]))) for key in quantities), 0),
    }
    edge = grid.nodes[:, 0] == 1
    checks["boundary_psi_max_over_psi_b"] = (np.max(np.abs(psi[edge]-PSI_B)) / PSI_B, 1e-3)
    axis_grid = LinearGrid(rho=0.0, theta=0.0, N=2*eq.N+9, NFP=NFP)
    axis = eq.compute(["R", "Z"], grid=axis_grid)
    az = axis_grid.nodes[:, 2]
    aa, bb = np.sqrt((h(S)-EPS)/2), np.sqrt((h(S)+EPS)/2)
    expected_r = 1 / np.sqrt(np.cos(az)**2/aa**2 + np.sin(az)**2/bb**2)
    axis_error = np.hypot(np.asarray(axis["R"])-expected_r, np.asarray(axis["Z"]))
    checks["axis_distance_over_minor_radius"] = (np.max(axis_error) / (KB/LAM), 1e-3)
    radial_grid = LinearGrid(rho=np.linspace(0, 1, 13), M=eq.M_grid+4, N=eq.N_grid+8, NFP=NFP)
    io = np.asarray(radial_grid.compress(eq.compute("iota", grid=radial_grid)["iota"]))
    r = np.asarray(radial_grid.nodes[radial_grid.unique_rho_idx, 0])
    io_exact = analytic_iota(np.sqrt(np.maximum(k2(r**2), 0)))
    # At configuration A, 1e-4 absolute iota is under 2e-5 relative to iota~6.
    checks["iota_max_absolute"] = (np.max(np.abs(io-io_exact)), 1e-4)
    checks["iota_shear_relative"] = (abs((io[-1]-io[0])/(io_exact[-1]-io_exact[0])-1), 1e-3)
    # Integrate DESC's solved B and pressure over its own volume element.
    # QuadratureGrid supplies Gaussian radial weights, unlike LinearGrid.
    beta_grid = QuadratureGrid(L=max(eq.L+8, 22), M=eq.M+6,
                               N=eq.N+16, NFP=NFP)
    beta_data = eq.compute(["B", "p", "sqrt(g)", '<beta>_vol'], grid=beta_grid)
    volume_weights = np.asarray(beta_grid.weights) * np.asarray(beta_data["sqrt(g)"])
    volume = np.sum(volume_weights)
    field = np.asarray(beta_data["B"])
    mean_pressure = np.sum(volume_weights * np.asarray(beta_data["p"])) / volume
    mean_b2 = np.sum(volume_weights * np.sum(field**2, axis=1)) / volume
    # beta_numeric = 2 * mu_0 * mean_pressure / mean_b2
    beta_numeric = float(beta_data['<beta>_vol'])
    beta_desc = {
        "volume_m3": volume,
        "mean_psi": np.sum(volume_weights * k2(beta_grid.nodes[:, 0]**2) / 2) / volume,
        "mean_P2": LAM**2 * np.sum(volume_weights * field[:, 2]**2) / (KAPPA**2 * volume),
        "horizontal_B2_over_kappa2": np.sum(volume_weights * np.sum(field[:, :2]**2, axis=1)) / (KAPPA**2 * volume),
        "mean_pressure_Pa": mean_pressure,
        "mean_B2_T2": mean_b2,
        "beta_volume": beta_numeric,
    }
    for name, tolerance in (
        ("volume_m3", 1e-3),
        ("mean_psi", 1e-3),
        ("mean_P2", 1e-3),
        ("horizontal_B2_over_kappa2", 1e-3),
        ("mean_pressure_Pa", 1e-3),
        ("mean_B2_T2", 1e-3),
        ("beta_volume", 1e-3),
    ):
        checks[f"{name}_relative_to_paper"] = (
            abs(beta_desc[name] / beta_reference[name] - 1), tolerance
        )
    min_jac = float(np.min(data["sqrt(g)"]))
    return checks, {"rho": r.tolist(),
                    "iota_DESC": [float(v) if np.isfinite(v) else None for v in io],
                    "iota_analytic": io_exact.tolist(),
                    "volume_averages": {
                        "paper": beta_reference,
                        "DESC": beta_desc,
                        "grid": {"kind": "QuadratureGrid", "num_rho": beta_grid.num_rho,
                                 "M": beta_grid.M, "N": beta_grid.N, "NFP": NFP},
                    },
                    "min_jacobian": min_jac if np.isfinite(min_jac) else None}


def check_reference(flux, k2, pressure, current, beta_reference):
    """Verify reference calculus, quadrature convergence, and fitted profiles."""
    rng = np.random.default_rng(20260912)
    rho = rng.uniform(0.02, 1, 257)
    xyz, jac = geometry(rho, rng.uniform(0, 2*np.pi, 257), rng.uniform(0, 2*np.pi, 257), k2)
    b, psi = exact_field(xyz)
    db, j, gps = exact_derivatives(xyz)
    gp = -KAPPA**2/(mu_0*LAM**2)*gps
    # Recover the physical poloidal rate from P=lambda*Bz/kappa and
    # Y=-sin(lambda*z), independently of the flux-quadrature iota formula.
    p_trans = LAM * b[:, 2] / KAPPA
    y_trans = -np.sin(LAM * xyz[:, 2])
    b_dot_grad_p = LAM / KAPPA * np.sum(b * db[:, 2, :], axis=1)
    b_dot_grad_y = -LAM * np.cos(LAM * xyz[:, 2]) * b[:, 2]
    chi_rate = ((y_trans * b_dot_grad_p - p_trans * b_dot_grad_y)
                / (p_trans**2 + y_trans**2))
    expected_chi_rate = KAPPA * np.sqrt(1 - y_trans**2)
    # Ampere's law along two poloidal loops checks the prescribed-current sign.
    loop_rho = np.array([0.5, 1.0])
    ntheta = 512
    loop_theta = np.arange(ntheta) * 2 * np.pi / ntheta
    loop_xyz = geometry(loop_rho[:, None], loop_theta[None, :], 0.0, k2)[0]
    frequencies = np.fft.fftfreq(ntheta, 1 / ntheta)
    loop_tangent = np.fft.ifft(
        1j * frequencies[None, :, None] * np.fft.fft(loop_xyz, axis=1), axis=1
    ).real
    loop_current = 2 * np.pi / mu_0 * np.mean(
        np.sum(exact_field(loop_xyz)[0] * loop_tangent, axis=-1), axis=1
    )
    expected_current = analytic_current(np.sqrt(k2(loop_rho**2)), n=1024)
    axis_t = np.arange(1024) * 2 * np.pi / 1024
    axis_iota = h(S) * np.mean(1 / np.cosh(EPS / 2 * np.sin(2 * axis_t)))
    r = np.linspace(0, 1, 64)
    k = np.sqrt(np.maximum(k2(r**2), 0))
    g = LinearGrid(rho=r)
    expected_p = KAPPA**2 * (KB**2-k**2)/(2*mu_0*LAM**2)
    refined_beta = paper_beta_quadratures(n_xi=128, n_t=768)
    return {
        "reference_div_B_relative": (np.sqrt(np.mean(np.trace(db, axis1=1, axis2=2)**2))/rms(db.reshape(-1,9)), 1e-12),
        "reference_force_relative": (rms(np.cross(j,b)-gp)/rms(gp), 1e-11),
        "reference_B_dot_grad_psi_relative": (np.sqrt(np.mean(np.sum(b*gps,axis=1)**2))/(rms(b)*rms(gps)), 1e-12),
        "reference_positive_chi_rate_relative": (np.max(np.abs(chi_rate/expected_chi_rate-1)), 1e-10),
        "reference_current_line_integral_relative": (np.max(np.abs(loop_current/expected_current-1)), 1e-9),
        "reference_map_psi_relative": (np.max(np.abs(psi-k2(rho**2)/2))/PSI_B, 1e-11),
        "reference_nonpositive_jacobian_count": (np.count_nonzero(jac<=0), 0),
        "flux_quadrature_relative": (abs(toroidal_flux(KB, 40, 1024)/flux-1), 1e-12),
        "flux_radius_fit_absolute": (max(abs(toroidal_flux(kk)/flux-rr**2) for rr,kk in zip(r,k)), 1e-11),
        "pressure_profile_relative": (np.max(np.abs(np.asarray(pressure.compute(g))-expected_p))/P_AXIS, 1e-11),
        "current_profile_relative": (np.max(np.abs(np.asarray(current.compute(g))-analytic_current(k,1024)))/abs(analytic_current(KB)[0]), 1e-11),
        "iota_axis_formula_absolute": (abs(analytic_iota(0)[0] - axis_iota), 1e-12),
        "iota_quadrature_absolute": (np.max(np.abs(analytic_iota(k) -
            flux_derivatives_over_k(k, 1024)[1] / flux_derivatives_over_k(k, 1024)[0])), 1e-12),
        "beta_quadrature_convergence_relative": (
            max(abs(beta_reference[key] / refined_beta[key] - 1)
                for key in beta_reference), 1e-10),
    }


def record_checks(checks):
    results = {}
    for name, (value, tolerance) in checks.items():
        value = float(value)
        passed = bool(np.isfinite(value) and value <= tolerance)
        results[name] = {"value": value if np.isfinite(value) else None, "tolerance": tolerance, "passed": passed}
        print(f"{'PASS' if passed else 'FAIL'}  {name:47s} {value:11.4e}  <= {tolerance:.2e}", flush=True)
    return results


def json_safe(value):
    """Keep failed numerical diagnostics writable under strict JSON semantics."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def vacuum_diagnostics(eq):
    """Report the separate vacuum solution; never compare it to the MHD field.

    Zero enclosed current alone is not a pointwise curl-free check. Report the
    independently sampled current density normalized by B/(mu_0*a) as well.
    """
    grid = LinearGrid(rho=np.array([0.035, 0.17, 0.37, 0.61, 0.83, 1.0]),
                      theta=(np.arange(4*eq.M+9)+0.23)*2*np.pi/(4*eq.M+9),
                      zeta=(np.arange(4*eq.N+11)+0.37)*np.pi/(4*eq.N+11), NFP=NFP)
    quantities = ["B", "J", "sqrt(g)", "p", "current"]
    data = eq.compute(quantities, grid=grid)
    shape = eq.surface.compute(["R0/a", "R0", "a", "V"])
    a = float(shape["a"])
    radial = LinearGrid(rho=np.linspace(0, 1, 13), M=eq.M_grid+4, N=eq.N_grid+8, NFP=NFP)
    iota = np.asarray(radial.compress(eq.compute("iota", grid=radial)["iota"]))
    return {"aspect_ratio": float(shape["R0/a"]),
            "major_radius_m": float(shape["R0"]), "minor_radius_m": a,
            "volume_m3": float(shape["V"]),
            "rho": radial.nodes[radial.unique_rho_idx, 0].tolist(),
            "iota_DESC": iota.tolist(), "boundary_iota": float(iota[-1]),
            "mu0_a_J_rms_over_B_rms": float(mu_0*a*rms(data["J"])/rms(data["B"])),
            "mu0_a_J_max_over_B_rms": float(mu_0*a*np.max(np.linalg.norm(data["J"], axis=-1))/rms(data["B"])),
            "max_abs_pressure_Pa": float(np.max(np.abs(data["p"]))),
            "max_abs_current_A": float(np.max(np.abs(data["current"]))),
            "min_jacobian": float(np.min(data["sqrt(g)"])),
            "nonfinite_value_count": int(sum(np.count_nonzero(~np.isfinite(np.asarray(v)))
                                             for v in [*(data[key] for key in quantities), iota]))}



class FixedBoundaryCoordinates:
    """O(number of coefficients) exact fixed-boundary coordinates.

    At rho=1 every Zernike radial polynomial equals one. For each Fourier
    (m,n), eliminate its lowest-l coefficient using the fixed boundary sum.
    The remaining R/Z coefficients and every symmetric lambda coefficient
    are free. Derived axis coefficients are reconstructed too.
    """

    def __init__(self, eq, precondition=True):
        if not eq.sym or eq.bdry_mode != "lcfs" or eq.surface.rho != 1:
            raise ValueError("Require a stellarator-symmetric rho=1 LCFS equilibrium")
        if np.any(np.all(np.asarray(eq.L_basis.modes)[:, 1:] == 0, axis=1)):
            raise ValueError("Lambda basis unexpectedly contains a gauge mode")
        self.base = {name: jnp.asarray(value) for name, value in eq.params_dict.items()}
        self.groups = {}
        self.axes = {}
        self.offsets = [0]
        self.indices = {}
        scales = []
        values = []
        # Geometrical scales from coefficients, without constructing another
        # compute graph or quadrature grid merely for preconditioning.
        rmode = np.asarray(eq.surface.R_basis.modes)
        zmode = np.asarray(eq.surface.Z_basis.modes)
        rcoef = np.asarray(eq.surface.R_lmn)
        zcoef = np.asarray(eq.surface.Z_lmn)
        r00 = np.all(rmode[:, 1:] == 0, axis=1)
        major = max(float(np.max(np.abs(rcoef[r00]))), 1e-6)
        minor = max(float(np.sqrt(np.sum(rcoef[rmode[:, 1] != 0]**2)
                                  + np.sum(zcoef[zmode[:, 1] != 0]**2))), 1e-6)
        self.characteristic_lengths = {"major_m": major, "minor_m": minor}

        for name, basis, surface_basis, boundary, axis_basis in (
            ("R_lmn", eq.R_basis, eq.surface.R_basis, eq.Rb_lmn, eq.axis.R_basis),
            ("Z_lmn", eq.Z_basis, eq.surface.Z_basis, eq.Zb_lmn, eq.axis.Z_basis),
        ):
            modes = np.asarray(basis.modes)
            surface_modes = np.asarray(surface_basis.modes)
            lookup = {tuple(mode[1:]): j for j, mode in enumerate(surface_modes)}
            group = np.array([lookup[tuple(mode[1:])] for mode in modes], dtype=int)
            pivots = []
            for j in range(len(surface_modes)):
                members = np.flatnonzero(group == j)
                if not len(members):
                    raise ValueError("Boundary contains an unavailable interior mode")
                pivots.append(members[np.argmin(modes[members, 0])])
            pivots = np.asarray(pivots, dtype=int)
            free = np.setdiff1d(np.arange(len(modes)), pivots)
            self.groups[name] = (jnp.asarray(free), jnp.asarray(group[free]),
                                 jnp.asarray(pivots), jnp.asarray(boundary), len(modes))
            self.indices[name] = free
            self.offsets.append(self.offsets[-1] + len(free))
            values.append(np.asarray(getattr(eq, name))[free])

            axis_idx = np.flatnonzero(modes[:, 1] == 0)
            axis_lookup = {int(mode[2]): j for j, mode in enumerate(axis_basis.modes)}
            axis_group = np.array([axis_lookup[int(modes[i, 2])] for i in axis_idx], dtype=int)
            axis_sign = (-1.0)**(modes[axis_idx, 0] // 2)
            self.axes[name] = (jnp.asarray(axis_idx), jnp.asarray(axis_group),
                               jnp.asarray(axis_sign), axis_basis.num_modes)
            l, m, n = modes[free].T
            # Algebraic scaling balances the two spatial derivatives in J/F.
            # Every scale stays positive, so the feasible set is unchanged.
            spectral = 1 + l*(l+1) + m*m + (n*eq.NFP*minor/major)**2
            scales.append(minor/spectral if precondition else np.ones(len(free)))

        modes = np.asarray(eq.L_basis.modes)
        self.offsets.append(self.offsets[-1] + len(modes))
        self.indices["L_lmn"] = np.arange(len(modes))
        values.append(np.asarray(eq.L_lmn))
        l, m, n = modes.T
        spectral = 1 + l*(l+1) + m*m + (n*eq.NFP*minor/major)**2
        scales.append(1/spectral if precondition else np.ones(len(modes)))
        self.scale = jnp.asarray(np.concatenate(scales))
        self.origin = jnp.asarray(np.concatenate(values))
        # SciPy sets the initial trust radius to ||x0/x_scale||. A default DESC
        # guess can have all free radial and lambda coefficients essentially
        # zero, which would otherwise give a roundoff-sized trust region and
        # false ftol convergence after a negligible step. An affine unit
        # starting point gives a useful radius without removing any freedom.
        self.y0 = np.ones(len(self.origin))
        self.dim = self.y0.size

    def recover(self, scaled):
        """Return a full params_dict satisfying all standard linear constraints."""
        values = self.origin + (scaled - 1) * self.scale
        params = dict(self.base)
        for block, name in enumerate(("R_lmn", "Z_lmn")):
            free, group, pivot, boundary, size = self.groups[name]
            coeff = values[self.offsets[block]:self.offsets[block+1]]
            sums = jnp.zeros(boundary.size).at[group].add(coeff)
            full = jnp.zeros(size).at[free].set(coeff).at[pivot].set(boundary-sums)
            params[name] = full
            idx, axis_group, sign, axis_size = self.axes[name]
            params["Ra_n" if name == "R_lmn" else "Za_n"] = (
                jnp.zeros(axis_size).at[axis_group].add(full[idx]*sign)
            )
        params["L_lmn"] = values[self.offsets[2]:self.offsets[3]]
        return params

    def apply(self, eq, scaled):
        params = self.recover(jnp.asarray(scaled))
        eq.R_lmn = params["R_lmn"]
        eq.Z_lmn = params["Z_lmn"]
        eq.L_lmn = params["L_lmn"]
        eq.axis = eq.get_axis()
        return eq


def make_matrix_free_problem(eq, grid, *, vacuum=False, precondition=True, verbose=1):
    """Build a residual and exact JVP/VJP operator without dense matrices."""
    if eq.iota is not None or eq.current is None:
        raise ValueError("Require prescribed current and unconstrained iota")
    if vacuum and (np.any(np.asarray(eq.p_l)) or np.any(np.asarray(eq.c_l))):
        raise ValueError("Vacuum solve requires identically zero pressure and current")
    coordinates = FixedBoundaryCoordinates(eq, precondition=precondition)
    objective = (CurrentDensity if vacuum else ForceBalance)(eq, grid=grid)
    objective.build(verbose=verbose)

    @jax.jit
    def residual(y):
        return objective.compute_scaled_error(coordinates.recover(y))

    @jax.jit
    def jvp(y, direction):
        return jax.jvp(residual, (y,), (direction,))[1]

    @jax.jit
    def vjp(y, cotangent):
        return jax.vjp(residual, y)[1](cotangent)[0]

    counts = {"residual": 0, "jvp": 0, "vjp": 0, "jacobian": 0}

    def fun(y):
        counts["residual"] += 1
        return np.asarray(residual(jnp.asarray(y)), dtype=float)

    def jac(y):
        counts["jacobian"] += 1
        state = jnp.asarray(y)

        def matvec(v):
            counts["jvp"] += 1
            return np.asarray(jvp(state, jnp.asarray(np.asarray(v).reshape(-1))), dtype=float)

        def rmatvec(w):
            counts["vjp"] += 1
            return np.asarray(vjp(state, jnp.asarray(np.asarray(w).reshape(-1))), dtype=float)

        return LinearOperator((objective.dim_f, coordinates.dim), matvec=matvec,
                              rmatvec=rmatvec, dtype=np.float64)

    return {"coordinates": coordinates, "objective": objective, "fun": fun,
            "jac": jac, "jvp": jvp, "vjp": vjp, "counts": counts}


def solve_fixed_boundary_matrix_free(eq, grid, *, vacuum=False, maxiter=30,
                                     ftol=1e-8, xtol=1e-10, gtol=1e-8,
                                     lsmr_maxiter=80, lsmr_tol=1e-5,
                                     precondition=True, verbose=2):
    """Solve DESC residuals via SciPy TRF/LSMR, returning (eq, result dict).

    maxiter limits residual evaluations, matching SciPy's max_nfev. nit is the
    number of accepted iterates after the initial one (njev-1). The result also
    reports nfev/njev explicitly. No dense Jacobian or nullspace is allocated.
    """
    started = time.perf_counter()
    problem = make_matrix_free_problem(eq, grid, vacuum=vacuum,
                                       precondition=precondition, verbose=verbose)
    coordinates = problem["coordinates"]
    result = least_squares(
        problem["fun"], coordinates.y0, jac=problem["jac"], method="trf",
        tr_solver="lsmr", x_scale=1.0, ftol=ftol, xtol=xtol, gtol=gtol,
        max_nfev=maxiter, verbose=min(verbose, 2),
        tr_options={"maxiter": lsmr_maxiter, "atol": lsmr_tol,
                    "btol": lsmr_tol, "regularize": True},
    )
    coordinates.apply(eq, result.x)
    metadata = {"success": bool(result.success), "message": str(result.message),
                "nit": max(int(result.njev)-1, 0), "nfev": int(result.nfev),
                "njev": int(result.njev), "cost": float(result.cost),
                "optimality": float(result.optimality), "status": int(result.status),
                "solver": "scipy-trf-lsmr-jvp-vjp", "reduced_dimension": coordinates.dim,
                "residual_dimension": problem["objective"].dim_f,
                "lsmr_maxiter": lsmr_maxiter, "lsmr_tol": lsmr_tol,
                "operator_evaluations": problem["counts"],
                "elapsed_seconds": time.perf_counter()-started}
    return eq, metadata


def main():
    global EPS, S, KAPPA, KB, LAM, PSI_B, P_AXIS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--L", type=int, default=10)
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[20], metavar="N")
    parser.add_argument("--maxiter", type=int, default=30,
                        help="Maximum iterations (residual evaluations for matrix-free).")
    parser.add_argument("--optimizer", default="matrix-free",
                        help="matrix-free TRF/LSMR (default), or a DESC optimizer such as lsq-exact.")
    parser.add_argument("--lsmr-maxiter", type=int, default=80)
    parser.add_argument("--lsmr-tol", type=float, default=1e-5)
    parser.add_argument("--jac-chunk-size", type=int, default=16,
                        help="AD batch size for a standard DESC optimizer; matrix-free uses JVP/VJP instead.")
    parser.add_argument("--epsilon", type=float, default=EPS)
    parser.add_argument("--S", type=float, default=S)
    parser.add_argument("--k-b", type=float, default=KB)
    parser.add_argument("--kappa", type=float, default=KAPPA)
    parser.add_argument("--lam", type=float, default=LAM,
                        help="Axial wavenumber (the paper's lambda, not DESC's lambda).")
    parser.add_argument("--vacuum", action="store_true",
                        help="Solve J=0 with p=0 and prescribed current=0 in the same boundary; skip analytic tests.")
    parser.add_argument("--initial-guess", type=Path,
                        help="Use a saved DESC guess for each MHD resolution or the first vacuum step; boundary/profiles stay prescribed.")
    for name, value in SOLVER_TOLERANCES.items():
        parser.add_argument("--" + name, type=float, default=None if name == "gtol" else value,
                            help=("Default: 1e-5 for matrix-free MHD, 1e-8 otherwise."
                                  if name == "gtol" else f"Default: {value:g}."))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.L < args.M or args.M < 2 or min(args.resolutions) < 2 or args.maxiter < 1 or args.jac_chunk_size < 1 or args.lsmr_maxiter < 1:
        parser.error("Require L >= M >= 2, N >= 2, and maxiter >= 1")
    EPS, S, KAPPA, KB = args.epsilon, args.S, args.kappa, args.k_b
    LAM = args.lam
    if not (np.all(np.isfinite([EPS, S, KAPPA, KB, LAM])) and EPS > 0 and
            KAPPA > 0 and LAM > 0 and 0 < KB < 1 and S > np.arcsin(KB)):
        parser.error("Require finite epsilon,kappa,lambda > 0, 0 < k_b < 1, S > arcsin(k_b)")
    tolerances = {name: getattr(args, name) for name in SOLVER_TOLERANCES}
    if tolerances["gtol"] is None:
        tolerances["gtol"] = (MHD_MATRIX_FREE_GTOL
                              if args.optimizer == "matrix-free" and not args.vacuum
                              else SOLVER_TOLERANCES["gtol"])
    if not all(np.isfinite(v) and v > 0 for v in [*tolerances.values(), args.lsmr_tol]):
        parser.error("Solver tolerances must be finite and positive")
    PSI_B = KB**2/2
    P_AXIS = KAPPA**2 * PSI_B/(mu_0*LAM**2)
    start = time.perf_counter()
    flux, k2, pressure, current = make_profiles()
    print(f"DESC {desc.__version__}; JAX {jax.__version__}; devices {jax.devices()}", flush=True)
    print(f"epsilon={EPS}, S={S}, k_b={KB}, lambda={LAM:.12g}, kappa={KAPPA}; Psi={flux:.16g} Wb", flush=True)
    if args.vacuum:
        pressure, current = PowerSeriesProfile([0.0]), PowerSeriesProfile([0.0])
        print("Vacuum solve: pressure=0, prescribed current=0, objective J=0. Analytic tests skipped.", flush=True)
        reference = {}
    else:
        print(f"Axis p={P_AXIS:.12g} Pa; edge current={analytic_current(KB)[0]:.12g} A", flush=True)
        print(f"Expected iota axis/edge: {analytic_iota([0,KB])}; current is prescribed, iota is solved.", flush=True)
        beta_reference = paper_beta_quadratures()
        print(f"Expected volume-averaged beta: {100*beta_reference['beta_volume']:.6f}%", flush=True)
        reference = record_checks(check_reference(flux, k2, pressure, current, beta_reference))
    if not all(item["passed"] for item in reference.values()):
        raise RuntimeError("Analytic reference checks failed before the DESC solve")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / (Path(__file__).stem + ("_vacuum" if args.vacuum else ""))
    history = []
    for step, n in enumerate(args.resolutions):
        print(f"\nSolving L={args.L}, M={args.M}, N={n}, NFP={NFP}", flush=True)
        if step < len(args.resolutions)-1:
            print("Coarse-resolution diagnostics; acceptance is checked at the final resolution.", flush=True)
        previous = eq if args.vacuum and step else (
            str(args.initial_guess) if args.initial_guess is not None else None)
        eq = initialize(args.L, args.M, n, flux, k2, pressure, current, previous)
        del previous
        # Correct Gaussian weights AND full angular resolution on every shell.
        # LinearGrid(rho=Gauss_nodes) assigns midpoint-spacing weights, not
        # Gaussian weights; ConcentricGrid thins the inner poloidal meshes.
        solve_grid = QuadratureGrid(L=2*args.L+2, M=eq.M_grid, N=eq.N_grid, NFP=NFP)
        if args.optimizer == "matrix-free":
            eq, result = solve_fixed_boundary_matrix_free(
                eq, solve_grid, vacuum=args.vacuum, maxiter=args.maxiter,
                lsmr_maxiter=args.lsmr_maxiter, lsmr_tol=args.lsmr_tol,
                **tolerances, verbose=2)
        else:
            objective_type = CurrentDensity if args.vacuum else ForceBalance
            objective = ObjectiveFunction(objective_type(eq, grid=solve_grid),
                                          jac_chunk_size=args.jac_chunk_size)
            eq, result = eq.solve(objective=objective, maxiter=args.maxiter,
                                  optimizer=args.optimizer, **tolerances, verbose=2)
        # Preserve the computed equilibrium even if a later diagnostic fails.
        eq.save(str(prefix) + ".h5")
        if args.vacuum:
            evaluated, profiles = {}, vacuum_diagnostics(eq)
            print("Vacuum diagnostics (analytic tests skipped):", json.dumps(json_safe(profiles), indent=2), flush=True)
        else:
            print("Independent validation:", flush=True)
            checks, profiles = validate(eq, k2, beta_reference)
            evaluated = record_checks(checks)
            profiles["aspect_ratio"] = float(eq.surface.compute("R0/a")["R0/a"])
            print(f"Volume-averaged beta: DESC {100*profiles['volume_averages']['DESC']['beta_volume']:.6f}%; "
                  f"paper {100*beta_reference['beta_volume']:.6f}%", flush=True)
        history.append({"L": args.L, "M": args.M, "N": n,
                        "solver_success": bool(result["success"]), "solver_message": str(result["message"]),
                        "iterations": int(result["nit"]),
                        "optimizer": args.optimizer, "jac_chunk_size": args.jac_chunk_size,
                        "solver_details": result if args.optimizer == "matrix-free" else None,
                        "solver_tolerances": {**tolerances, "maxiter": args.maxiter},
                        "solve_grid": {"kind": "QuadratureGrid", "L": 2*args.L+2, "num_rho": solve_grid.num_rho,
                                       "M": eq.M_grid, "N": eq.N_grid, "NFP": NFP},
                        "checks": evaluated, "profiles": profiles})
        # MHD resolutions are independent, each using the analytic initializer
        # or the requested saved guess. Vacuum uses continuation instead.
        if step != len(args.resolutions)-1:
            if not args.vacuum:
                del eq
            jax.clear_caches()
    eq.save(str(prefix) + ".h5")
    passed = None if args.vacuum else (history[-1]["solver_success"] and
                                       all(item["passed"] for item in history[-1]["checks"].values()))
    report = {"passed": passed, "mode": "vacuum" if args.vacuum else "MHD",
              "analytic_tests_skipped": args.vacuum, "python_executable": sys.executable,
              "initial_guess": str(args.initial_guess) if args.initial_guess is not None else None,
              "desc_version": desc.__version__, "jax_version": jax.__version__,
              "parameters": {"epsilon": EPS, "S": S, "kappa": KAPPA, "lambda": LAM, "k_b": KB,
                             "p_boundary_Pa": 0, "Psi_Wb": flux, "mu_0": mu_0},
              "reference_checks": reference, "resolution_history": history,
              "elapsed_seconds": time.perf_counter()-start}
    Path(str(prefix) + "_checks.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    print(f"\nSaved {prefix}.h5\nSaved {prefix}_checks.json", flush=True)
    status = "SKIPPED (vacuum)" if args.vacuum else ("PASS" if passed else "FAIL")
    print(f"Elapsed: {report['elapsed_seconds']:.1f} s; final analytic checks {status}", flush=True)
    if passed is False:
        raise SystemExit("MHD solve or numerical validation failed; inspect the JSON and increase resolution, tighten solver tolerances, or allow more iterations as indicated.")
    if args.vacuum and (not history[-1]["solver_success"] or
                       profiles["nonfinite_value_count"] or profiles["min_jacobian"] <= 0):
        raise SystemExit("Vacuum solve did not converge to a regular finite solution; inspect its diagnostics.")


if __name__ == "__main__":
    main()
