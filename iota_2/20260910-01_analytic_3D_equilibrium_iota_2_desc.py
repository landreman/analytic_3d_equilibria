#!/usr/bin/env python3
"""Solve and verify the exact 3-D MHD equilibrium in the accompanying note.

Install: python -m pip install 'desc-opt==0.17.3'
Run:     python 20260910-01_analytic_3D_equilibrium_desc.py
Refine:  python 20260910-01_analytic_3D_equilibrium_desc.py --resolution 10

This is a fixed-boundary DESC solve, initialized by a spectral fit to the
analytic flux surfaces. R, Z, and lambda are subsequently adjusted by DESC's
force-balance solver. The comparisons use the *solved* field, not the initial
fit. No external coils or free-boundary/vacuum matching are implied.

Source: "20260910-01 Astra analytic nonaxisymmetric MHD solution.md", Eqs. 1-37.
DESC API: https://desc-docs.readthedocs.io/en/stable/

Conventions (essential):
  * Interpret the note's position values in meters and B values in tesla.
    DESC uses J x B = grad(p), J = curl(B)/mu_0, so p_DESC = p_note/mu_0.
  * The note's psi is a surface label, NOT DESC's toroidal magnetic flux.
    Its constant coordinate Jacobian gives Psi_boundary = pi*a*b*delta Wb.
    Therefore rho = sqrt(psi_note/delta).
  * Let beta = arg(u+epsilon/2 + i*v). Choose zeta = physical phi and
    theta = 2*phi-beta. DESC's theta increases clockwise in an R,Z section;
    this choice is right handed, has iota_DESC=+2, and lambda=0 analytically.
    The note's counterclockwise geometric transform is -2. All field lines
    close after one full toroidal turn. There are two field periods (NFP=2).

Checks include independent complex-step analytic derivatives, physical B and
J at identical Cartesian points, pressure *variation* and its gradient,
force balance, divergence, convection, flux surfaces, boundary, axis, toroidal
flux, volume, energies, and numerical field-line tracing. Comparisons after
solving are coordinate invariant: DESC may change its poloidal-angle gauge.
Fixed profile/iota and closure checks alone are not evidence of field accuracy.

The volume averages in "20260910-02_Astra_beta_for_analytic_3d_equilibria_with_iota_2.md"
are tested against <p_note> = pa-delta, <|B|^2> = 1-epsilon**2/2+delta,
and beta_V = 2*<p_note>/<|B|^2>. Beta is a dimensionless ratio of volume
averages, not the volume average of local beta. Set --pa to 2*delta for
zero boundary pressure (e.g. --pa 0.03125 at the default delta).

The script writes an equilibrium .h5 and a machine-readable checks .json.
It exits nonzero if the optimizer or any accuracy check fails. Sampled checks
are numerical evidence, not a proof throughout the continuum. Increase the
resolution for stricter accuracy or more strongly deformed parameter choices.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.constants import mu_0
from scipy.integrate import solve_ivp

import desc
from desc.backend import jit, jnp
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import Grid, LinearGrid, QuadratureGrid
from desc.profiles import PowerSeriesProfile


def field(xyz, epsilon):
    """Note Eq. (1); preserve complex dtype for complex-step differentiation."""
    x, y, z = np.asarray(xyz).T
    a, b = np.sqrt(1 + epsilon), np.sqrt(1 - epsilon)
    s = (x / a) ** 2 + (y / b) ** 2
    F = np.sqrt(2 * s - s * s - 4 * z * z)
    return np.column_stack(((2*z*x - a/b*F*y)/s,
                            (2*z*y + b/a*F*x)/s, 1-s))


def flux_label(xyz, epsilon):
    """Note Eq. (2), using B*B (not abs(B)**2) for complex analyticity."""
    x, y, z = np.asarray(xyz).T
    B = field(xyz, epsilon)
    return (x*x + y*y + 4*z*z + np.sum(B*B, axis=1) - 2 + epsilon**2)/4


def labels(xyz, epsilon):
    """Independent, single-valued Cartesian field-line labels, note Eq. (18)."""
    a, b = np.sqrt(1 + epsilon), np.sqrt(1 - epsilon)
    X, Y = np.asarray(xyz)[:, 0]/a, np.asarray(xyz)[:, 1]/b
    B = field(xyz, epsilon)
    return np.column_stack(((X*X + (B[:, 0]/a)**2 - 1)/2,
                            (X*Y + B[:, 0]*B[:, 1]/(a*b))/2))


def section(u, v, phi, epsilon):
    """Exact cylindrical (R,phi,Z) at fixed labels and *physical* phi.

    Invert the xy ellipse's 2x2 matrix to find cos(t),sin(t); no angle
    unwrapping or nonlinear root finder is required.
    """
    a, b = np.sqrt(1 + epsilon), np.sqrt(1 - epsilon)
    ell = np.sqrt((1 + np.sqrt(1 - 4*(u*u + v*v)))/2)
    A, C = a*(ell + u/ell), a*v/ell
    D, E = b*v/ell, b*(ell - u/ell)
    ct, st = E*np.cos(phi) - C*np.sin(phi), A*np.sin(phi) - D*np.cos(phi)
    h = np.hypot(ct, st)
    ct, st = ct/h, st/h
    R = np.hypot(A*ct + C*st, D*ct + E*st)
    Z = v*(ct*ct - st*st) - 2*u*st*ct
    return np.column_stack((R, phi, Z))


def geometry(nodes, epsilon, delta):
    rho, theta, phi = np.asarray(nodes).T
    beta = 2*phi - theta
    u = -epsilon/2 + np.sqrt(delta)*rho*np.cos(beta)
    v = np.sqrt(delta)*rho*np.sin(beta)
    return section(u, v, phi, epsilon)


def xyz_from_rpz(rpz):
    R, phi, Z = np.asarray(rpz).T
    return np.column_stack((R*np.cos(phi), R*np.sin(phi), Z))


def vector_to_xyz(vector, phi):
    """DESC vectors are physical orthonormal cylindrical components."""
    R, P, Z = np.asarray(vector).T
    return np.column_stack((R*np.cos(phi)-P*np.sin(phi),
                            R*np.sin(phi)+P*np.cos(phi), Z))


def analytic_derivatives(xyz, epsilon):
    """Independent Cartesian complex-step derivatives of analytic B and psi."""
    n, h = len(xyz), 1e-30
    dB, gradpsi = np.empty((n, 3, 3)), np.empty((n, 3))
    for k in range(3):
        shifted = np.asarray(xyz, dtype=complex).copy()
        shifted[:, k] += 1j*h
        dB[:, :, k] = field(shifted, epsilon).imag/h  # component, derivative
        gradpsi[:, k] = flux_label(shifted, epsilon).imag/h
    curl = np.column_stack((dB[:, 2, 1]-dB[:, 1, 2],
                            dB[:, 0, 2]-dB[:, 2, 0],
                            dB[:, 1, 0]-dB[:, 0, 1]))
    return dB, curl, gradpsi


def compute(eq, names, nodes):
    """Keep arbitrary node order; no surface averages on scattered grids."""
    grid = Grid(nodes, sort=False)
    # Supply the actual prescribed profile and derivatives. Otherwise DESC's
    # dependency graph also evaluates unused current-based iota integrals,
    # which require quadrature spacing unavailable on a scattered grid.
    profiles = {key: eq.iota.compute(grid, dr=k)
                for k, key in enumerate(['iota', 'iota_r', 'iota_rr'])}
    data = eq.compute(names, grid=grid, data=profiles, override_grid=False)
    return {key: np.asarray(data[key]) for key in names}


class Checks:
    """Collect failures so every diagnostic runs before the final exit status."""
    def __init__(self):
        self.results = []

    def error(self, name, values, limit):
        values = np.asarray(values, dtype=float)
        if values.ndim > 1:
            values = np.linalg.norm(values, axis=-1)
        finite = bool(np.all(np.isfinite(values)))
        maximum = float(np.max(np.abs(values))) if finite else None
        rms = float(np.sqrt(np.mean(values**2))) if finite else None
        passed = finite and maximum <= limit
        self.results.append(dict(name=name, max=maximum, rms=rms,
                                 limit=limit, passed=passed))
        measured = f"{maximum:.3e}" if finite else "nonfinite"
        print(f"{'PASS' if passed else 'FAIL'}  {name:49s} {measured:>11s} <= {limit:.1e}",
              flush=True)

    def condition(self, name, passed):
        self.error(name, 0.0 if passed else 1.0, 0.0)

    @property
    def passed(self):
        return all(row['passed'] for row in self.results)


def build_equilibrium(args):
    """Fit the independent boundary and a volume initial guess; then solve."""
    m, n, ell = args.resolution, args.resolution, args.resolution + 2
    boundary_grid = LinearGrid(rho=[1.0], M=2*m, N=2*n, NFP=2, sym=False)
    rpz = geometry(boundary_grid.nodes, args.epsilon, args.delta)
    surface = FourierRZToroidalSurface.from_values(
        rpz, np.asarray(boundary_grid.nodes)[:, 1], M=m, N=n, NFP=2, sym=True)
    eq = Equilibrium(
        L=ell, M=m, N=n, NFP=2, sym=True, surface=surface,
        L_grid=2*ell, M_grid=2*m, N_grid=2*n,
        Psi=np.pi*np.sqrt(1-args.epsilon**2)*args.delta,
        pressure=PowerSeriesProfile([args.pa/mu_0, -2*args.delta/mu_0], modes=[0, 2]),
        iota=PowerSeriesProfile([2.0]), ensure_nested=False)
    fit_grid = LinearGrid(rho=np.linspace(0.02, 1, ell+2),
                          M=2*m, N=2*n, NFP=2, sym=False)
    rpz = geometry(fit_grid.nodes, args.epsilon, args.delta)
    # Omitted lambda data initializes its coefficients to zero, as derived above.
    eq.set_initial_guess(fit_grid, rpz[:, 0], rpz[:, 2], ensure_nested=False)
    # Point fitting updates volume coefficients only. Synchronize DESC's separate
    # axis representation, so its consistency projection preserves this guess.
    eq.axis = eq.get_axis()
    # The temporary scaled-boundary guess was not used. Check the fitted guess.
    if not eq.is_nested():
        raise RuntimeError("Analytic spectral fit is not nested; increase --resolution.")
    return eq


def check_local(eq, args, checks, nodes):
    eps, delta = args.epsilon, args.delta
    names = ['R', 'phi', 'Z', 'B', 'J', 'p', 'grad(p)', 'grad(B)',
             'sqrt(g)', 'iota', 'e^rho']
    d = compute(eq, names, nodes)
    rpz = np.column_stack((d['R'], d['phi'], d['Z']))
    xyz = xyz_from_rpz(rpz)
    x, y, z = xyz.T
    s = x*x/(1+eps) + y*y/(1-eps)
    inside = bool(np.all(2*s-s*s-4*z*z > 0))
    checks.condition('All sampled points inside analytic field domain', inside)
    if not inside:
        raise RuntimeError("Solved geometry leaves analytic domain; cannot compare B.")

    B = vector_to_xyz(d['B'], d['phi'])
    curlB = mu_0*vector_to_xyz(d['J'], d['phi'])
    gradp = mu_0*vector_to_xyz(d['grad(p)'], d['phi'])
    exact = field(xyz, eps)
    dB, curl_exact, gradpsi = analytic_derivatives(xyz, eps)
    psi = flux_label(xyz, eps)
    rho = nodes[:, 0]
    # Fixed characteristic scales avoid division by vanishing local quantities.
    Bscale = np.sqrt(np.mean(np.sum(exact**2, axis=1)))
    Jscale = np.sqrt(np.mean(np.sum(curl_exact**2, axis=1)))
    Pscale = np.sqrt(np.mean(np.sum((2*gradpsi)**2, axis=1)))
    checks.error('Analytic divergence (1/m)', np.trace(dB, axis1=1, axis2=2), 1e-11)
    checks.error('Analytic force identity / pressure-gradient scale',
                 (np.cross(curl_exact, exact)+2*gradpsi)/Pscale, 1e-11)
    uv = labels(xyz, eps)
    checks.error('Two independent analytic psi formulas / delta',
                 (psi-((uv[:, 0]+eps/2)**2+uv[:, 1]**2))/delta, 1e-11)
    checks.error('B vector / analytic RMS B', (B-exact)/Bscale, 1e-4)
    checks.error('Current vector / analytic RMS current', (curlB-curl_exact)/Jscale, 2e-3)
    checks.error('Flux-surface label error / delta', (psi-delta*rho**2)/delta, 2e-4)
    # Normalize pressure errors by the pressure DROP, not the large offset pa.
    checks.error('Pressure error / axis-to-edge pressure drop',
                 (mu_0*d['p']-(args.pa-2*psi))/(2*delta), 2e-4)
    checks.error('Pressure gradient / analytic RMS gradient',
                 (gradp+2*gradpsi)/Pscale, 2e-3)
    checks.error('DESC force balance / pressure-gradient scale',
                 (np.cross(curlB, B)-gradp)/Pscale, 2e-3)
    checks.error('Force vs analytic pressure gradient / scale',
                 (np.cross(curlB, B)+2*gradpsi)/Pscale, 2e-3)
    checks.error('B tangent to analytic surfaces / B gradpsi scale',
                 np.sum(B*gradpsi, axis=1)/(Bscale*Pscale/2), 2e-4)
    # DESC's tensor has derivative direction FIRST, vector component SECOND.
    checks.error('DESC divergence / RMS B (1/m)',
                 np.trace(d['grad(B)'], axis1=1, axis2=2)/Bscale, 1e-9)
    convection = vector_to_xyz(np.einsum('ni,nij->nj', d['B'], d['grad(B)']), d['phi'])
    target = -xyz*np.array([1, 1, 4])
    checks.error('Convective derivative vs (-x,-y,-4z), relative',
                 (convection-target)/np.sqrt(np.mean(np.sum(target**2, axis=1))), 2e-3)
    checks.condition('Positive sampled off-axis coordinate Jacobian', np.all(d['sqrt(g)'] > 0))
    checks.condition('Positive sampled physical toroidal B', np.all(d['B'][:, 1] > 0))
    checks.condition('Nonnegative pressure', np.all(d['p'] >= 0))
    checks.error('Prescribed DESC iota = +2', d['iota']-2, 1e-12)
    boundary = np.isclose(rho, 1)
    checks.error('Boundary psi error / delta', (psi[boundary]-delta)/delta, 2e-4)
    normals = d['e^rho'][boundary]
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    checks.error('Numerical B normal to numerical boundary / RMS B',
                 np.sum(d['B'][boundary]*normals, axis=1)/Bscale, 1e-10)
    return dict(B_rms_T=Bscale, curlB_rms_T_per_m=Jscale,
                mu0_gradp_rms_T2_per_m=Pscale,
                minimum_jacobian=float(np.min(d['sqrt(g)'])),
                maximum_B_error_T=float(np.max(np.linalg.norm(B-exact, axis=1))))


def check_axis(eq, args, checks):
    phi = np.linspace(0, 2*np.pi, 129, endpoint=False) + 0.017
    nodes = np.column_stack((np.zeros_like(phi), np.zeros_like(phi), phi))
    d = compute(eq, ['R', 'phi', 'Z', 'B', 'J'], nodes)
    Raxis = np.sqrt(1-args.epsilon**2)
    checks.error('Axis R error / minor-radius scale',
                 (d['R']-Raxis)/np.sqrt(args.delta), 2e-4)
    checks.error('Nonplanar axis Z error / minor-radius scale',
                 (d['Z']-args.epsilon/2*np.sin(2*phi))/np.sqrt(args.delta), 2e-4)
    xyz = xyz_from_rpz(np.column_stack((d['R'], phi, d['Z'])))
    B = vector_to_xyz(d['B'], phi)
    checks.error('Axis B vector (T)', B-field(xyz, args.epsilon), 2e-4)
    target = 4/(1-args.epsilon**2)*d['B']
    checks.error('Axis parallel-current identity, relative',
                 (mu_0*d['J']-target)/np.sqrt(np.mean(np.sum(target**2, axis=1))), 2e-3)


def check_integrals(eq, args, checks):
    # Higher-resolution quadrature than the force solve; weights span the full torus.
    g = QuadratureGrid(L=2*eq.L+4, M=2*eq.M+3, N=2*eq.N+3, NFP=eq.NFP)
    d = eq.compute(['sqrt(g)', '|B|^2', 'p', 'B^zeta', '<beta>_vol'],
                   grid=g, override_grid=False)
    dv = np.asarray(d['sqrt(g)'])*np.asarray(g.weights)
    volume = np.sum(dv)
    meanB2 = np.sum(dv*np.asarray(d['|B|^2']))/volume
    meanp = np.sum(dv*np.asarray(d['p']))/volume
    meanp_note = mu_0*meanp
    # Attachment Eqs. (1), (5), (7): dV is uniform in the analytic psi.
    # DESC pressure is in Pa; multiplying by mu_0 restores the note's units.
    Vexact = 2*np.pi**2*np.sqrt(1-args.epsilon**2)*args.delta
    pexact_note = args.pa-args.delta
    B2exact = 1-args.epsilon**2/2+args.delta
    beta = 2*meanp_note/meanB2
    beta_exact = 2*pexact_note/B2exact
    beta_desc = float(d['<beta>_vol'])
    # Integral errors converge faster than pointwise field/gradient errors.
    # Require 0.1 ppm agreement, including pressure relative to its drop so
    # the large additive pressure offset cannot conceal a quadrature error.
    average_rtol = 1e-7
    # Integral of B^zeta sqrt(g) over all three coordinates = 2*pi*Psi.
    flux = np.sum(dv*np.asarray(d['B^zeta']))/(2*np.pi)
    checks.error('Integrated toroidal flux, relative', (flux-eq.Psi)/eq.Psi, 1e-9)
    checks.error('Total volume, relative', (volume-Vexact)/Vexact, 2e-4)
    checks.error('Volume-average B squared, relative',
                 (meanB2-B2exact)/B2exact, average_rtol)
    checks.error('Volume-average pressure, relative',
                 (meanp_note-pexact_note)/pexact_note, average_rtol)
    checks.error('Mean pressure error / pressure drop',
                 (meanp_note-pexact_note)/(2*args.delta), average_rtol)
    checks.error('Volume-average beta (ratio of averages), relative',
                 (beta-beta_exact)/beta_exact, average_rtol)
    checks.error('DESC beta vs integrated ratio of averages, relative',
                 (beta_desc-beta)/beta, 1e-12)
    print(f'Volume averages (numerical / analytic):\n'
          f'  <p_note> = {meanp_note:.12g} / {pexact_note:.12g}\n'
          f'  <|B|^2> = {meanB2:.12g} / {B2exact:.12g} T^2\n'
          f'  beta_V  = {beta:.12g} / {beta_exact:.12g} (dimensionless)',
          flush=True)
    energy = volume*meanB2/(2*mu_0)
    energy_exact = Vexact*B2exact/(2*mu_0)
    checks.error('Magnetic energy, relative', (energy-energy_exact)/energy_exact, 2e-4)
    return dict(volume_m3=float(volume), exact_volume_m3=float(Vexact),
                toroidal_flux_Wb=float(flux), magnetic_energy_J=float(energy),
                exact_magnetic_energy_J=float(energy_exact),
                mean_pressure_Pa=float(meanp), exact_mean_pressure_Pa=float(pexact_note/mu_0),
                mean_pressure_note=float(meanp_note), exact_mean_pressure_note=float(pexact_note),
                mean_B_squared_T2=float(meanB2), exact_mean_B_squared_T2=float(B2exact),
                beta_vol=float(beta), exact_beta_vol=float(beta_exact),
                desc_beta_vol=beta_desc)


def check_field_lines(eq, args, checks):
    """Integrate the *solved* lambda field; test independent analytic labels."""
    coeff = jnp.asarray(eq.L_lmn)

    @jit
    def rhs(phi, theta, rho):
        nodes = jnp.array([[rho, theta[0], phi]])
        lt = (eq.L_basis.evaluate(nodes, derivatives=[0, 1, 0]) @ coeff)[0]
        lz = (eq.L_basis.evaluate(nodes, derivatives=[0, 0, 1]) @ coeff)[0]
        return jnp.array([(2.0-lz)/(1.0+lt)])

    phi = np.linspace(0.17, 0.17+2*np.pi, 801)
    for k, (rho, theta0) in enumerate([(0.25, 0.21), (0.60, 1.1), (0.90, 2.4)], 1):
        sol = solve_ivp(lambda p, t: np.asarray(rhs(p, t, rho)),
                        (phi[0], phi[-1]), [theta0], t_eval=phi,
                        rtol=2e-10, atol=2e-11, max_step=0.04, method='DOP853')
        checks.condition(f'Line {k}: numerical integration succeeded', sol.success)
        if not sol.success:
            continue
        nodes = np.column_stack((np.full_like(phi, rho), sol.y[0], phi))
        d = compute(eq, ['R', 'Z'], nodes)
        rpz = np.column_stack((d['R'], phi, d['Z']))
        xyz = xyz_from_rpz(rpz)
        uv = labels(xyz, args.epsilon)
        reference = section(np.full_like(phi, uv[0, 0]), np.full_like(phi, uv[0, 1]),
                            phi, args.epsilon)
        checks.error(f'Line {k}: analytic u,v drift / minor-radius scale',
                     (uv-uv[0])/np.sqrt(args.delta), 4e-4)
        checks.error(f'Line {k}: trajectory vs exact line / radius',
                     (xyz-xyz_from_rpz(reference))/np.sqrt(args.delta), 4e-4)
        psi = flux_label(xyz, args.epsilon)
        checks.error(f'Line {k}: analytic flux drift / delta', (psi-psi[0])/args.delta, 4e-4)
        angle = np.unwrap(np.arctan2(d['Z']-args.epsilon/2*np.sin(2*phi),
                                    d['R']-np.sqrt(1-args.epsilon**2)))
        winding = (angle[-1]-angle[0])/(2*np.pi)
        checks.error(f'Line {k}: geometric winding minus (-2)', winding+2, 1e-7)
        checks.error(f'Line {k}: one-turn return distance / radius',
                     np.linalg.norm(xyz[-1]-xyz[0])/np.sqrt(args.delta), 1e-7)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--epsilon', type=float, default=0.25)
    parser.add_argument('--delta', type=float, default=1/64)
    parser.add_argument('--pa', type=float, default=1.0, help='Pressure offset in note units')
    parser.add_argument('--resolution', type=int, default=8, help='M=N; radial L=M+2 (default 8)')
    parser.add_argument('--maxiter', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260910)
    parser.add_argument('--output-dir', type=Path, default=Path('analytic_3D_equilibrium_results'))
    args = parser.parse_args()
    if not (0 < args.epsilon < 1 and 0 < args.delta < (1-args.epsilon)**2/4):
        parser.error('Require 0 < epsilon < 1 and 0 < delta < (1-epsilon)^2/4.')
    if not args.pa >= 2*args.delta:
        parser.error('Require pa >= 2*delta so pressure is nonnegative throughout.')
    if args.resolution < 2 or args.maxiter < 1:
        parser.error('Require resolution >= 2 and maxiter >= 1.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checks = Checks()
    report = dict(desc_version=desc.__version__, numpy_version=np.__version__,
                  parameters={k: str(v) if isinstance(v, Path) else v
                              for k, v in vars(args).items()}, checks=checks.results,
                  completed=False)
    stem = args.output_dir / '20260910-01_analytic_3D_equilibrium_desc'
    print(f'DESC {desc.__version__}; epsilon={args.epsilon}, delta={args.delta}', flush=True)
    print('DESC iota=+2; physical geometric winding=-2; SI p=p_note/mu_0.', flush=True)
    try:
        eq = build_equilibrium(args)
        # Independent points, including near-axis samples and the complete boundary.
        rng = np.random.default_rng(args.seed)
        random = np.column_stack((np.sqrt(rng.uniform(0.0025, 1, 1000)),
                                  rng.uniform(0, 2*np.pi, 1000),
                                  rng.uniform(0, 2*np.pi, 1000)))
        shell = LinearGrid(rho=[0.01, 0.17, 0.49, 0.83, 1.0],
                           M=args.resolution+3, N=args.resolution+3, NFP=2, sym=False)
        shifted = np.asarray(shell.nodes).copy()
        shifted[:, 1:] += [0.113, 0.071]  # different from fitting/solver angles
        nodes = np.vstack((random, shifted))
        initial = compute(eq, ['R', 'phi', 'Z', 'B'], nodes)
        xyz = xyz_from_rpz(np.column_stack((initial['R'], initial['phi'], initial['Z'])))
        initial_error = float(np.max(np.linalg.norm(
            vector_to_xyz(initial['B'], initial['phi'])-field(xyz, args.epsilon), axis=1)))
        if not np.isfinite(initial_error):
            raise RuntimeError('Initial fit has nonfinite B; increase --resolution.')
        report['initial_maximum_B_error_T'] = initial_error
        print(f"Initial fit max |B-B_exact|: {report['initial_maximum_B_error_T']:.3e} T", flush=True)
        eq, info = eq.solve(ftol=1e-10, xtol=1e-10, gtol=1e-8,
                            maxiter=args.maxiter, verbose=2)
        report['solver'] = dict(success=bool(info['success']), message=str(info['message']),
                                iterations=int(info.get('nit', -1)))
        checks.condition('DESC optimizer reported convergence', info['success'])
        # Save even when checks fail, so the computed result remains inspectable.
        eq.save(str(stem.with_suffix('.h5')))
        report['sample_count'] = len(nodes)
        report['local'] = check_local(eq, args, checks, nodes)
        check_axis(eq, args, checks)
        report['integrals'] = check_integrals(eq, args, checks)
        check_field_lines(eq, args, checks)
        report['completed'] = True
    except Exception as exc:
        checks.condition('Execution completed without an exception', False)
        report['exception'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['passed'] = report['completed'] and checks.passed
        stem.with_suffix('.checks.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(f'Check report: {stem.with_suffix(".checks.json").resolve()}', flush=True)
    failed = [row['name'] for row in checks.results if not row['passed']]
    if failed:
        print('\nFAILED: ' + '; '.join(failed), file=sys.stderr)
        print('Try a higher --resolution; inspect the report before using the equilibrium.', file=sys.stderr)
        return 1
    print(f'\nAll {len(checks.results)} checks passed. Equilibrium: {stem.with_suffix(".h5").resolve()}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
