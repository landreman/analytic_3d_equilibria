"""Exact non-axisymmetric toroidal scalar-pressure MHD equilibrium.

All quantities are dimensionless, with curl(B) x B = grad(p).
Positions have final axis of length 3. The routines also accept complex-step
perturbations for derivative verification. No field-line integration is needed.

Run ``python exact_equilibrium.py`` to execute the independent numerical checks.
Dependencies: Python >= 3.10, NumPy.
"""
from __future__ import annotations

import json
from typing import Callable
import numpy as np
from numpy.typing import ArrayLike, NDArray


def _scales(epsilon: float) -> tuple[float, float]:
    if not np.isfinite(epsilon) or not 0 <= epsilon < 1:
        raise ValueError("epsilon must satisfy 0 <= epsilon < 1")
    return np.sqrt(1 + epsilon), np.sqrt(1 - epsilon)


def field(xyz: ArrayLike, epsilon: float = 0.25) -> NDArray:
    """Evaluate the Cartesian field, eq. (2.5) of the accompanying manuscript."""
    a, b = _scales(epsilon)
    xyz = np.asarray(xyz)
    if xyz.shape[-1:] != (3,):
        raise ValueError("xyz must have final axis of length 3")
    x, y, z = np.moveaxis(xyz, -1, 0)
    s = (x / a) ** 2 + (y / b) ** 2
    radicand = 2 * s - s * s - 4 * z * z
    if np.any(np.real(radicand) <= 0) or np.any(np.real(s) <= 0):
        raise ValueError("Position is outside the analytic domain F^2 > 0")
    f = np.sqrt(radicand)
    return np.stack(((2 * z * x - (a / b) * f * y) / s,
                     (2 * z * y + (b / a) * f * x) / s,
                     1 - s), axis=-1)


def flux(xyz: ArrayLike, epsilon: float = 0.25) -> NDArray:
    """Flux-surface label psi, zero on the nonplanar magnetic axis.

    psi is not the toroidal flux divided by 2 pi; that label is a*b*psi/2.
    Squared quantities are computed without complex conjugation to allow
    complex-step differentiation.
    """
    xyz = np.asarray(xyz)
    magnetic_field = field(xyz, epsilon)
    x, y, z = np.moveaxis(xyz, -1, 0)
    return (x * x + y * y + 4 * z * z
            + np.sum(magnetic_field * magnetic_field, axis=-1)
            - 2 + epsilon * epsilon) / 4


def pressure(xyz: ArrayLike, epsilon: float = 0.25,
             p_axis: float = 1.0) -> NDArray:
    return p_axis - 2 * flux(xyz, epsilon)


def embedding(u: ArrayLike, v: ArrayLike, t: ArrayLike,
              epsilon: float = 0.25) -> NDArray:
    """Analytic diffeomorphism from u^2+v^2 < 1/4, t mod 2 pi, to F^2>0."""
    a, b = _scales(epsilon)
    u, v, t = np.broadcast_arrays(u, v, t)
    q2 = u * u + v * v
    if np.any(np.real(q2) >= 0.25):
        raise ValueError("Orbit labels must satisfy u^2 + v^2 < 1/4")
    ell = np.sqrt((1 + np.sqrt(1 - 4 * q2)) / 2)
    ct, st = np.cos(t), np.sin(t)
    return np.stack((a * (ell * ct + (u * ct + v * st) / ell),
                     b * (ell * st + (v * ct - u * st) / ell),
                     v * np.cos(2 * t) - u * np.sin(2 * t)), axis=-1)


def tangent(u: ArrayLike, v: ArrayLike, t: ArrayLike,
            epsilon: float = 0.25) -> NDArray:
    """Partial derivative of embedding with respect to t."""
    a, b = _scales(epsilon)
    u, v, t = np.broadcast_arrays(u, v, t)
    q2 = u * u + v * v
    if np.any(np.real(q2) >= 0.25):
        raise ValueError("Orbit labels must satisfy u^2 + v^2 < 1/4")
    ell = np.sqrt((1 + np.sqrt(1 - 4 * q2)) / 2)
    ct, st = np.cos(t), np.sin(t)
    return np.stack((a * (-ell * st + (-u * st + v * ct) / ell),
                     b * (ell * ct + (-v * st - u * ct) / ell),
                     -2 * v * np.sin(2 * t) - 2 * u * np.cos(2 * t)), axis=-1)


def inverse_embedding(xyz: ArrayLike, epsilon: float = 0.25) -> NDArray:
    """Recover the real coordinates (u, v, t), with t in (-pi, pi]."""
    a, b = _scales(epsilon)
    xyz = np.asarray(xyz)
    if np.iscomplexobj(xyz):
        raise ValueError("inverse_embedding expects real positions")
    field(xyz, epsilon)  # Validate the position.
    x, y, z = np.moveaxis(xyz, -1, 0)
    w = x / a + 1j * y / b
    eta = ((x / a) ** 2 + (y / b) ** 2 - 1) / 2 + 1j * z
    ell = np.sqrt((1 + np.sqrt(1 - 4 * np.abs(eta) ** 2)) / 2)
    eit = w / (ell + eta / ell)
    c = eta * eit * eit
    return np.stack((np.real(c), np.imag(c), np.angle(eit)), axis=-1)


def axis(t: ArrayLike, epsilon: float = 0.25) -> NDArray:
    _scales(epsilon)
    t = np.asarray(t)
    radius = np.sqrt(1 - epsilon * epsilon)
    return np.stack((radius * np.cos(t), radius * np.sin(t),
                     epsilon / 2 * np.sin(2 * t)), axis=-1)


def surface(psi: float, beta: ArrayLike, t: ArrayLike,
            epsilon: float = 0.25) -> NDArray:
    """Evaluate a complete embedded flux surface within the stated bound."""
    _scales(epsilon)
    if not np.isfinite(psi) or not 0 <= psi < (1 - epsilon) ** 2 / 4:
        raise ValueError("psi must satisfy 0 <= psi < (1-epsilon)^2/4")
    beta = np.asarray(beta)
    return embedding(-epsilon / 2 + np.sqrt(psi) * np.cos(beta),
                     np.sqrt(psi) * np.sin(beta), t, epsilon)


def _jacobian(function: Callable, point: ArrayLike) -> NDArray:
    """Complex-step Jacobian. Requires analytic arithmetic in function."""
    point = np.asarray(point, dtype=float)
    h = 1e-30
    values = [np.imag(function(point + 1j * h * e)) / h for e in np.eye(3)]
    return np.stack(values, axis=-1)


def _curl(jac: NDArray) -> NDArray:
    return np.array([jac[2, 1] - jac[1, 2],
                     jac[0, 2] - jac[2, 0],
                     jac[1, 0] - jac[0, 1]])


def verify(n_per_epsilon: int = 120, seed: int = 8) -> dict:
    """Verify Cartesian identities, coordinate map, axial current and winding.

    Numerical checks are not used as substitutes for the analytic proofs.
    Residuals are absolute dimensionless errors. The tests include the
    axisymmetric seed and several finite non-axisymmetric deformations.
    """
    rng = np.random.default_rng(seed)
    maxima = {key: 0.0 for key in (
        "divergence", "force_balance", "field_tangency", "acceleration",
        "embedding_tangent", "volume_jacobian", "inverse_embedding",
        "flux_in_labels", "axial_current")}
    winding = []
    for eps in (0.0, 0.25, 0.5, 0.8):
        a, b = _scales(eps)
        delta = 0.6 * (1 - eps) ** 2 / 4
        radii = np.sqrt(delta * rng.uniform(size=n_per_epsilon))
        beta = rng.uniform(0, 2 * np.pi, n_per_epsilon)
        u, v = -eps / 2 + radii * np.cos(beta), radii * np.sin(beta)
        t = rng.uniform(0, 2 * np.pi, n_per_epsilon)
        positions = embedding(u, v, t, eps)
        labels = inverse_embedding(positions, eps)
        rebuilt = embedding(labels[:, 0], labels[:, 1], labels[:, 2], eps)
        maxima["inverse_embedding"] = max(maxima["inverse_embedding"],
                                          float(np.max(np.abs(rebuilt - positions))))
        maxima["flux_in_labels"] = max(maxima["flux_in_labels"],
                                      float(np.max(np.abs(flux(positions, eps) - radii ** 2))))
        maxima["embedding_tangent"] = max(maxima["embedding_tangent"],
            float(np.max(np.abs(field(positions, eps) - tangent(u, v, t, eps)))))
        for point, uu, vv, tt in zip(positions, u, v, t):
            bb = field(point, eps)
            jac = _jacobian(lambda r: field(r, eps), point)
            current = _curl(jac)
            grad_psi = _jacobian(lambda r: flux(r, eps), point)
            mapped_jac = _jacobian(lambda c: embedding(*c, eps), [uu, vv, tt])
            residuals = {
                "divergence": abs(np.trace(jac)),
                "force_balance": np.linalg.norm(np.cross(current, bb) + 2 * grad_psi),
                "field_tangency": abs(bb @ grad_psi),
                "acceleration": np.linalg.norm(jac @ bb + point * np.array([1, 1, 4])),
                "volume_jacobian": abs(np.linalg.det(mapped_jac) + a * b),
            }
            for key, value in residuals.items():
                maxima[key] = max(maxima[key], float(value))
        ts = np.linspace(0, 2 * np.pi, 80, endpoint=False)
        for point in axis(ts, eps):
            current = _curl(_jacobian(lambda r: field(r, eps), point))
            axial_res = np.linalg.norm(current - 4 / (1 - eps ** 2) * field(point, eps))
            maxima["axial_current"] = max(maxima["axial_current"], float(axial_res))
        ts = np.linspace(0, 2 * np.pi, 4097)
        line = surface(delta / 2, 0.37, ts, eps)
        phi = np.unwrap(np.arctan2(line[:, 1], line[:, 0]))
        r = np.hypot(line[:, 0], line[:, 1])
        theta = np.unwrap(np.angle(r - np.sqrt(1 - eps * eps)
                          + 1j * (line[:, 2] - eps / 2 * np.sin(2 * phi))))
        toroidal = float((phi[-1] - phi[0]) / (2 * np.pi))
        poloidal = float((theta[-1] - theta[0]) / (2 * np.pi))
        closure = float(np.linalg.norm(line[-1] - line[0]))
        assert np.all(np.diff(phi) > 0)
        assert abs(toroidal - 1) < 1e-12 and abs(poloidal + 2) < 1e-12
        assert closure < 1e-12
        winding.append({"epsilon": eps, "toroidal_turns": toroidal,
                        "poloidal_turns": poloidal, "return_distance": closure})
    assert max(maxima.values()) < 1e-10, maxima
    return {"seed": seed, "number_of_interior_points": 4 * n_per_epsilon,
            "maximum_absolute_residuals": maxima, "winding_checks": winding,
            "status": "All checks passed"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
