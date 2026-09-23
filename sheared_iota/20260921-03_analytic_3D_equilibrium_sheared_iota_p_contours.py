#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt

# # Configuration A:
# eps = 1.08
# S = 3
# delta = 0.245
# lambd = 3.5

# Configuration B:
eps = 4
S = 3.5
delta = 0.245
lambd = 3.5

# # Configurtion C:
# eps = 5.6
# S = 4
# delta = 0.32
# lambd = 2

phis = np.linspace(0, 1 * np.pi, 16, endpoint=False)
n_R = 80
n_Z = 51

R_min = 1.15
R_max = 2.9
Z_max = 0.5
Z_min = -Z_max

R_1D = np.linspace(R_min, R_max, n_R)
Z_1D = np.linspace(Z_min, Z_max, n_Z)
R, Z = np.meshgrid(R_1D, Z_1D)

I = 1.0j

def compute(R, phi, Z):
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    z = Z
    omega = x + I * y
    omegabar = x - I * y
    K = omegabar * np.sqrt(1 + eps / omegabar**2)
    Xi = omega * K + np.pi / 2 - S
    Bx = np.real(I * np.exp(-I * lambd * z) * np.sin(Xi) / (2 * K))
    By = np.imag(I * np.exp(-I * lambd * z) * np.sin(Xi) / (2 * K))
    Bz = (1 / lambd) * np.real(np.exp(-I * lambd * z) * np.cos(Xi))
    psi = (np.sin(lambd * z)**2 + np.real(np.exp(-I * lambd * z) * np.cos(Xi))**2) / 2
    pa = delta / (lambd**2)
    p = pa - (1 / lambd)**2 * psi
    return Bx, By, Bz, psi, p

plt.figure(figsize=(14.5, 8.1))
n_rows = 4
n_cols = 4

index = 0
for i in range(n_rows):
    for j in range(n_cols):
        phi = phis[index]
        index += 1
        plt.subplot(n_rows, n_cols, i * n_cols + j + 1)

        Bx, By, Bz, psi, p = compute(R, phi, Z)
        p = np.where(p > 0, p, 0)

        plt.contourf(R, Z, p)
        plt.colorbar()
        plt.gca().set_aspect("equal")
        plt.xlabel("R")
        plt.ylabel("Z")
        plt.title(f"pressure for phi = {phi:.3f}")

plt.suptitle(f"iota = 2 family, epsilon = {eps}, delta = {delta}, S = {S}, lambda = {lambd}")
plt.figtext(0.5, 0.005, os.path.abspath(__file__), ha="center", fontsize=7)
plt.tight_layout()
plt.show()