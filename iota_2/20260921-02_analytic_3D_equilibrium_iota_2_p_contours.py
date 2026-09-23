#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt

eps = 0.5
delta = 1 / 64

a = np.sqrt(1 + eps)
b = np.sqrt(1 - eps)

phis = np.linspace(0, 2 * np.pi, 16, endpoint=False)
n_R = 60
n_Z = 61

R_min = 0.5
R_max = 1.3
Z_max = 0.4
Z_min = -Z_max

R_1D = np.linspace(R_min, R_max, n_R)
Z_1D = np.linspace(Z_min, Z_max, n_Z)
R, Z = np.meshgrid(R_1D, Z_1D)

def compute(R, phi, Z):
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    z = Z
    s = (x / a)**2 + (y / b)**2
    F = np.sqrt(1 - (1 - s)**2 - 4 * z**2)
    Bx = (2 * z * x - (a / b) * F * y) / s
    By = (2 * z * y + (b / a) * F * x) / s
    Bz = 1 - s
    modB2 = Bx**2 + By**2 + Bz**2
    psi = (x**2 + y**2 + 4*z**2 + modB2 - 2 + eps**2) / 4
    pa = 2 * delta
    p = pa - 2 * psi
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

plt.suptitle(f"iota = 2 family, epsilon = {eps}, delta = {delta}")
plt.figtext(0.5, 0.005, os.path.abspath(__file__), ha="center", fontsize=7)
plt.tight_layout()
plt.show()