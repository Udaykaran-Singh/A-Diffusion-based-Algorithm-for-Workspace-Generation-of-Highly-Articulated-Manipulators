import numpy as np
from scipy.linalg import expm
from scipy.special import jv

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


L_B = 7
L_B_PRIME = 4
P_MAX = 100.0

PAPER_ALPHA = 0.0
PAPER_BETA = 0.12
PAPER_EPSILON = 0.08
PAPER_L4 = 6.0
PAPER_L5 = 7.5
PAPER_REPORTED_MIN_COST = 0.5748


PAPER_BETA_GRID = np.arange(0.02, 0.74 + 1e-12, 0.02)
PAPER_EPSILON_GRID = np.arange(0.00, 0.80 + 1e-12, 0.04)


# ------>
# Fourier representation of the SE(2) diffusion equation

def build_eta_matrices(p, l_b=L_B):
    """Build eta(X1,p) and eta(X3,p) from the paper.

    Eq. (4):  eta_mn(X1,p) = -j m delta_mn
    Eq. (6):  eta_mn(X3,p) = p/2 (delta_{m,n+1} - delta_{m,n-1})
    """
    m_values = np.arange(-l_b, l_b + 1)

    # Eq. (4)
    eta_x1 = np.diag(-1j * m_values).astype(complex)

    # Eq. (6)
    size = 2 * l_b + 1
    eta_x3 = np.zeros((size, size), dtype=complex)
    for i, m in enumerate(m_values):
        for j, n in enumerate(m_values):
            eta_x3[i, j] = (p / 2.0) * ((1.0 if m == n + 1 else 0.0) - (1.0 if m == n - 1 else 0.0))

    return eta_x1, eta_x3


def build_B(p, alpha, beta, epsilon, l_b=L_B):
    """Fourier-space coefficient matrix corresponding to paper Eq. (7)/(8)."""
    eta_x1, eta_x3 = build_eta_matrices(p, l_b=l_b)
    return alpha * eta_x1 + beta * (eta_x1 @ eta_x1) + eta_x3 + epsilon * (eta_x3 @ eta_x3)


def fhat_full(p, L, alpha, beta, epsilon, l_b=L_B):
    """Paper Eq. (8): f_hat(p;L) = exp(B(p)L)."""
    return expm(build_B(p, alpha, beta, epsilon, l_b=l_b) * L)


def fhat_row0(p, L, alpha, beta, epsilon, l_b=L_B):
    """m=0 row, sufficient after integrating positional density over phi."""
    return fhat_full(p, L, alpha, beta, epsilon, l_b=l_b)[l_b, :]


def precompute_row0(p_grid, L, alpha, beta, epsilon, l_b=L_B):
    return np.asarray([fhat_row0(p, L, alpha, beta, epsilon, l_b=l_b) for p in p_grid])


def positional_density_polar(r, theta, p_grid, row0_grid, l_b=L_B, l_b_prime=L_B_PRIME):
    """Reconstruct positional density by the SE(2) inverse Fourier formula.

    Integrating over phi leaves the m=0 row."""
    total = 0.0j

    for n in range(-l_b_prime, l_b_prime + 1):
        col = n + l_b
        fhat_0n = row0_grid[:, col]

        integrand = fhat_0n * jv(-n, p_grid * r) * p_grid
        radial_integral = np.trapezoid(integrand, p_grid)

        total += (
            (1j) ** (-n)
            * np.exp(-1j * n * theta)
            * radial_integral
        )

    return float((2.0 * np.pi * total).real)


def compute_density_field(L, alpha, beta, epsilon, extent, n_grid=110, p_max=P_MAX, n_p=400, forward_is_positive_y=True):
    """Compute the positional workspace density on an x-y grid."""
    p_grid = np.linspace(1e-6, p_max, n_p)
    row0_grid = precompute_row0(p_grid, L, alpha, beta, epsilon)

    xs = np.linspace(-extent, extent, n_grid)
    ys = np.linspace(-extent, extent, n_grid)
    density = np.zeros((n_grid, n_grid), dtype=float)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            y_for_formula = -y if forward_is_positive_y else y
            r = np.hypot(x, y_for_formula)

            if r < 1e-12:
                continue

            theta = np.arctan2(y_for_formula, x)
            density[iy, ix] = positional_density_polar(r, theta, p_grid, row0_grid)

    # Band-limited inverse transforms can have tiny negative values.
    density[density < 0.0] = 0.0
    return xs, ys, density


# ------>
# Full (x,y,phi) moment reconstruction for the paper's cost function

def precompute_full_matrices(p_grid, L, alpha, beta, epsilon):
    return np.asarray([fhat_full(p, L, alpha, beta, epsilon) for p in p_grid])


def C_m_at_point(m, r, theta, p_grid, matrix_grid, l_b=L_B, l_b_prime=L_B_PRIME):
    """Coefficient of exp(-j m phi) in the inverse SE(2) transform."""
    row = m + l_b
    total = 0.0j

    for n in range(-l_b_prime, l_b_prime + 1):
        col = n + l_b
        fhat_mn = matrix_grid[:, row, col]
        integrand = fhat_mn * jv(m - n, p_grid * r) * p_grid
        radial_integral = np.trapezoid(integrand, p_grid)

        total += (
            (1j) ** (m - n)
            * np.exp(-1j * (n - m) * theta)
            * radial_integral
        )

    return total


def phi_first_integral(m):
    """Integral_{-pi}^{pi} phi exp(-j m phi) dphi."""
    if m == 0:
        return 0.0
    return 2.0 * np.pi * 1j * ((-1) ** m) / m


def phi_second_integral(m):
    """Integral_{-pi}^{pi} phi^2 exp(-j m phi) dphi."""
    if m == 0:
        return 2.0 * np.pi**3 / 3.0
    return 4.0 * np.pi * ((-1) ** m) / (m**2)


def diffusion_moments(L, alpha, beta, epsilon, extent, n_grid=30, p_max=P_MAX, n_p=700, forward_is_positive_y=True):
    """Nine moments used by the paper's matching cost function."""
    p_grid = np.linspace(1e-6, p_max, n_p)
    matrices = precompute_full_matrices(p_grid, L, alpha, beta, epsilon)

    xs = np.linspace(-extent, extent, n_grid)
    ys = np.linspace(-extent, extent, n_grid)
    dx = xs[1] - xs[0]
    area = dx * dx

    S0 = np.zeros((n_grid, n_grid), dtype=float)
    S1phi = np.zeros((n_grid, n_grid), dtype=complex)
    S2phi = np.zeros((n_grid, n_grid), dtype=complex)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            y_for_formula = -y if forward_is_positive_y else y
            r = np.hypot(x, y_for_formula)

            if r < 1e-12:
                continue

            theta = np.arctan2(y_for_formula, x)
            coeffs = {m: C_m_at_point(m, r, theta, p_grid, matrices) for m in range(-L_B_PRIME, L_B_PRIME + 1)}

            S0[iy, ix] = (2.0 * np.pi * coeffs[0]).real
            S1phi[iy, ix] = sum(coeffs[m] * phi_first_integral(m) for m in coeffs)
            S2phi[iy, ix] = sum(coeffs[m] * phi_second_integral(m) for m in coeffs)

    S0[S0 < 0.0] = 0.0
    X, Y = np.meshgrid(xs, ys)

    normalizer = S0.sum() * area
    if normalizer <= 0:
        raise RuntimeError("Density normalization failed; increase numerical resolution.")

    mean_x = (S0 * X).sum() * area / normalizer
    mean_y = (S0 * Y).sum() * area / normalizer

    var_x = (S0 * (X - mean_x) ** 2).sum() * area / normalizer
    var_y = (S0 * (Y - mean_y) ** 2).sum() * area / normalizer
    cov_xy = (S0 * (X - mean_x) * (Y - mean_y)).sum() * area / normalizer

    mean_phi = (S1phi.sum() * area / normalizer).real
    mean_phi2 = (S2phi.sum() * area / normalizer).real
    var_phi = mean_phi2 - mean_phi**2

    cov_xphi = ((X * S1phi).sum() * area / normalizer).real - mean_x * mean_phi
    cov_yphi = ((Y * S1phi).sum() * area / normalizer).real - mean_y * mean_phi

    return {
        "mean_x": mean_x,
        "mean_y": mean_y,
        "mean_phi": mean_phi,
        "var_x": var_x,
        "var_y": var_y,
        "var_phi": var_phi,
        "cov_xy": cov_xy,
        "cov_xphi": cov_xphi,
        "cov_yphi": cov_yphi,
    }


def paper_cost(goal, model):
    """Nine-term squared-error cost used in the paper's parameter matching."""
    keys = (
        "mean_x", "mean_y", "mean_phi",
        "var_x", "var_y", "var_phi",
        "cov_xy", "cov_xphi", "cov_yphi",
    )
    return sum((goal[k] - model[k]) ** 2 for k in keys)


# ------>
# 16-state brute-force workspace (surrogate only)

# The paper gives 16 states/module and 1.5 units of L per module, but does not
# publish the 16 numerical local transforms. This symmetric state model is used
# only so the script can still draw Figure-2-style point clouds.
SURROGATE_MODULE_LENGTH = 1.5
SURROGATE_BEND_ANGLES = np.linspace(-np.pi / 4.0, np.pi / 4.0, 16)


def enumerate_workspace_surrogate(n_modules):
    """Enumerate 16^n surrogate configurations efficiently."""
    x = np.zeros(1, dtype=float)
    y = np.zeros(1, dtype=float)
    phi = np.zeros(1, dtype=float)

    for _ in range(n_modules):
        count = phi.size
        phi_old = np.repeat(phi, 16)
        x_old = np.repeat(x, 16)
        y_old = np.repeat(y, 16)
        bends = np.tile(SURROGATE_BEND_ANGLES, count)

        phi = phi_old + bends
        x = x_old + SURROGATE_MODULE_LENGTH * np.sin(phi)
        y = y_old + SURROGATE_MODULE_LENGTH * np.cos(phi)

    return np.column_stack((x, y, phi))


# ------>
# Paper figures

def make_figure2(filename="figure2_workspace_modules.png"):
    """Figure-2-style workspaces for 1,2,3,4 modules."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    for ax, n_modules in zip(axes.flat, [1, 2, 3, 4]):
        pts = enumerate_workspace_surrogate(n_modules)
        ax.scatter(pts[:, 0], pts[:, 1], s=.5, c="k", alpha=0.15)
        ax.set_title(f"{n_modules} module{'s' if n_modules > 1 else ''}")
        ax.set_aspect("equal")
        ax.set_xlim(-6, 6)
        ax.set_ylim(-3, 6)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle("Figure 2 style: workspace with different numbers of modules")
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close(fig)
    print(f"saved {filename}")


def make_figure3(filename="figure3_parameter_effects.png"):
    """Paper Figure 3 parameter combinations, exactly as listed in the paper."""

    cases = [
        ("(a) L=1, alpha=0, beta=1, epsilon=0",
         dict(L=1.0, alpha=0.0, beta=1.0, epsilon=0.0)),
        ("(b) L=2, alpha=0, beta=1, epsilon=0",
         dict(L=2.0, alpha=0.0, beta=1.0, epsilon=0.0)),
        ("(c) L=2, alpha=0, beta=1, epsilon=0.04",
         dict(L=2.0, alpha=0.0, beta=1.0, epsilon=0.04)),
        ("(d) L=2, alpha=0, beta=1.5, epsilon=0",
         dict(L=2.0, alpha=0.0, beta=1.5, epsilon=0.0)),
        ("(e) L=2, alpha=0.6, beta=1, epsilon=0",
         dict(L=2.0, alpha=0.6, beta=1.0, epsilon=0.0)),
        ("(f) L=2, alpha=-0.6, beta=1, epsilon=0",
         dict(L=2.0, alpha=-0.6, beta=1.0, epsilon=0.0)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for ax, (title, params) in zip(axes.flat, cases):
        L = params["L"]
        extent = 2.2 if L == 1.0 else 3.0

        xs, ys, Z = compute_density_field(extent=extent, n_grid=105, n_p=400, **params)

        vmax = np.percentile(Z[Z > 0], 98) if np.any(Z > 0) else None
        ax.contour(xs, ys, Z, cmap="gray_r", vmax=vmax)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()
    plt.savefig(filename, dpi=170)
    plt.close(fig)
    print(f"saved {filename}")


def make_paper_comparison(n_modules, L, filename, alpha=PAPER_ALPHA, beta=PAPER_BETA, epsilon=PAPER_EPSILON):
    """Paper Fig. 5/6 style: brute-force workspace beside diffusion density."""
    pts = enumerate_workspace_surrogate(n_modules)

    extent = max(7.0, L + 0.5)
    xs, ys, Z = compute_density_field(L=L, alpha=alpha, beta=beta, epsilon=epsilon, extent=extent, n_grid=120, n_p=450)

    vmax = np.percentile(Z[Z > 0], 98) if np.any(Z > 0) else None

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].scatter(pts[:, 0], pts[:, 1], s=.5, c="k", alpha=0.10)
    axes[0].set_title(f"Brute force surrogate ({n_modules} modules)")
    axes[0].set_aspect("equal")
    axes[0].set_xlim(-extent, extent)
    axes[0].set_ylim(-3, extent)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    axes[1].pcolormesh(xs, ys, Z, cmap="gray_r", shading="auto", vmax=vmax)
    axes[1].set_title(f"Diffusion: L={L:g}, alpha={alpha:g}, beta={beta:g}, epsilon={epsilon:g}")
    axes[1].set_aspect("equal")
    axes[1].set_xlim(-extent, extent)
    axes[1].set_ylim(-3, extent)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close(fig)
    print(f"saved {filename}")


def print_paper_parameter_search_spec():
    """Print the exact parameter search specification reported in the paper."""
    print("Paper parameter matching specification")
    print("--------------------------------------")
    print(f"L fixed to: {PAPER_L4}")
    print(f"alpha fixed to: {PAPER_ALPHA}")
    print("beta search: 0.02 ... 0.74, step 0.02")
    print("epsilon search: 0.00 ... 0.80, step 0.04")
    print(
        "Reported optimum: "
        f"alpha={PAPER_ALPHA}, beta={PAPER_BETA}, epsilon={PAPER_EPSILON}"
    )
    print(f"Reported minimum cost: {PAPER_REPORTED_MIN_COST}")
    print(f"4-module L: {PAPER_L4}")
    print(f"5-module L: {PAPER_L5}")


# ------>
# Tests and main

def sanity_check_eta_x3():
    """Check Eq. (6) pattern for one arbitrary p."""
    p = 0.7
    _, eta_x3 = build_eta_matrices(p)
    center = L_B

    assert np.isclose(eta_x3[center + 1, center], +p / 2.0)
    assert np.isclose(eta_x3[center - 1, center], -p / 2.0)
    return True


def sanity_check_matrix_exponential():
    """For alpha=beta=epsilon=0, compare with the known Bessel closed form."""
    p = 0.5
    L = 1.0
    row0 = fhat_row0(p, L, alpha=0.0, beta=0.0, epsilon=0.0)

    errors = []
    for n in range(-L_B, L_B + 1):
        expected = ((-1) ** n) * jv(n, p * L)
        errors.append(abs(row0[n + L_B].real - expected))

    return max(errors)


if __name__ == "__main__":
    print("=" * 78)
    print("Wang & Chirikjian ICRA 2002 -- paper-faithful implementation")
    print("=" * 78)

    sanity_check_eta_x3()
    err = sanity_check_matrix_exponential()
    print(f"Matrix-exponential sanity-check max error: {err:.3e}")

    print()
    print_paper_parameter_search_spec()

    print("\nGenerating assignment figures...")

    make_figure2("figure2_workspace_modules.png")

    make_figure3("figure3_parameter_effects.png")

    make_paper_comparison(
        n_modules=4,
        L=PAPER_L4,
        filename="figure5_four_module_comparison.png",
    )

    make_paper_comparison(
        n_modules=5,
        L=PAPER_L5,
        filename="figure6_five_module_comparison.png",
    )

    print("\nDone.")
