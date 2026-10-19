#!/usr/bin/env python3
"""
Renyi--de Bruijn identity: numeric check.

Paper (proposed corollary of the normalizer identity of Lemma mixflow):
for K=2, no reference, S=1, exponents (beta, 1-beta), shared forward drift,

  hat Z_t = int (p1_t)^beta (p2_t)^{1-beta} = exp{-(1-beta) D_beta(p1_t||p2_t)}

and hence

  d/dt D_beta(p1_t || p2_t)
      = - (g_t^2 beta / 2) E_{p^beta_t} || grad log p1_t - grad log p2_t ||^2 ,

where p^beta_t ~ (p1_t)^beta (p2_t)^{1-beta} is the geometric (escort)
mixture. As beta -> 1 this reduces to the classical de Bruijn identity.

Checked here on Gaussian pairs under VE (f=0, g=1) and OU (f=-x/2, g=1)
noising, both in closed form; errors are at the finite-difference level.
"""
import numpy as np

MU1, MU2 = 1.5, -1.0
S1SQ, S2SQ = 0.25, 1.44


def make_gaussians(kind):
    """Return (mean_fns, var_fns, g2) for VE or OU noising."""
    if kind == "VE":
        return (lambda t: MU1, lambda t: MU2,
                lambda t: S1SQ + t, lambda t: S2SQ + t, 1.0)
    e = np.exp
    return (lambda t: MU1 * e(-t / 2), lambda t: MU2 * e(-t / 2),
            lambda t: 1 + (S1SQ - 1) * e(-t), lambda t: 1 + (S2SQ - 1) * e(-t),
            1.0)


def zhat_beta(t, beta, m1f, m2f, v1f, v2f):
    """int p1^beta p2^{1-beta} dx for 1D Gaussians (closed form)."""
    a, b = v1f(t), v2f(t)
    m1, m2 = m1f(t), m2f(t)
    Lam = beta / a + (1 - beta) / b
    C = beta * (1 - beta) * (m1 - m2) ** 2 / ((1 - beta) * a + beta * b)
    return np.exp(-0.5 * np.log(Lam) - 0.5 * beta * np.log(a)
                  - 0.5 * (1 - beta) * np.log(b) - 0.5 * C)


def renyi_D(t, beta, m1f, m2f, v1f, v2f):
    return -np.log(zhat_beta(t, beta, m1f, m2f, v1f, v2f)) / (1 - beta)


def escort_E_ds2(t, beta, m1f, m2f, v1f, v2f):
    """E_{p^beta_t} ||s1 - s2||^2 in closed form (p^beta_t Gaussian)."""
    a, b = v1f(t), v2f(t)
    m1, m2 = m1f(t), m2f(t)
    Lam = beta / a + (1 - beta) / b
    vb = 1.0 / Lam
    nu = (beta * m1 / a + (1 - beta) * m2 / b) * vb
    A = 1.0 / b - 1.0 / a                       # s1 - s2 = A x + B
    B = m1 / a - m2 / b
    return A * A * (vb + nu * nu) + 2 * A * B * nu + B * B


def check(kind, betas=(0.3, 0.5, 0.7), ts=(0.1, 0.5, 0.9), h=1e-6):
    m1f, m2f, v1f, v2f, g2 = make_gaussians(kind)
    worst = 0.0
    print(f"--- {kind} noising ---")
    for beta in betas:
        for t in ts:
            dD = (renyi_D(t + h, beta, m1f, m2f, v1f, v2f)
                  - renyi_D(t - h, beta, m1f, m2f, v1f, v2f)) / (2 * h)
            rhs = -g2 * beta / 2 * escort_E_ds2(t, beta, m1f, m2f, v1f, v2f)
            err = abs(dD - rhs)
            worst = max(worst, err)
            print(f"  beta={beta:.1f} t={t:.1f}: dD/dt={dD:+.8f}  "
                  f"RHS={rhs:+.8f}  |diff|={err:.2e}")
    print(f"  worst |diff| ({kind}): {worst:.2e}\n")
    return worst


def main():
    w = max(check("VE"), check("OU"))
    print(f"identity verified: worst finite-difference discrepancy {w:.2e}")


if __name__ == "__main__":
    main()
