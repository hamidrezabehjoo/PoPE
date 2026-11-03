#!/usr/bin/env python3
"""
Dimension scaling of the CFG bed: product bed vs mean-field bed
(Remark rem:dimcomp and Figure fig:dimscale of
"When Diffusion Composition Fails: Exact Products of Path Measures").

Per-coordinate bed: p0 = N(0, 2.25), pc = N(0.8, 0.25), g = 1 on [0,1].
Naive data-end law: closed-form moment ODEs; weight moments E[W^lam]:
exact Gaussian path MGF (gaussian_path_mgf, discrete Cameron-Martin).

Geometry A (product bed): d iid copies of the fixed bed.
  KL(naive||q_w) = d*KL_1, log ESS = d*log ESS_1, w* dimension-independent.
Geometry B (mean-field bed): mu_d = 0.8/sqrt(d), vc_d = 2.25(1+eta/sqrt(d)),
  eta = 0.25/2.25 - 1. Then total naive KL = Theta(1/d), ESS -> 1, and
  w*(d) ~ 4.1 sqrt(d).

Reference values (this script):
  product bed, w=1.25/1.5/1.75: KL_1 = 0.00420/0.01282/0.02219,
    ESS_1 = 0.973/0.863/0.590
  mean-field bed, w=1.5: d*KL_tot -> 0.0086 (1/d decay)
  w*(d) = 1.95, 6.18, 14.58, 31.40, 65.04, 132.32 at d = 1,4,16,64,256,1024
"""
import math

import numpy as np
from scipy.integrate import solve_ivp

from gmm2d import gaussian_path_mgf

V0, VC, MU = 2.25, 0.25, 0.8
ETA = VC / V0 - 1.0


def naive_law_cfg(v0, vc, mu, w):
    """Data-end moments of the naive composed sampler (continuum ODEs)."""
    def rhs(t, y):
        M, V = y
        A = (1 - w) / (v0 + t) + w / (vc + t)
        C = -w * mu / (vc + t)
        return [A * M + C, 2 * A * V - 1.0]
    P1 = (1 - w) / (v0 + 1.0) + w / (vc + 1.0)
    sol = solve_ivp(rhs, [1.0, 0.0], [(w * mu / (vc + 1.0)) / P1, 1.0 / P1],
                    rtol=1e-12, atol=1e-14)
    return sol.y[0, -1], sol.y[1, -1]


def target_cfg(v0, vc, mu, w):
    P = (1 - w) / v0 + w / vc
    return (w * mu / vc) / P, 1.0 / P


def kl_gauss(m1, v1, m2, v2):
    return 0.5 * (v1 / v2 - 1 + (m2 - m1) ** 2 / v2 + math.log(v2 / v1))


def ess1(v0, vc, mu, w, M=600):
    e1, _ = gaussian_path_mgf(v0, vc, mu, w, 1.0, M)
    e2, ok = gaussian_path_mgf(v0, vc, mu, w, 2.0, M)
    return e1, (e1**2 / e2 if ok else 0.0)


def lambda_star_bed(v0, vc, mu, w, M=400, tol=5e-3):
    if w <= 1.0:
        return np.inf
    lo, hi = 0.0, 1.0
    while True:
        _, ok = gaussian_path_mgf(v0, vc, mu, w, hi, M)
        if not ok:
            break
        lo = hi
        hi *= 2.0
        if hi > 1e5:
            return np.inf
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        _, ok = gaussian_path_mgf(v0, vc, mu, w, mid, M)
        lo, hi = (mid, hi) if ok else (lo, mid)
    return lo


def wstar_bed(v0, vc, mu, M=300):
    lo, hi = 1.01, 3.0
    if lambda_star_bed(v0, vc, mu, hi, M) > 2:
        lo2 = hi
        for h2 in [5, 8, 12, 20, 40, 80, 160]:
            if lambda_star_bed(v0, vc, mu, h2, M) < 2:
                lo, hi = lo2, h2
                break
            lo2 = h2
        else:
            return np.inf
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if lambda_star_bed(v0, vc, mu, mid, M) > 2:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    print("=== Geometry A: product bed (fixed per-coordinate disagreement) ===")
    for w in [1.25, 1.5, 1.75]:
        mn, vn = naive_law_cfg(V0, VC, MU, w)
        mt, vt = target_cfg(V0, VC, MU, w)
        _, e1 = ess1(V0, VC, MU, w)
        print(f"  w={w}: KL_1={kl_gauss(mn, vn, mt, vt):.5f}, ESS_1={e1:.3f}"
              "  (totals: KL=d*KL_1, log ESS=d*log ESS_1, w*=1.95 fixed)")
    print("\n=== Geometry B: mean-field bed (disagreement ~ 1/sqrt d) ===")
    for d in [1, 4, 16, 64, 256, 1024]:
        mud, vcd = MU / math.sqrt(d), V0 * (1 + ETA / math.sqrt(d))
        for w in [1.25, 1.5]:
            mn, vn = naive_law_cfg(V0, vcd, mud, w)
            mt, vt = target_cfg(V0, vcd, mud, w)
            klt = d * kl_gauss(mn, vn, mt, vt)
            _, e1 = ess1(V0, vcd, mud, w)
            print(f"  d={d:>5} w={w}: KL_tot={klt:.6f}  "
                  f"-log10 ESS={-d * math.log10(e1):.6f}")
    print("\n=== w*(d) on the mean-field bed ===")
    for d in [1, 4, 16, 64, 256, 1024]:
        mud, vcd = MU / math.sqrt(d), V0 * (1 + ETA / math.sqrt(d))
        wsd = wstar_bed(V0, vcd, mud)
        print(f"  d={d:>5}: w*={wsd:.3f}   w*/sqrt(d)={wsd / math.sqrt(d):.3f}")


if __name__ == "__main__":
    main()
