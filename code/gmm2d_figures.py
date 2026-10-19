#!/usr/bin/env python3
"""
Full figure driver for the 2D test beds (Section 4.4 of the paper).

Reproduces, into ../output/:
  gmm2d_degeneracy.{png,pdf}   ESS/N and E[W] vs guidance weight w
  gmm2d_scaling_tv.{png,pdf}   ESS/N vs particle count N; TV to exact target
  gmm2d_densities.{png,pdf}    sampler clouds vs tempered-target HDR contours
  gmm2d_sweep.csv              full diagnostics table

Corrected vs the original development script:
  * deterministic stable_seed() instead of salted hash();
  * the N-scaling loop no longer breaks early (N = 3e5 point included).

Runtime: sweep ~20 min, N-scaling ~30 min (skip with --no-nscale).
Use --quick for a smoke-sized run (figures will be rough).
"""
import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gmm2d import (FINE, FXY, GAUSS_BED, GMM_BED, Zratio_grid,
                   gaussian_path_mgf, smc_cfg, stable_seed, tempered_on,
                   tv_to_target)

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "output")
C_G, C_M, C_EX = '#2166ac', '#b2182b', '#4393c3'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-nscale", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    if args.quick:
        NS, MS = 6_000, 300
        WS = [0.5, 1.0, 1.5, 2.0]
        WGRID = np.linspace(0.5, 3.0, 9)
    else:
        NS, MS = 60_000, 2_000
        WS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
        WGRID = np.linspace(0.5, 3.0, 26)

    # exact Gaussian reference curve and threshold
    ess_exact_g = []
    for w in WGRID:
        e1x, _ = gaussian_path_mgf(2.25, 0.25, 0.8, w, 1.0)
        e1y, _ = gaussian_path_mgf(2.25, 0.25, 0.0, w, 1.0)
        e2x, okx = gaussian_path_mgf(2.25, 0.25, 0.8, w, 2.0)
        e2y, oky = gaussian_path_mgf(2.25, 0.25, 0.0, w, 2.0)
        EW1 = e1x * e1y
        EW2 = e2x * e2y if (okx and oky) else np.inf
        ess_exact_g.append(EW1**2 / EW2 if np.isfinite(EW2) else 0.0)
    ess_exact_g = np.array(ess_exact_g)

    # main sweep
    results = {}
    for key, bed in (('G', GAUSS_BED), ('M', GMM_BED)):
        for w in WS:
            t0 = time.time()
            r = smc_cfg(bed, w=w, N=NS, M=MS, seed=stable_seed("sweep", key, w))
            r['Z0Z1'] = Zratio_grid(bed, w)
            r['tv_w'] = tv_to_target(r['x0'], r['wn'], bed, w)
            r['tv_n'] = tv_to_target(r['x0'], np.ones(len(r['x0'])) / len(r['x0']),
                                     bed, w)
            results[(key, w)] = r
            print(f"{key} w={w}: ESS/N={r['ess']:.4f} E[W]={r['EW']:.4f} "
                  f"(exact {r['Z0Z1']:.4f}) TVw={r['tv_w']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # N-scaling (phase-transition diagnostic)
    nscale = {}
    if not args.no_nscale and not args.quick:
        for N in (30_000, 100_000, 300_000):      # no early break: 3e5 included
            for w in (1.5, 1.6, 1.7, 1.75):
                rr = smc_cfg(GMM_BED, w=w, N=N, M=MS,
                             seed=stable_seed("NS", N, w))
                nscale[(N, w)] = rr['ess']
                print(f"NS GMM N={N} w={w}: ESS/N={rr['ess']:.4f}", flush=True)
        for N, w in ((240_000, 1.75), (240_000, 2.0)):
            rr = smc_cfg(GAUSS_BED, w=w, N=N, M=MS, seed=stable_seed("GC", N, w))
            nscale[(N, w)] = rr['ess']
            print(f"NS GAUSS N={N} w={w}: ESS/N={rr['ess']:.4f}", flush=True)
        for s in (101, 102):     # replicate seeds for the w=1.5 plateau
            rr = smc_cfg(GMM_BED, w=1.5, N=100_000, M=MS, seed=s)
            nscale[(100_000, f"1.5_s{s}")] = rr['ess']

    # CSV
    with open(os.path.join(OUTDIR, 'gmm2d_sweep.csv'), 'w', newline='') as f:
        wtr = csv.writer(f)
        wtr.writerow(['bed', 'w', 'ESS_over_N', 'E[W]_MC', 'E[W]_exact_Z0/Z1',
                      'SD_logW', 'lognormal_proxy', 'TV_corrected', 'TV_naive',
                      'ESS_t=0.75', 'ESS_t=0.50', 'ESS_t=0.25', 'N', 'M'])
        for key, name in (('G', 'Gaussian'), ('M', 'GMM')):
            for w in WS:
                r = results[(key, w)]
                sn = r['ess_snap']
                wtr.writerow([name, w, f"{r['ess']:.5f}", f"{r['EW']:.5f}",
                              f"{r['Z0Z1']:.5f}", f"{r['sdlogW']:.5f}",
                              f"{np.exp(-r['sdlogW']**2):.5f}",
                              f"{r['tv_w']:.5f}", f"{r['tv_n']:.5f}",
                              f"{sn.get(0.75, float('nan')):.5f}",
                              f"{sn.get(0.5, float('nan')):.5f}",
                              f"{sn.get(0.25, float('nan')):.5f}", NS, MS])

    # ----------------------------- figures ------------------------------------
    essG = [results[('G', w)]['ess'] for w in WS]
    essM = [results[('M', w)]['ess'] for w in WS]
    proxyG = [np.exp(-results[('G', w)]['sdlogW'] ** 2) for w in WS]
    proxyM = [np.exp(-results[('M', w)]['sdlogW'] ** 2) for w in WS]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    ax = axes[0]
    ax.semilogy(WS, proxyG, '--', color=C_G, lw=1.1, alpha=.7,
                label='lognormal proxy (Gaussian)')
    ax.semilogy(WS, proxyM, '--', color=C_M, lw=1.1, alpha=.7,
                label='lognormal proxy (GMM)')
    ax.semilogy(WGRID, np.maximum(ess_exact_g, 1e-4), '-', color=C_EX, lw=2,
                label='Gaussian exact $(\\mathbb{E}W)^2/\\mathbb{E}[W^2]$')
    ax.semilogy(WS, essG, 'o', color=C_G, ms=6, mfc='white', mew=1.6,
                label=f'Gaussian SMC ($N{{=}}{NS:.0e}$)')
    ax.semilogy(WS, essM, 's', color=C_M, ms=6, mfc='white', mew=1.6,
                label=f'GMM SMC ($N{{=}}{NS:.0e}$)')
    ax.axvline(1.9539, color=C_G, ls=':', lw=1.4)
    ax.axvline(1.55, color=C_M, ls=':', lw=1.4)
    ax.text(1.9539, 1.6, ' $w^*$ Gaussian $=1.95$', color=C_G, fontsize=8.5,
            ha='left', va='top', rotation=90)
    ax.text(1.55, 1.6, ' $w^*$ GMM $\\approx1.55$', color=C_M, fontsize=8.5,
            ha='right', va='top', rotation=90)
    ax.set_xlabel('guidance weight $w$')
    ax.set_ylabel('terminal ESS/$N$')
    ax.set_ylim(1e-3, 2.2); ax.set_xlim(0.4, 3.1)
    ax.legend(fontsize=7.6, loc='lower left', framealpha=.95)
    ax.set_title('(a) Weight degeneracy: GMM collapses earlier', fontsize=9.5)
    ax = axes[1]
    ax.plot(WS, [results[('G', w)]['EW'] for w in WS], 'o', color=C_G, ms=6,
            mfc='white', mew=1.6, label='Gaussian SMC')
    ax.plot(WS, [results[('G', w)]['Z0Z1'] for w in WS], '-', color=C_G, lw=1.6,
            alpha=.75, label='Gaussian exact $\\hat Z_0/\\hat Z_1$')
    ax.plot(WS, [results[('M', w)]['EW'] for w in WS], 's', color=C_M, ms=6,
            mfc='white', mew=1.6, label='GMM SMC')
    ax.plot(WS, [results[('M', w)]['Z0Z1'] for w in WS], '-', color=C_M, lw=1.6,
            alpha=.75, label='GMM exact $\\hat Z_0/\\hat Z_1$')
    ax.set_xlabel('guidance weight $w$')
    ax.set_ylabel('$\\mathbb{E}[W]=\\hat Z_0/\\hat Z_1$')
    ax.set_yscale('log'); ax.legend(fontsize=7.6, loc='upper left')
    ax.set_xlim(0.4, 3.1)
    ax.set_title('(b) Partition ratio: SMC unbiased below $w^*$', fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'gmm2d_degeneracy.png'), dpi=150)
    fig.savefig(os.path.join(OUTDIR, 'gmm2d_degeneracy.pdf'))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    ax = axes[0]
    nsGmm = {1.5: ('#1b7837', [(30e3, nscale.get((30000, 1.5))),
                               (60e3, results[('M', 1.5)]['ess']),
                               (100e3, nscale.get((100000, 1.5))),
                               (300e3, nscale.get((300000, 1.5)))],
                   'GMM $w{=}1.5$ (below $w^*$)'),
             1.6: ('#b2182b', [(30e3, nscale.get((30000, 1.6))),
                               (100e3, nscale.get((100000, 1.6)))],
                   'GMM $w{=}1.6$ (above $w^*$)'),
             1.7: ('#e08214', [(30e3, nscale.get((30000, 1.7))),
                               (100e3, nscale.get((100000, 1.7)))],
                   'GMM $w{=}1.7$'),
             1.75: ('#762a83', [(30e3, nscale.get((30000, 1.75))),
                                (100e3, nscale.get((100000, 1.75)))],
                    'GMM $w{=}1.75$')}
    for w, (c, pts, lab) in nsGmm.items():
        pts = [(N, e) for N, e in pts if e is not None]
        if pts:
            Ns, Es = zip(*pts)
            ax.plot(Ns, Es, '-o', color=c, lw=1.9, ms=5.5, label=lab)
    for N, w, mk, lab in ((240e3, 1.75, 'o', 'Gaussian $w{=}1.75$ (below $w^*$)'),
                          (240e3, 2.0, 'D', 'Gaussian $w{=}2.0$ (above, exact $=0$)')):
        if (N, w) in nscale and ('G', w) in results:
            ax.plot([60e3, N], [results[('G', w)]['ess'], nscale[(N, w)]], '--',
                    color='#666666', lw=1.3, marker=mk, ms=5, mfc='white',
                    label=lab)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('particle count $N$'); ax.set_ylabel('terminal ESS/$N$')
    ax.legend(fontsize=7.4, loc='lower left')
    ax.set_title('(a) Finite-$N$ scaling: plateau vs collapse', fontsize=9.5)
    ax = axes[1]
    tvGw = [results[('G', w)]['tv_w'] for w in WS]
    tvGn = [results[('G', w)]['tv_n'] for w in WS]
    tvMw = [results[('M', w)]['tv_w'] for w in WS]
    tvMn = [results[('M', w)]['tv_n'] for w in WS]
    ax.plot(WS, tvGn, 'o--', color=C_G, lw=1.2, mfc='white', ms=5.5,
            label='Gaussian naive')
    ax.plot(WS, tvGw, 'o-', color=C_G, lw=1.8, ms=5.5, label='Gaussian corrected')
    ax.plot(WS, tvMn, 's--', color=C_M, lw=1.2, mfc='white', ms=5.5,
            label='GMM naive')
    ax.plot(WS, tvMw, 's-', color=C_M, lw=1.8, ms=5.5, label='GMM corrected')
    ax.set_xlabel('guidance weight $w$')
    ax.set_ylabel('TV to exact tempered target')
    ax.set_ylim(0, 0.30); ax.legend(fontsize=7.8, loc='upper left')
    ax.axvline(1.9539, color=C_G, ls=':', lw=1.2)
    ax.axvline(1.55, color=C_M, ls=':', lw=1.2)
    ax.set_title('(b) Exactness: corrected sampler hits the target, naive drifts',
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'gmm2d_scaling_tv.png'), dpi=150)
    fig.savefig(os.path.join(OUTDIR, 'gmm2d_scaling_tv.pdf'))
    plt.close(fig)

    def hdr_contours(ax, q2d, levels=(0.5, 0.9, 0.99), color='k'):
        q = q2d.ravel()
        order = np.argsort(q)[::-1]
        cq = np.cumsum(q[order]) / q.sum()
        thr = sorted(q[order][np.searchsorted(cq, p)] for p in levels)
        ax.contour(FINE, FINE, q2d.T, levels=thr, colors=color,
                   linewidths=[1.6, 1.1, 0.7])

    dens_ws = (0.5, 1.5, 2.0) if all(('M', w) in results for w in (0.5, 1.5, 2.0)) \
        else tuple(WS[:3])
    fig, axes = plt.subplots(2, len(dens_ws), figsize=(11.5, 7.4))
    for j, w in enumerate(dens_ws):
        r = results[('M', w)]
        q = tempered_on(FXY, GMM_BED, w, 0.0).reshape(len(FINE), len(FINE))
        for row, (wn, ttl) in enumerate(
                [(np.ones(len(r['x0'])) / len(r['x0']), 'naive (uniform weights)'),
                 (r['wn'], 'corrected (path-product tilt)')]):
            ax = axes[row, j]
            hdr_contours(ax, q)
            ax.plot(GMM_BED['mc'][:, 0], GMM_BED['mc'][:, 1], '+',
                    color='dimgrey', ms=7, mew=1.6)
            rng = np.random.default_rng(4)
            if row == 0:
                idx = rng.choice(len(r['x0']), min(9000, len(r['x0'])),
                                 replace=False)
                ax.scatter(r['x0'][idx, 0], r['x0'][idx, 1], s=3, c='#2166ac',
                           alpha=0.30, edgecolors='none')
            else:
                idx = rng.choice(len(r['x0']), min(12000, len(r['x0'])),
                                 p=r['wn'])
                ax.scatter(r['x0'][idx, 0], r['x0'][idx, 1], s=3, c='#b2182b',
                           alpha=0.30, edgecolors='none')
            tv = tv_to_target(r['x0'], wn, GMM_BED, w)
            ax.set_title(f'$w={w}$   {ttl}\nTV $= {tv:.3f}$', fontsize=9)
            ax.set_xlim(-3.4, 3.4); ax.set_ylim(-3.4, 3.4); ax.set_aspect('equal')
            if row == 1: ax.set_xlabel('$x_1$')
            if j == 0: ax.set_ylabel('$x_2$')
    fig.suptitle('GMM experts: tempered target $q_w\\propto(p^0)^{1-w}(p^c)^w$ '
                 '(HDR contours) vs sampler output', fontsize=10.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(os.path.join(OUTDIR, 'gmm2d_densities.png'), dpi=150)
    fig.savefig(os.path.join(OUTDIR, 'gmm2d_densities.pdf'))
    plt.close(fig)
    print('all figures + CSV written to', os.path.abspath(OUTDIR))


if __name__ == '__main__':
    main()
