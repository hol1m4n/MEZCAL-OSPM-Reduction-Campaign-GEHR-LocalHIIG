import sys
import glob
import csv
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # safe for batch use; remove if interactive
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from astropy.io import fits
from scipy.optimize import curve_fit
import argparse

warnings.filterwarnings("ignore", category=RuntimeWarning)


from pyraf import iraf
from pyraf import gki

import ast
import os


iraf.noao()
iraf.twodspec()
iraf.apextract()



def read_iraf_spectrum(filepath):
    """
    Return (wavelength, flux) arrays for an IRAF-reduced 1-D FITS spectrum.

    Handles:
      • Simple 1-D images  (NAXIS=1)
      • 2-D/3-D arrays with degenerate axes (squeezed to 1-D)
      • LINEAR WCS via CRVAL1 / CDELT1 (or CD1_1) / CRPIX1
    """
    with fits.open(filepath) as hdul:
        hdr  = hdul[0].header
        data = hdul[0].data

    flux = np.asarray(data, dtype=float).squeeze()
    if flux.ndim > 1:
        flux = flux[0]          # take first aperture if multi-aperture

    npts  = len(flux)
    crval = float(hdr.get("CRVAL1", 1.0))
    cdelt = float(hdr.get("CDELT1", hdr.get("CD1_1", 1.0)))
    crpix = float(hdr.get("CRPIX1", 1.0))

    pixels = np.arange(1, npts + 1, dtype=float)
    wave   = crval + (pixels - crpix) * cdelt

    return wave, flux


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Profile models
# ─────────────────────────────────────────────────────────────────────────────

def gaussian(x, amp, center, sigma, cont):
    """
    Single Gaussian with a flat continuum.
    f(x) = cont + amp * exp[ -0.5 * ((x-center)/sigma)² ]
    """
    w = (x - center) / sigma
    return cont + amp * np.exp(-0.5 * w**2)


def gauss_hermite(x, amp, center, sigma, h3, h4, cont):
    """
    Gauss-Hermite series (van der Marel & Franx 1993, eq. 2-3; Riffel 2010).

    f(x) = cont + amp * exp[-0.5*w²] * [1 + h3*H3(w) + h4*H4(w)]

    where  w  = (x - center) / sigma

    Orthonormal Hermite polynomials (probabilist's convention):
        H3(w) = (2w³ − 3w) / √6
        H4(w) = (4w⁴ − 12w² + 3) / √24

    h3  quantifies skewness of the profile (asymmetric wings).
    h4  quantifies kurtosis / leptokurtosis (broader/narrower peak).
    When h3 = h4 = 0 the function reduces to a pure Gaussian.
    """
    w  = (x - center) / sigma
    H3 = (2.0 * w**3 - 3.0 * w) / np.sqrt(6.0)
    H4 = (4.0 * w**4 - 12.0 * w**2 + 3.0) / np.sqrt(24.0)
    return cont + amp * np.exp(-0.5 * w**2) * (1.0 + h3 * H3 + h4 * H4)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FWHM calculation
# ─────────────────────────────────────────────────────────────────────────────

def fwhm_gaussian(sigma):
    """Analytic FWHM for a pure Gaussian."""
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(sigma)


def fwhm_gauss_hermite(sigma, h3, h4, npts=20_000):
    """
    Numerical FWHM for a Gauss-Hermite profile sampled over ±8σ.
    Falls back to the Gaussian value if the profile does not cross half-max.
    """
    x  = np.linspace(-8.0 * sigma, 8.0 * sigma, npts)
    w  = x / sigma
    H3 = (2.0 * w**3 - 3.0 * w) / np.sqrt(6.0)
    H4 = (4.0 * w**4 - 12.0 * w**2 + 3.0) / np.sqrt(24.0)
    y  = np.exp(-0.5 * w**2) * (1.0 + h3 * H3 + h4 * H4)

    peak      = np.max(y)
    half_max  = 0.5 * peak
    above     = y >= half_max
    crossings = np.where(np.diff(above.astype(int)))[0]

    if len(crossings) >= 2:
        # Linear interpolation to sub-pixel precision at each crossing
        def interp_crossing(idx, rising):
            if rising:
                x0, x1 = x[idx], x[idx + 1]
                y0, y1 = y[idx], y[idx + 1]
            else:
                x0, x1 = x[idx + 1], x[idx]
                y0, y1 = y[idx + 1], y[idx]
            return x0 + (half_max - y0) * (x1 - x0) / (y1 - y0)

        left  = interp_crossing(crossings[0],  rising=True)
        right = interp_crossing(crossings[-1], rising=False)
        return right - left
    else:
        return fwhm_gaussian(sigma)   # safe fallback


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Reduced chi-squared
# ─────────────────────────────────────────────────────────────────────────────

def reduced_chi2(flux_obs, flux_model, sigma_noise, n_free):
    """χ²_ν = Σ[(obs-model)²/σ²] / (N - n_free)"""
    residuals = flux_obs - flux_model
    chi2      = np.sum((residuals / sigma_noise) ** 2)
    dof       = max(len(flux_obs) - n_free, 1)
    return chi2 / dof


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Monte Carlo error estimation
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo_errors(wave_win, flux_win, popt_g, popt_gh,
                       sigma_noise, n_iter=1000):
    """
    Perturb the spectrum with Gaussian noise (σ = sigma_noise) n_iter times
    and refit both profiles.  Returns distributions of all best-fit parameters
    and FWHM values, from which means and standard deviations are derived.
    """
    fwhm_g_mc   = []
    fwhm_gh_mc  = []
    params_g_mc  = []
    params_gh_mc = []

    rng = np.random.default_rng()

    for _ in range(n_iter):
        noisy = flux_win + rng.normal(0.0, sigma_noise, size=len(flux_win))

        # — Gaussian refit —
        try:
            p, _ = curve_fit(gaussian, wave_win, noisy, p0=popt_g,
                             maxfev=6000)
            params_g_mc.append(p)
            fwhm_g_mc.append(fwhm_gaussian(p[2]))
        except (RuntimeError, ValueError):
            pass

        # — Gauss-Hermite refit —
        try:
            p, _ = curve_fit(
                gauss_hermite, wave_win, noisy, p0=popt_gh,
                bounds=(
                    [0,     -np.inf, 0.05, -0.5, -0.5, -np.inf],
                    [np.inf, np.inf, 15.0,  0.5,  0.5,  np.inf],
                ),
                maxfev=6000,
            )
            params_gh_mc.append(p)
            fwhm_gh_mc.append(fwhm_gauss_hermite(abs(p[2]), p[3], p[4]))
        except (RuntimeError, ValueError):
            pass

    params_g_mc  = np.array(params_g_mc)
    params_gh_mc = np.array(params_gh_mc)
    fwhm_g_mc    = np.array(fwhm_g_mc)
    fwhm_gh_mc   = np.array(fwhm_gh_mc)

    return {
        "gaussian": {
            "params_mc" : params_g_mc,
            "fwhm_mc"   : fwhm_g_mc,
            "fwhm_mean" : np.mean(fwhm_g_mc)  if len(fwhm_g_mc)  else np.nan,
            "fwhm_std"  : np.std(fwhm_g_mc)   if len(fwhm_g_mc)  else np.nan,
        },
        "gauss_hermite": {
            "params_mc" : params_gh_mc,
            "fwhm_mc"   : fwhm_gh_mc,
            "fwhm_mean" : np.mean(fwhm_gh_mc) if len(fwhm_gh_mc) else np.nan,
            "fwhm_std"  : np.std(fwhm_gh_mc)  if len(fwhm_gh_mc) else np.nan,
        },
        "sigma_noise": sigma_noise,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Publication-style figure
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(wave_full, flux_full,
                wave_win, flux_win,
                popt_g, popt_gh,
                fwhm_g_mean, fwhm_g_err,
                fwhm_gh_mean, fwhm_gh_err,
                chi2_g, chi2_gh,
                mc, label,
                save_path=None):

    wave_fine = np.linspace(wave_win[0], wave_win[-1], 2000)
    fit_g_fine  = gaussian(wave_fine, *popt_g)
    fit_gh_fine = gauss_hermite(wave_fine, *popt_gh)

    res_g  = flux_win - gaussian(wave_win, *popt_g)
    res_gh = flux_win - gauss_hermite(wave_win, *popt_gh)

    fig = plt.figure(figsize=(11, 10))
    fig.patch.set_facecolor("white")

    # Grid:  row0 = main spectrum, row1 = residuals, row2 = MC histograms
    outer = gridspec.GridSpec(3, 1, figure=fig,
                              height_ratios=[3.5, 1.0, 2.2],
                              hspace=0.2)
    ax_main = fig.add_subplot(outer[0])
    ax_res  = fig.add_subplot(outer[1], sharex=ax_main)
    gs_hist = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2],
                                               wspace=0.35)
    ax_hg   = fig.add_subplot(gs_hist[0])
    ax_hgh  = fig.add_subplot(gs_hist[1])

    sigma_noise = mc["sigma_noise"]

    # ── Main panel ──────────────────────────────────────────────────────────
    ax_main.errorbar(wave_win, flux_win, yerr=sigma_noise,
                     fmt="k.", ms=0.1, elinewidth=0.7, capsize=0,
                      zorder=2)
    ax_main.step(wave_win, flux_win,lw=0.5,color='k',label="Data")
    ax_main.plot(wave_fine, fit_g_fine,  "b--", lw=1.0, label="Gaussian",      zorder=4, alpha=0.5)
    ax_main.plot(wave_fine, fit_gh_fine, "r-",  lw=1.2, label="Gauss-Hermite", zorder=5, alpha=0.5)
    ax_main.axhline(popt_g[-1], color="green", ls="--", lw=1.2,
                    label="Continuum", zorder=3)
    ax_main.axhline(0, color="gray", ls=":", lw=0.8, zorder=1)

    # Inset annotation box
    txt = (
        f"Center$_C$ = {popt_g[1]:.3f} Å\n"
        f"Center$_{{GH}}$ = {popt_gh[1]:.3f} Å\n"
        f"FWHM$_C$ = {fwhm_g_mean:.3f} $\\pm$ {fwhm_g_err:.4f} Å\n"
        f"FWHM$_{{GH}}$ = {fwhm_gh_mean:.3f} $\\pm$ {fwhm_gh_err:.4f} Å\n"
        f"$\\chi^2_{{\\nu,C}}$ = {chi2_g:.3f}\n"
        f"$\\chi^2_{{\\nu,GH}}$ = {chi2_gh:.3f}"
    )
    ax_main.text(0.03, 0.97, txt, transform=ax_main.transAxes,
                 va="top", ha="left", fontsize=9.5, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85, ec="gray"))

    ax_main.set_title(label, fontsize=12, fontweight="bold")
    ax_main.set_ylabel("Counts", fontsize=11)
    ax_main.legend(fontsize=9.5, loc="upper right", framealpha=0.9)
    ax_main.tick_params(labelbottom=False)

    # ── Residuals panel ──────────────────────────────────────────────────────
    ax_res.scatter(wave_win, res_g,  c="blue",   marker="+",  s=30,
                   lw=1.2, label="Gaussian",      zorder=3)
    ax_res.scatter(wave_win, res_gh, c="red",    marker="x",  s=30,
                   lw=1.2, label="Gauss-Hermite", zorder=3)
    ax_res.axhline(0, color="k", lw=1.0)
    ax_res.axhline( sigma_noise, color="gray", ls="--", lw=0.8)
    ax_res.axhline(-sigma_noise, color="gray", ls="--", lw=0.8)
    ax_res.set_ylabel("Residuals", fontsize=9)
    ax_res.set_xlabel("λ (Å)", fontsize=11)
    ax_res.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.8)

    # ── Monte Carlo histograms ────────────────────────────────────────────────
    for ax, key, fwhm_mean, fwhm_err, color, title in [
        (ax_hg,  "gaussian",      fwhm_g_mean,  fwhm_g_err,  "#3a7fd5", "Gaussian"),
        (ax_hgh, "gauss_hermite", fwhm_gh_mean, fwhm_gh_err, "#e05040", "Gauss-Hermite"),
    ]:
        fwhm_arr = mc[key]["fwhm_mc"]
        if len(fwhm_arr) == 0:
            ax.text(0.5, 0.5, "No MC data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        ax.hist(fwhm_arr, bins=30, color=color, alpha=0.75,
                edgecolor="k", linewidth=0.4)
        ax.axvline(fwhm_mean, color="black", lw=2.0, ls="-",
                   label=f"μ = {fwhm_mean:.4f} Å")
        ax.axvline(fwhm_mean - fwhm_err, color="gray", lw=1.2, ls="--")
        ax.axvline(fwhm_mean + fwhm_err, color="gray", lw=1.2, ls="--",
                   label=f"σ = {fwhm_err:.4f} Å")
        ax.set_xlabel("FWHM (Å)", fontsize=10)
        ax.set_ylabel("N", fontsize=10)
        ax.set_title(f"Monte Carlo – {title}", fontsize=10)
        ax.legend(fontsize=8.5, framealpha=0.9)

    if save_path:
        plt.savefig(save_path, dpi=80, bbox_inches="tight")
        print(f"    Figure → {save_path} \n")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main fitting routine for one spectrum
# ─────────────────────────────────────────────────────────────────────────────

def fit_halpha(filepath,
               ha_center=6562.80,
               window_A=35.0,
               n_mc=1000,
               save_fig=True):
    """
    Full pipeline for one FITS spectrum.

    Parameters
    ----------
    filepath  : str
        Path to IRAF-reduced 1-D FITS file.
    ha_center : float
        Expected observed wavelength of Hα in Å.
        = 6562.80 * (1 + z) for a galaxy at redshift z.
    window_A  : float
        Half-width in Å of the fitting window around ha_center.
    n_mc      : int
        Number of Monte Carlo iterations.
    save_fig  : bool
        If True, save figure as <stem>_halpha_fit.png.

    Returns
    -------
    dict with all best-fit parameters, FWHM values, χ², and MC results.
    """

    label = Path(filepath).stem
    print(f"\n{'═'*60}")
    print(f"  {label}")
    print(f"{'═'*60}")

    # ─ Read ────────────────────────────────────────────────────────────────
    wave, flux = read_iraf_spectrum(filepath)

    # ─ Extract window ───────────────────────────────────────────────────────
    mask     = (wave >= ha_center - window_A) & (wave <= ha_center + window_A)
    wave_win = wave[mask]
    flux_win = flux[mask]

    if len(wave_win) < 15:
        raise ValueError(
            f"  Only {len(wave_win)} pixels in window [{ha_center-window_A:.1f}, "
            f"{ha_center+window_A:.1f}] Å.\n"
            f"  Check ha_center / window_A or the wavelength calibration."
        )

    # ─ Estimate continuum & noise from the window edges ────────────────────
    n_edge   = max(5, int(0.15 * len(wave_win)))
    edge_pix = np.concatenate([flux_win[:n_edge], flux_win[-n_edge:]])
    cont0    = np.median(edge_pix)
    noise0   = np.std(edge_pix)          # initial noise estimate from continuum

    # ─ Initial guesses ──────────────────────────────────────────────────────
    amp0    = np.max(flux_win) - cont0
    cen0    = wave_win[np.argmax(flux_win)]
    # Estimate σ from half-power points
    above_half = flux_win > (cont0 + 0.5 * amp0)
    sig0 = (np.sum(above_half) * np.median(np.diff(wave_win))) / 2.355
    sig0 = np.clip(sig0, 0.2, 5.0)

    # ─ Gaussian fit ─────────────────────────────────────────────────────────
    p0_g = [amp0, cen0, sig0, cont0]
    bounds_g = (
        [0,      ha_center - 8,  0.05, -np.inf],
        [np.inf, ha_center + 8,  15.0,  np.inf],
    )
    try:
        popt_g, _ = curve_fit(gaussian, wave_win, flux_win,
                              p0=p0_g, bounds=bounds_g, maxfev=10_000)
    except RuntimeError as e:
        raise RuntimeError(f"  Gaussian fit failed: {e}")

    fwhm_g_best = fwhm_gaussian(popt_g[2])

    # Refine noise from Gaussian residuals (better than edge estimate
    # if the continuum is curved)
    res_g_best  = flux_win - gaussian(wave_win, *popt_g)
    sigma_noise = np.std(res_g_best[~(np.abs(res_g_best) > 5 * noise0)])
    sigma_noise = max(sigma_noise, noise0)   # never underestimate

    chi2_g = reduced_chi2(flux_win, gaussian(wave_win, *popt_g),
                          sigma_noise, n_free=len(popt_g))

    # ─ Gauss-Hermite fit ────────────────────────────────────────────────────
    p0_gh = [popt_g[0], popt_g[1], popt_g[2], 0.0, 0.0, popt_g[3]]
    bounds_gh = (
        [0,      ha_center - 8,  0.05, -0.5, -0.5, -np.inf],
        [np.inf, ha_center + 8,  15.0,  0.5,  0.5,  np.inf],
    )
    try:
        popt_gh, _ = curve_fit(gauss_hermite, wave_win, flux_win,
                               p0=p0_gh, bounds=bounds_gh, maxfev=10_000)
    except RuntimeError as e:
        raise RuntimeError(f"  Gauss-Hermite fit failed: {e}")

    fwhm_gh_best = fwhm_gauss_hermite(abs(popt_gh[2]), popt_gh[3], popt_gh[4])
    chi2_gh = reduced_chi2(flux_win, gauss_hermite(wave_win, *popt_gh),
                           sigma_noise, n_free=len(popt_gh))

    # ─ Monte Carlo errors ───────────────────────────────────────────────────
    print(f"  Running {n_mc} MC iterations …", end=" ", flush=True)
    mc = monte_carlo_errors(wave_win, flux_win,
                            popt_g, popt_gh,
                            sigma_noise=sigma_noise,
                            n_iter=n_mc)
    print("done.")

    fwhm_g_mean  = mc["gaussian"]["fwhm_mean"]
    fwhm_g_err   = mc["gaussian"]["fwhm_std"]
    fwhm_gh_mean = mc["gauss_hermite"]["fwhm_mean"]
    fwhm_gh_err  = mc["gauss_hermite"]["fwhm_std"]

    # ─ Print summary ────────────────────────────────────────────────────────
    w = 40
    '''
    print(f"\n  {'── GAUSSIAN FIT ':─<{w}}")
    print(f"    Center         = {popt_g[1]:.3f} Å")
    print(f"    σ              = {abs(popt_g[2]):.4f} Å")
    print(f"    FWHM (MC)      = {fwhm_g_mean:.4f} ± {fwhm_g_err:.4f} Å")
    print(f"    χ²_ν           = {chi2_g:.3f}")

    print(f"\n  {'── GAUSS-HERMITE FIT ':─<{w}}")
    print(f"    Center         = {popt_gh[1]:.3f} Å")
    print(f"    σ              = {abs(popt_gh[2]):.4f} Å")
    print(f"    h3             = {popt_gh[3]:.4f}")
    print(f"    h4             = {popt_gh[4]:.4f}")
    print(f"    FWHM (MC)      = {fwhm_gh_mean:.4f} ± {fwhm_gh_err:.4f} Å")
    print(f"    χ²_ν           = {chi2_gh:.3f}")

    print(f"\n  Noise σ (continuum) = {sigma_noise:.4f}  counts")
    print(f"  MC success  G / GH  = "
          f"{len(mc['gaussian']['fwhm_mc'])} / "
          f"{len(mc['gauss_hermite']['fwhm_mc'])} / {n_mc}")
    '''
          

    # ─ Figure ───────────────────────────────────────────────────────────────
    fig_path = str(Path(filepath).with_suffix("")) + "_halpha_fit.png" \
               if save_fig else None
    make_figure(
        wave, flux, wave_win, flux_win,
        popt_g, popt_gh,
        fwhm_g_mean, fwhm_g_err,
        fwhm_gh_mean, fwhm_gh_err,
        chi2_g, chi2_gh,
        mc, label,
        save_path=fig_path,
    )

    return {
        "label"        : label,
        "filepath"     : str(filepath),
        # Gaussian
        "amp_G"        : popt_g[0],
        "center_G"     : popt_g[1],
        "sigma_G"      : abs(popt_g[2]),
        "cont_G"       : popt_g[3],
        "fwhm_G"       : fwhm_g_mean,
        "e_fwhm_G"     : fwhm_g_err,
        "chi2_G"       : chi2_g,
        # Gauss-Hermite
        "amp_GH"       : popt_gh[0],
        "center_GH"    : popt_gh[1],
        "sigma_GH"     : abs(popt_gh[2]),
        "h3"           : popt_gh[3],
        "h4"           : popt_gh[4],
        "cont_GH"      : popt_gh[5],
        "fwhm_GH"      : fwhm_gh_mean,
        "e_fwhm_GH"    : fwhm_gh_err,
        "chi2_GH"      : chi2_gh,
        # Diagnostics
        "sigma_noise"  : sigma_noise,
        "mc"           : mc,
        # Raw fits
        "popt_g"       : popt_g,
        "popt_gh"      : popt_gh,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Batch processing + CSV output
# ─────────────────────────────────────────────────────────────────────────────

CSV_HEADER = [
    "label",
    "center_G(A)",  "sigma_G(A)",  "FWHM_G(A)",  "e_FWHM_G(A)",  "chi2nu_G",
    "center_GH(A)", "sigma_GH(A)", "h3",          "h4",
    "FWHM_GH(A)",   "e_FWHM_GH(A)", "chi2nu_GH",
    "sigma_noise",
]

def fmt(v, d=4):
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return str(v)


def batch_fit(file_list,
              ha_center  = 6562.80,
              window_A   = 35.0,
              n_mc       = 1000,
              output_csv = "halpha_fwhm_results.csv",
              save_figs  = True):
    """
    Fit all spectra in file_list and write a CSV summary.
    """
    all_results = []

    for fp in file_list:
        try:
            res = fit_halpha(
                fp,
                ha_center = ha_center,
                window_A  = window_A,
                n_mc      = n_mc,
                save_fig  = save_figs,
            )
            all_results.append(res)
        except Exception as exc:
            print(f"\n  ✗  ERROR – {fp}: {exc}")

    # Write CSV
    with open(output_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        for r in all_results:
            writer.writerow([
                r["label"],
                fmt(r["center_G"],  3), fmt(r["sigma_G"],  4),
                fmt(r["fwhm_G"],    4), fmt(r["e_fwhm_G"], 4),
                fmt(r["chi2_G"],    3),
                fmt(r["center_GH"], 3), fmt(r["sigma_GH"], 4),
                fmt(r["h3"],        4), fmt(r["h4"],        4),
                fmt(r["fwhm_GH"],   4), fmt(r["e_fwhm_GH"],4),
                fmt(r["chi2_GH"],   3),
                fmt(r["sigma_noise"],4),
            ])

    print(f"\n{'─'*60}")
    print(f"  Results saved → {output_csv}")
    print(f"  Spectra processed: {len(all_results)} / {len(file_list)}")

    return all_results


def arc_extract_1d(file_arc,spec_ref):

    iraf.apall(
        input =f"{file_arc}"                 , #List of input images
        nfind =                 1, #Number of apertures to be found automatically
        output = f"{file_arc.replace('wlcal','1d')}"             , #List of output spectra
        apertures = ""            , #Apertures
        format = "multispec"    , #Extracted spectra format
        references = f"{spec_ref}"             , #List of aperture reference images
        profiles = ""             , #List of aperture profile images\n
        interactive = "no"            , #Run task interactively?
        find = "no"            , #Find apertures?
        recenter = "no"            , #Recenter apertures?
        resize = "no"            , #Resize apertures?
        edit = "no"            , #Edit apertures?
        trace = "no"            , #Trace apertures?
        fittrace = "no"            , #Fit the traced points interactively?
        extract = "yes"            , #Extract spectra?
        extras = "no"            , #Extract sky, sigma, etc.?
        review = "no"            , #Review extractions?\n
        line = "INDEF"          , #Dispersion line
        nsum = 500, #500             , #Number of dispersion lines to sum or median\n\n# DEFAULT APERTURE PARAMETERS\n
        lower = -5.0            , #Lower aperture limit relative to center
        upper = 5.0             , #Upper aperture limit relative to center
        apidtable = ""             , #Aperture ID table (optional)\n\n# DEFAULT BACKGROUND PARAMETERS\n
        b_function = "chebyshev"    , #Background function
        b_order = 2              , #Background function order
        b_sample = "-180:-100,80:170"  , #Background sample regions
        b_naverage = -4             , #Background average or median
        b_niterate = 1              , #Background rejection iterations
        b_low_reject = 3.0             , #Background lower rejection sigma
        b_high_rejec = 3.0             , #Background upper rejection sigma
        b_grow = 0.0             , #Background rejection growing radius\n\n# APERTURE CENTERING PARAMETERS\n
        width = 5.0             , #Profile centering width
        radius = 10.0            , #Profile centering radius
        threshold = 10.0             , #Detection threshold for profile centering\n\n# AUTOMATIC FINDING AND ORDERING PARAMETERS\n
        minsep = 3.0             , #Minimum separation between spectra
        maxsep = 100000.0        , #Maximum separation between spectra
        order = "increasing"   , #Order of apertures\n\n# RECENTERING PARAMETERS\n
        aprecenter = ""             , #Apertures for recentering calculation
        npeaks = "INDEF"          , #Select brightest peaks
        shift = "no"            , #Use average shift instead of recentering?\n\n# RESIZING PARAMETERS\n
        llimit = "INDEF"          , #Lower aperture limit relative to center
        ulimit = "INDEF"          , #Upper aperture limit relative to center
        ylevel = 0.1            , #Fraction of peak or intensity for automatic width
        peak = "yes"            , #Is ylevel a fraction of the peak?
        bkg = "yes"            , #Subtract background in automatic width?
        r_grow = 0.0             , #Grow limits by this factor
        avglimits = "no"             , #Average limits over all apertures?\n\n# TRACING PARAMETERS\n
        t_nsum = 100             , #Number of dispersion lines to sum
        t_step = 10             , #Tracing step
        t_nlost = 3              , #Number of consecutive times profile is lost before quitting
        t_function = "legendre"     , #Trace fitting function
        t_order = 3              , #Trace fitting function order
        t_sample = "*"            , #Trace sample regions
        t_naverage = 1              , #Trace average or median
        t_niterate = 1              , #Trace rejection iterations
        t_low_reject = 2.5             , #Trace lower rejection sigma
        t_high_rejec = 2.5             , #Trace upper rejection sigma
        t_grow = 0.0             , #Trace rejection growing radius\n\n# EXTRACTION PARAMETERS\n
        background = "none"         , #Background to subtract
        skybox = 1              , #Box car smoothing length for sky
        weights = "none"         , #Extraction weights (none|variance)
        pfit = "fit1d"        , #Profile fitting type (fit1d|fit2d)
        clean = "no"             , #Detect and replace bad pixels?
        saturation = "INDEF"          , #Saturation level
        readnoise = "RDNOISE"           , #Read out noise sigma (photons)
        gain = "GAIN"           , #Photon gain (photons/data number)
        lsigma = 4.             , #Lower rejection threshold
        usigma = 4.             , #Upper rejection threshold
        nsubaps = 1              , #Number of subapertures per aperture
    )



def sigma_calculator_function(Host = '',
                            FWHM_G_Ha = 0,
                            centroide_G = 0,
                            centroide_GH = 0,
                            e_FWHM_G_Ha = 0,
                            FWHM_GH_Ha = 0,
                            e_FWHM_GH_Ha = 0,
                            FWHM_G_Th = 0,
                            e_FWHM_G_Th = 0,
                            sigma_thermal = 9.1,
                            sigma_finestructure = 2.4,
                            work_dir= ''):

    luz = 299792.458
    C_FWHM = 1/(2 * np.sqrt(2*np.log(2)))
    K = luz * C_FWHM

    sigma_G = 0 #Gaussian
    sigma_GH = 0 #Gauss-Hermite
    e_sigma_G = 0
    e_sigma_GH = 0
    sigma_Th = 0 # Instrumental
    e_sigma_Th = 0   

    sigma_T_G = 0
    sigma_T_GH = 0
    e_sigma_T_G = 0
    e_sigma_T_GH = 0


    ruta = f"{work_dir}Sigma_fit/sigma_results"

    sigma_G = FWHM_G_Ha * K/centroide_G
    sigma_GH = FWHM_GH_Ha * K/centroide_GH
    e_sigma_G = e_FWHM_G_Ha * K/centroide_G
    e_sigma_GH = e_FWHM_GH_Ha * K/centroide_GH
    sigma_Th = FWHM_G_Th * K/6583.906
    e_sigma_Th = e_FWHM_G_Th * K/6583.906 

    sigma_T_G = np.sqrt((sigma_G**2)-(sigma_Th**2)-(sigma_thermal**2)-(sigma_finestructure**2))
    sigma_T_GH = np.sqrt((sigma_GH**2)-(sigma_Th**2)-(sigma_thermal**2)-(sigma_finestructure**2))

    e_sigma_T_G = (1/(sigma_T_G)) * np.sqrt((((K/centroide_G)**2)*FWHM_G_Ha*e_FWHM_G_Ha)**2+(((K/6583.906)**2)*FWHM_G_Th*e_FWHM_G_Th)**2)
    e_sigma_T_GH = (1/(sigma_T_GH)) * np.sqrt((((K/centroide_GH)**2)*FWHM_GH_Ha*e_FWHM_GH_Ha)**2+(((K/6583.906)**2)*FWHM_G_Th*e_FWHM_G_Th)**2)

    with open(ruta, 'a') as f:
        f.write(f"{Host}	{FWHM_G_Ha}	{e_FWHM_G_Ha}	{sigma_G}	{e_sigma_G}	{FWHM_GH_Ha}	{e_FWHM_GH_Ha}	{sigma_GH}	{e_sigma_GH}	{FWHM_G_Th}	{e_FWHM_G_Th}	{sigma_Th}	{e_sigma_Th}	{sigma_thermal}	{sigma_finestructure}	{sigma_T_G}	{e_sigma_T_G}	{sigma_T_GH}	{e_sigma_T_GH}\n")

    print(f"{Host}	{FWHM_G_Ha}	{e_FWHM_G_Ha}	{sigma_G}	{e_sigma_G}	{FWHM_GH_Ha}	{e_FWHM_GH_Ha}	{sigma_GH}	{e_sigma_GH}	{FWHM_G_Th}	{e_FWHM_G_Th}	{sigma_Th}	{e_sigma_Th}	{sigma_thermal}	{sigma_finestructure}	{sigma_T_G}	{e_sigma_T_G}	{sigma_T_GH}	{e_sigma_T_GH}\n")





def sigma_runner(work_dir):


    sigma_start = input('Ya se existe/se movio el archivo de centroides a la carpeta Sigma_fit?(y/n):   ')
    if sigma_start == 'y':
        print('Continuamos...\n')  

    ruta = f"{work_dir}Sigma_fit/sigma_results"
    if not os.path.exists(ruta):
        print(f"Archivo no encontrado. Creando {ruta} para guardar valores... \n")
        with open(ruta, 'w') as f:
            pass




    centroides = [[],[]]

    with open(f'{work_dir}Sigma_fit/centroides', 'r') as f:
        for linea in f:
            linea = linea.strip()
            
            if linea:
                dupla = ast.literal_eval(linea)

                obj_selec = dupla[0] #[f for f in obj_files if dupla[0] in f][0]
                centro_selec = dupla[1] #[f for f in arc_files if dupla[1] in f][0]

                centroides[0].append(obj_selec)
                centroides[1].append(centro_selec)

    print(centroides,"\n")

    with open(f'{work_dir}OBJS/association', 'r') as f:
        for linea in f:
            linea = linea.strip()
            
            if linea:
                dupla = ast.literal_eval(linea)

                Objeto = dupla[0]
                Arco = dupla[1]

                for i in range(len(centroides[0])):
                    if centroides[0][i] in Objeto:


                        Info_sigma_spec = fit_halpha(
                            filepath=f"{work_dir}Sigma_fit/"+ Objeto +"o_1d.fits",
                            ha_center=centroides[1][i],
                            window_A=12,
                            n_mc=2500,
                            save_fig=True,
                        )

                        arc_extract_1d(f"{work_dir}ARCS/"+Arco+"o_wlcal.fits",f"{work_dir}OBJS/"+Objeto+"o_wlcal.fits")

                        Info_sigma_arco = fit_halpha(
                            filepath=f"{work_dir}ARCS/"+Arco+"o_1d.fits",
                            ha_center=6583.906,
                            window_A=5,
                            n_mc=2500,
                            save_fig=True,
                        )

                        sigma_calculator_function(Host= centroides[0][i],
                                                centroide_G=Info_sigma_spec["center_G"],
                                                centroide_GH=Info_sigma_spec["center_GH"],
                                                FWHM_G_Ha = Info_sigma_spec["fwhm_G"],
                                                e_FWHM_G_Ha = Info_sigma_spec["e_fwhm_G"],
                                                FWHM_GH_Ha = Info_sigma_spec["fwhm_GH"],
                                                e_FWHM_GH_Ha = Info_sigma_spec["e_fwhm_GH"],
                                                FWHM_G_Th = Info_sigma_arco["fwhm_G"],
                                                e_FWHM_G_Th = Info_sigma_arco["e_fwhm_G"],
                                                work_dir = work_dir)



                        print(Objeto,Arco,centroides[0][i],centroides[1][i],"\n")




