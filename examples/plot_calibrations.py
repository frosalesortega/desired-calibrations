"""
Visualize all DESIRED strong-line metallicity calibrations
==========================================================

Generates diagnostic plots for every calibration stored in
calibrations.json, showing the polynomial fit with MC error
bands for both the t2=0 and t2>=0 scenarios.

Usage:
    python plot_calibrations.py                 # Plot all calibrations
    python plot_calibrations.py N2 O3N2 R3      # Plot specific ones
    python plot_calibrations.py --save           # Save to PDF instead of showing

Reference: Rosales-Ortega et al. (2026a)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from desired import list_calibrations, get_info
from desired.calibrate import _mc_poly_error, _eval_disp


def plot_single_branch(ax, name, coeff, disp_global, disp_coeff, xmin, xmax,
                       scenario_label, color, mean_err_frac=0.05):
    """
    Plot a single calibration branch with MC and dispersion bands.
    Shows sigma(x) variable band when disp_coeff is available.
    """
    x = np.linspace(xmin, xmax, 100)

    # Assumed mean index error: 5% of the validity range
    mean_err = mean_err_frac * (xmax - xmin)
    x_e = mean_err * np.ones_like(x)

    # Polynomial evaluation
    poly_coeffs = coeff
    y_std, y_mean = _mc_poly_error(x, x_e, poly_coeffs)

    # Evaluate dispersions
    sigma_global = np.full_like(x, disp_global)
    if disp_coeff is not None and len(disp_coeff) > 0:
        sigma_local = _eval_disp(x, disp_global, disp_coeff, "local")
        sigma_max = _eval_disp(x, disp_global, disp_coeff, "max")
        has_local = True
    else:
        sigma_local = sigma_global
        sigma_max = sigma_global
        has_local = False

    # --- Plot ---
    # Calibration line
    ax.plot(x, y_mean, '-', color=color, lw=2, label=scenario_label, zorder=3)

    # MC error band
    ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                    color=color, alpha=0.30, zorder=2,
                    label='MC 1σ')

    # Total band: MC + max[sigma_cal, sigma(x)]
    total_err = np.sqrt(y_std**2 + sigma_max**2)
    ax.fill_between(x, y_mean - total_err, y_mean + total_err,
                    color=color, alpha=0.12, zorder=1,
                    label=f'MC + σ(x) (max)')

    # sigma(x) profile as dashed line (on a secondary visual)
    if has_local:
        # Plot sigma_global as horizontal reference
        ax_sigma = ax.twinx()
        ax_sigma.plot(x, sigma_local, '--', color=color, alpha=0.5, lw=1)
        ax_sigma.axhline(disp_global, color=color, alpha=0.3, lw=0.8, ls=':')
        ax_sigma.set_ylabel('σ(x) [dex]', fontsize=8, alpha=0.5)
        ax_sigma.tick_params(labelsize=7, colors='grey')
        ax_sigma.set_ylim(0, max(np.max(sigma_local), disp_global) * 2.5)


def plot_calibration(name, save=False, save_dir='.'):
    """
    Plot a complete calibration diagnostic figure.
    """
    info = get_info(name)
    cal_type = info["type"]
    definition = info.get("definition", name)
    ymin, ymax = 6.5, 9.8

    if cal_type == "monotonic":
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        fig.suptitle(f'DESIRED  {name}  calibration\n{definition}',
                     fontsize=13, fontweight='bold')

        for j, (scenario, label) in enumerate([("t2eq0", "t²=0"), ("t2ge0", "t²≥0")]):
            ax = axes[j]
            sc = info[scenario]
            coeff = sc["coeff"]
            disp = sc["disp"]
            disp_coeff = sc.get("disp_coeff", None)

            if not coeff or info["xmin"] is None:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=14, color='gray')
                ax.set_title(label)
                continue

            color = '#2166ac' if j == 0 else '#b2182b'
            plot_single_branch(ax, name, coeff, disp, disp_coeff,
                               info["xmin"], info["xmax"],
                               label, color)

            ax.set_xlim(info["xmin"], info["xmax"])
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel(name, fontsize=12)
            ax.set_ylabel('12 + log(O/H)', fontsize=12)
            ax.set_title(label, fontsize=12)
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)

            deg = len(coeff) - 1
            dc_str = f'\nσ(x) coeff: {disp_coeff}' if disp_coeff else ''
            ax.text(0.03, 0.97, f'Poly degree: {deg}\nσ_fit: {disp:.2f} dex{dc_str}',
                    transform=ax.transAxes, fontsize=7, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    elif cal_type == "double_valued":
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
        fig.suptitle(f'DESIRED  {name}  calibration (double-valued)\n{definition}',
                     fontsize=13, fontweight='bold')

        for i, (branch, br_label) in enumerate([("upper", "Upper branch"),
                                                 ("lower", "Lower branch")]):
            br_data = info[branch]
            xmin = br_data["xmin"]
            xmax = br_data["xmax"]

            for j, (scenario, sc_label) in enumerate([("t2eq0", "t²=0"),
                                                       ("t2ge0", "t²≥0")]):
                ax = axes[i, j]
                sc = br_data[scenario]
                coeff = sc["coeff"]
                disp = sc["disp"]
                disp_coeff = sc.get("disp_coeff", None)

                if not coeff or xmin is None:
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                            ha='center', va='center', fontsize=14, color='gray')
                    ax.set_title(f'{br_label} — {sc_label}')
                    continue

                color = '#2166ac' if j == 0 else '#b2182b'
                plot_single_branch(ax, name, coeff, disp, disp_coeff,
                                   xmin, xmax, sc_label, color)

                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_xlabel(name, fontsize=12)
                ax.set_ylabel('12 + log(O/H)', fontsize=12)
                ax.set_title(f'{br_label} — {sc_label}', fontsize=12)
                ax.legend(fontsize=8, loc='best')
                ax.grid(True, alpha=0.3)

                deg = len(coeff) - 1
                ax.text(0.03, 0.97,
                        f'Poly degree: {deg}\nσ_fit: {disp:.2f} dex\n'
                        f'Range: [{xmin:.2f}, {xmax:.2f}]',
                        transform=ax.transAxes, fontsize=7, va='top',
                        bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    plt.tight_layout()

    if save:
        fname = f'{save_dir}/desired_{name}.pdf'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f'  Saved: {fname}')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_all_overview(save=False, save_dir='.'):
    """
    Plot a compact overview of ALL calibrations (t2eq0 only)
    in a single multi-panel figure.
    """
    mono_names = list_calibrations(kind="monotonic")
    dv_names = list_calibrations(kind="double_valued")

    mono_valid = [n for n in mono_names
                  if get_info(n)["t2eq0"]["coeff"] and get_info(n)["xmin"] is not None]
    dv_valid = [n for n in dv_names
                if (get_info(n)["upper"]["t2eq0"]["coeff"]
                    and get_info(n)["upper"]["xmin"] is not None)]

    all_names = mono_valid + dv_valid
    n = len(all_names)

    if n == 0:
        print("No calibrations with data found in JSON.")
        return

    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                             sharey=True)
    axes = np.atleast_2d(axes)
    fig.suptitle('DESIRED Calibrations Overview (t²=0)',
                 fontsize=15, fontweight='bold', y=1.01)

    ymin, ymax = 6.5, 9.8

    for idx, name in enumerate(all_names):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        info = get_info(name)

        if info["type"] == "monotonic":
            coeff = info["t2eq0"]["coeff"]
            disp = info["t2eq0"]["disp"]
            disp_coeff = info["t2eq0"].get("disp_coeff", None)
            xmin, xmax = info["xmin"], info["xmax"]
            color = '#2166ac'
            title = name
        else:
            coeff = info["upper"]["t2eq0"]["coeff"]
            disp = info["upper"]["t2eq0"]["disp"]
            disp_coeff = info["upper"]["t2eq0"].get("disp_coeff", None)
            xmin = info["upper"]["xmin"]
            xmax = info["upper"]["xmax"]
            color = '#7570b3'
            title = f'{name} (upper)'

        x = np.linspace(xmin, xmax, 80)
        mean_err = 0.05 * (xmax - xmin)
        x_e = mean_err * np.ones_like(x)
        y_std, y_mean = _mc_poly_error(x, x_e, coeff)

        # Use max[sigma_cal, sigma(x)] for the band
        sigma = _eval_disp(x, disp, disp_coeff, "max")
        total_err = np.sqrt(y_std**2 + sigma**2)

        ax.plot(x, y_mean, '-', color=color, lw=1.5)
        ax.fill_between(x, y_mean - total_err, y_mean + total_err,
                        color=color, alpha=0.15)
        ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                        color=color, alpha=0.3)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel(name, fontsize=9)
        if col == 0:
            ax.set_ylabel('12+log(O/H)', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)

        deg = len(coeff) - 1
        ax.text(0.05, 0.95, f'deg={deg}, σ={disp:.2f}',
                transform=ax.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    # Hide empty subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout()

    if save:
        fname = f'{save_dir}/desired_overview.pdf'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f'Saved: {fname}')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


# -----------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------
if __name__ == "__main__":
    args = sys.argv[1:]
    save = '--save' in args
    args = [a for a in args if a != '--save']

    if not args:
        print("Plotting overview...")
        plot_all_overview(save=save)

        all_names = list_calibrations()
        for name in all_names:
            info = get_info(name)
            has_data = False
            if info["type"] == "monotonic":
                has_data = bool(info["t2eq0"]["coeff"]) and info["xmin"] is not None
            else:
                has_data = (bool(info["upper"]["t2eq0"]["coeff"])
                            and info["upper"]["xmin"] is not None)
            if has_data:
                print(f"Plotting {name}...")
                plot_calibration(name, save=save)
    else:
        for name in args:
            print(f"Plotting {name}...")
            plot_calibration(name, save=save)
