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
from desired.calibrate import _mc_poly_error


def plot_single_branch(ax, name, coeff, disp, xmin, xmax, definition,
                       scenario_label, color, mean_err_frac=0.05):
    """
    Plot a single calibration branch (one scenario) on the given axes.

    Parameters
    ----------
    ax             : matplotlib Axes
    name           : str, calibration name (for labels)
    coeff          : list, ascending power coefficients [a0, a1, ...]
    disp           : float, intrinsic dispersion
    xmin, xmax     : float, validity range
    definition     : str, index definition (for xlabel)
    scenario_label : str, e.g. 't²=0' or 't²≥0'
    color          : str, color for the fit line and bands
    mean_err_frac  : float, assumed fractional mean error in index
                     (used for MC band width; default 5% of range)
    """
    x = np.linspace(xmin, xmax, 100)

    # Assumed mean index error: 5% of the validity range
    mean_err = mean_err_frac * (xmax - xmin)
    x_e = mean_err * np.ones_like(x)

    # Polynomial evaluation (polyval needs descending order)
    poly_coeffs = coeff
    y_std, y_mean = _mc_poly_error(x, x_e, poly_coeffs)

    # Plot
    ax.plot(x, y_mean, '-', color=color, lw=2, label=scenario_label, zorder=2)
    ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                    color=color, alpha=0.30, zorder=1,
                    label=f'MC 1σ')
    ax.fill_between(x, y_mean - (y_std + disp), y_mean + (y_std + disp),
                    color=color, alpha=0.12, zorder=0,
                    label=f'MC 1σ + σ_int ({disp:.2f})')


def plot_calibration(name, save=False, save_dir='.'):
    """
    Plot a complete calibration diagnostic figure.

    For monotonic: one row with t2eq0 (left) and t2ge0 (right).
    For double-valued: two rows (upper/lower) × two columns (t2eq0/t2ge0).
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

            if not coeff or info["xmin"] is None:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=14, color='gray')
                ax.set_title(label)
                continue

            color = '#2166ac' if j == 0 else '#b2182b'
            plot_single_branch(ax, name, coeff, disp,
                               info["xmin"], info["xmax"],
                               definition, label, color)

            ax.set_xlim(info["xmin"], info["xmax"])
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel(name, fontsize=12)
            ax.set_ylabel('12 + log(O/H)', fontsize=12)
            ax.set_title(label, fontsize=12)
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)

            # Annotate polynomial degree
            deg = len(coeff) - 1
            ax.text(0.03, 0.97, f'Poly degree: {deg}\nσ_int: {disp:.2f} dex',
                    transform=ax.transAxes, fontsize=8, va='top',
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

                if not coeff or xmin is None:
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                            ha='center', va='center', fontsize=14, color='gray')
                    ax.set_title(f'{br_label} — {sc_label}')
                    continue

                color = '#2166ac' if j == 0 else '#b2182b'
                plot_single_branch(ax, name, coeff, disp,
                                   xmin, xmax, definition, sc_label, color)

                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_xlabel(name, fontsize=12)
                ax.set_ylabel('12 + log(O/H)', fontsize=12)
                ax.set_title(f'{br_label} — {sc_label}', fontsize=12)
                ax.legend(fontsize=8, loc='best')
                ax.grid(True, alpha=0.3)

                deg = len(coeff) - 1
                ax.text(0.03, 0.97,
                        f'Poly degree: {deg}\nσ_int: {disp:.2f} dex\n'
                        f'Range: [{xmin:.2f}, {xmax:.2f}]',
                        transform=ax.transAxes, fontsize=8, va='top',
                        bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    plt.tight_layout()

    if save:
        fname = f'{save_dir}/desired_{name}.pdf'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f'  Saved: {fname}')
        plt.close(fig)
    else:
        plt.show()


def plot_all_overview(save=False, save_dir='.'):
    """
    Plot a compact overview of ALL calibrations (t2eq0 only)
    in a single multi-panel figure.
    """
    mono_names = list_calibrations(kind="monotonic")
    dv_names = list_calibrations(kind="double_valued")

    # Filter to those with actual data
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
            xmin, xmax = info["xmin"], info["xmax"]
            color = '#2166ac'
            title = name
        else:
            # Double-valued: plot upper branch
            coeff = info["upper"]["t2eq0"]["coeff"]
            disp = info["upper"]["t2eq0"]["disp"]
            xmin = info["upper"]["xmin"]
            xmax = info["upper"]["xmax"]
            color = '#7570b3'
            title = f'{name} (upper)'

        x = np.linspace(xmin, xmax, 80)
        mean_err = 0.05 * (xmax - xmin)
        x_e = mean_err * np.ones_like(x)
        poly_coeffs = coeff
        y_std, y_mean = _mc_poly_error(x, x_e, poly_coeffs)

        ax.plot(x, y_mean, '-', color=color, lw=1.5)
        ax.fill_between(x, y_mean - (y_std + disp), y_mean + (y_std + disp),
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

        # Add degree and disp annotation
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


# -----------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------
if __name__ == "__main__":
    args = sys.argv[1:]
    save = '--save' in args
    args = [a for a in args if a != '--save']

    if not args:
        # Plot overview + individual for all with data
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
