"""Generate publication-quality figures for the JOSS manuscript.

Outputs:
- paper/figure1_pipeline.png
- paper/figure2_disc_comparison.png
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Ensure src is in sys.path
src_dir = str(Path(__file__).resolve().parents[1] / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import osis
from osis.physics.optics import mtf_incoherent, airy_intensity



from matplotlib.patches import FancyBboxPatch


def generate_figure1():
    """Figure 1: OSIS Architecture and Physics Readout Channel Pipeline."""
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)
    ax.axis("off")

    stages = [
        ("Stage 1: Multilayer Optics", "Abelès 2×2 TMM\n• Matrix propagation\n• Land/mark reflectance\n• Complex index ñ = n + ik"),
        ("Stage 2: Diffraction & OTF", "Rayleigh & Scalar OTF\n• Spot: r = 0.61 λ/NA\n• Pupil autocorrelation\n• MTF modulation factor"),
        ("Stage 3: Optoelectronics", "Photodiode Detection\n• Responsivity ℛ(λ)\n• Optical coupling η_c\n• Signal power P_sig = I² R_L"),
        ("Stage 4: Physical Noise", "3-Component Noise\n• Quantum shot noise\n• Laser RIN noise\n• Johnson-Nyquist thermal"),
        ("Output Metrics", "Channel Performance\n• Carrier-to-Noise (CNR)\n• Bit Error Rate (BER)\n• Diagnostics & sweeps"),
    ]

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

    n_stages = len(stages)
    box_width = 1.55
    box_height = 2.0
    spacing = 0.45
    total_width = n_stages * box_width + (n_stages - 1) * spacing
    start_x = 0.2

    for i, ((title, desc), color) in enumerate(zip(stages, colors)):
        x = start_x + i * (box_width + spacing)
        y = 0.6

        # Draw box
        rect = FancyBboxPatch(
            (x, y), box_width, box_height,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            facecolor="#f8f9fa",
            edgecolor=color,
            linewidth=2.2,
            zorder=2,
        )
        ax.add_patch(rect)


        # Title bar
        ax.text(
            x + box_width / 2, y + box_height - 0.28, title,
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color=color, zorder=3,
        )

        # Description
        ax.text(
            x + 0.1, y + 0.3, desc,
            ha="left", va="bottom", fontsize=7.2, color="#333333",
            linespacing=1.35, zorder=3,
        )

        # Draw connecting arrow
        if i < n_stages - 1:
            arrow_x = x + box_width + 0.05
            arrow_y = y + box_height / 2
            ax.annotate(
                "",
                xy=(arrow_x + spacing - 0.1, arrow_y),
                xytext=(arrow_x, arrow_y),
                arrowprops=dict(arrowstyle="->", lw=2.0, color="#555555"),
                zorder=4,
            )

    ax.set_xlim(0, start_x + total_width + 0.2)
    ax.set_ylim(0, 3.2)
    plt.tight_layout()
    out_path = Path(__file__).parent / "figure1_pipeline.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Generated {out_path}")


def generate_figure2():
    """Figure 2: Multi-panel optical disc comparison, MTF curves, and sensitivity ranking."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.8), dpi=300)

    # Panel 1: Airy spot intensity profile
    r_um = np.linspace(-1.5, 1.5, 300) * 1e-6
    presets = [
        ("CD (780 nm, NA 0.45)", osis.CDConfig(), "#1f77b4"),
        ("DVD (650 nm, NA 0.60)", osis.DVDConfig(), "#ff7f0e"),
        ("Blu-ray (405 nm, NA 0.85)", osis.BluRayConfig(), "#2ca02c"),
    ]

    for name, cfg, color in presets:
        intensity = airy_intensity(r_um, cfg.wavelength_m, cfg.numerical_aperture)
        ax1.plot(r_um * 1e6, intensity, label=name, color=color, lw=2.0)

    ax1.set_title("(a) Focal Spot Airy Profiles", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Radial Position (µm)", fontsize=9)
    ax1.set_ylabel("Normalized Intensity", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=7.5, loc="upper right")
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-0.02, 1.05)

    # Panel 2: Incoherent MTF vs Spatial Frequency
    nu_lp_um = np.linspace(0, 4.5, 200) * 1e6
    for name, cfg, color in presets:
        mtf_vals = mtf_incoherent(nu_lp_um, cfg.wavelength_m, cfg.numerical_aperture)
        ax2.plot(nu_lp_um * 1e-6, mtf_vals, label=name, color=color, lw=2.0)


    ax2.set_title("(b) Incoherent Modulation Transfer Function", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Spatial Frequency (cycles / µm)", fontsize=9)
    ax2.set_ylabel("MTF(ν)", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=7.5, loc="upper right")
    ax2.set_xlim(0, 4.5)
    ax2.set_ylim(0, 1.05)

    # Panel 3: Blu-ray Parameter Sensitivity Elasticities
    cfg_bd = osis.BluRayConfig()
    sens = osis.one_at_a_time_sensitivity(cfg_bd, delta_fraction=0.05)
    ranked = osis.rank_parameters_by_influence(sens, metric="elasticity")

    param_names = [p.replace("_", " ").title() for p, _ in ranked[:6]]
    elasticities = [sens[p]["elasticity"] for p, _ in ranked[:6]]
    bar_colors = ["#2ca02c" if e > 0 else "#d62728" for e in elasticities]

    y_pos = np.arange(len(param_names))
    ax3.barh(y_pos, elasticities, color=bar_colors, alpha=0.85, edgecolor="black", height=0.6)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(param_names, fontsize=8)
    ax3.invert_yaxis()
    ax3.axvline(0, color="black", lw=1.0)
    ax3.set_title("(c) BD-RE Sensitivity Elasticities", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Elasticity: (% ΔCNR) / (% Δp)", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.5, axis="x")

    plt.tight_layout()
    out_path = Path(__file__).parent / "figure2_disc_comparison.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Generated {out_path}")


if __name__ == "__main__":
    generate_figure1()
    generate_figure2()
