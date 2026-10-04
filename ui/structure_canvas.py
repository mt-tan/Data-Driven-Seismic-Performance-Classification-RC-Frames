"""Visual schematic of the RC frame entered by the user.

A lightweight Matplotlib canvas drawn from the input parameters. It is
purely informational and does not affect the prediction. The drawing
shows the frame elevation at the top, the column cross-section in the
middle, and the beam cross-section at the bottom — a vertical stack to
keep the two sections visually separated and easy to read.
"""
from __future__ import annotations

from typing import Dict

import matplotlib.patches as mp
from matplotlib.figure import Figure


# Light-theme palette
COLOR_BG = "#FFFFFF"
COLOR_CONCRETE = "#D6DBDF"
COLOR_BEAM_FILL = "#7FB3D5"
COLOR_COLUMN_FILL = "#A6ACAF"
COLOR_OUTLINE = "#34495E"
COLOR_REBAR = "#212121"
COLOR_REBAR_EDGE = "#FFFFFF"
COLOR_GROUND = "#5D6D7E"
COLOR_TEXT = "#212121"


def _draw_frame_elevation(ax, params: Dict[str, float]) -> None:
    ax.clear()
    ax.set_facecolor(COLOR_BG)
    N_story = int(params["N_story"])
    N_bay = int(params["N_bay"])
    Lbay = float(params["Lbay"])
    h1 = float(params["h1"])
    htyp = float(params["htyp"])

    col_w = max(0.2, min(0.4, Lbay * 0.06))
    beam_h = max(0.15, min(0.30, htyp * 0.10))

    x_cols = [i * Lbay for i in range(N_bay + 1)]
    y_floors = [0.0, h1] + [h1 + i * htyp for i in range(1, N_story)]
    Htot = h1 + (N_story - 1) * htyp

    # Floor levels (beams)
    for j, y in enumerate(y_floors[1:], start=1):
        ax.add_patch(mp.Rectangle(
            (0 - col_w / 2, y - beam_h / 2),
            x_cols[-1] + col_w, beam_h,
            facecolor=COLOR_BEAM_FILL, edgecolor=COLOR_OUTLINE, linewidth=0.8,
        ))

    # Columns
    for i, x in enumerate(x_cols):
        ax.add_patch(mp.Rectangle(
            (x - col_w / 2, 0), col_w, Htot,
            facecolor=COLOR_COLUMN_FILL, edgecolor=COLOR_OUTLINE,
            linewidth=0.8, alpha=0.85,
        ))

    # Ground hatching
    ax.add_patch(mp.Rectangle(
        (-col_w, -0.4), x_cols[-1] + 2 * col_w, 0.4,
        facecolor="none", edgecolor=COLOR_GROUND, hatch="//", linewidth=0.7,
    ))

    ax.set_xlim(-1.0, x_cols[-1] + 1.0)
    ax.set_ylim(-0.6, Htot + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{N_story}-story, {N_bay}-bay frame   (H = {Htot:.2f} m)",
                  fontsize=11, fontweight="bold", color=COLOR_TEXT, pad=6)


def _draw_column_section(ax, params: Dict[str, float]) -> None:
    ax.clear()
    ax.set_facecolor(COLOR_BG)
    B = float(params["col_B"])
    H = float(params["col_H"])
    cover = float(params["cover"])
    rho = float(params["rho_col"])

    ax.add_patch(mp.Rectangle((0, 0), B, H, facecolor=COLOR_CONCRETE,
                                edgecolor=COLOR_OUTLINE, linewidth=1.0))
    ax.add_patch(mp.Rectangle((cover, cover), B - 2 * cover, H - 2 * cover,
                                facecolor="none", edgecolor=COLOR_OUTLINE,
                                linewidth=0.7, linestyle="--"))

    Ag = B * H
    As = rho * Ag
    diam = (4 * As / (8 * 3.14159)) ** 0.5
    pts = [
        (cover + diam, cover + diam),
        (B / 2, cover + diam),
        (B - cover - diam, cover + diam),
        (cover + diam, H / 2),
        (B - cover - diam, H / 2),
        (cover + diam, H - cover - diam),
        (B / 2, H - cover - diam),
        (B - cover - diam, H - cover - diam),
    ]
    for x, y in pts:
        ax.add_patch(mp.Circle((x, y), diam, facecolor=COLOR_REBAR,
                                edgecolor=COLOR_REBAR_EDGE, linewidth=0.4))

    ax.set_xlim(-B * 0.1, B * 1.1)
    ax.set_ylim(-H * 0.1, H * 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"Column  {B:.2f}\u00D7{H:.2f} m   $\\rho$ = {rho * 100:.2f}%",
        fontsize=10, color=COLOR_TEXT, pad=4)


def _draw_beam_section(ax, params: Dict[str, float]) -> None:
    ax.clear()
    ax.set_facecolor(COLOR_BG)
    B = float(params["beam_b"])
    H = float(params["beam_h"])
    cover = float(params["cover"])
    rho_top = float(params["rho_beam_top"])
    rho_bot = float(params["rho_beam_bot"])

    ax.add_patch(mp.Rectangle((0, 0), B, H, facecolor=COLOR_CONCRETE,
                                edgecolor=COLOR_OUTLINE, linewidth=1.0))
    ax.add_patch(mp.Rectangle((cover, cover), B - 2 * cover, H - 2 * cover,
                                facecolor="none", edgecolor=COLOR_OUTLINE,
                                linewidth=0.7, linestyle="--"))

    Ag = B * H
    diam_top = max(0.005, (4 * rho_top * Ag / (3 * 3.14159)) ** 0.5)
    diam_bot = max(0.005, (4 * rho_bot * Ag / (3 * 3.14159)) ** 0.5)

    for i in range(3):
        x = cover + diam_top + i * (B - 2 * cover - 2 * diam_top) / 2
        ax.add_patch(mp.Circle((x, H - cover - diam_top), diam_top,
                                facecolor=COLOR_REBAR,
                                edgecolor=COLOR_REBAR_EDGE, linewidth=0.4))
        ax.add_patch(mp.Circle((x, cover + diam_bot), diam_bot,
                                facecolor=COLOR_REBAR,
                                edgecolor=COLOR_REBAR_EDGE, linewidth=0.4))

    ax.set_xlim(-B * 0.1, B * 1.1)
    ax.set_ylim(-H * 0.1, H * 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"Beam  {B:.2f}\u00D7{H:.2f} m   "
        f"$\\rho_t$={rho_top * 100:.2f}%, $\\rho_b$={rho_bot * 100:.2f}%",
        fontsize=10, color=COLOR_TEXT, pad=4)


def render_structure(fig: Figure, params: Dict[str, float]) -> None:
    """Render the elevation and two cross-sections vertically.

    Layout: 3 rows x 1 column.
      Row 1 - frame elevation  (largest, 2.4x height)
      Row 2 - column section
      Row 3 - beam section
    """
    fig.clear()
    fig.patch.set_facecolor(COLOR_BG)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.4, 1.0, 1.0],
                           hspace=0.35, left=0.05, right=0.95,
                           top=0.96, bottom=0.04)

    ax_frame = fig.add_subplot(gs[0])
    ax_col = fig.add_subplot(gs[1])
    ax_beam = fig.add_subplot(gs[2])
    for ax in (ax_frame, ax_col, ax_beam):
        ax.set_facecolor(COLOR_BG)
    _draw_frame_elevation(ax_frame, params)
    _draw_column_section(ax_col, params)
    _draw_beam_section(ax_beam, params)
