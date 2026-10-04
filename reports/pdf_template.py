"""PDF report generator.

Creates a single-page assessment report from a :class:`PredictionResult`
using the ``fpdf2`` library. The report contains the building parameters,
the predicted pushover quantities, the engineered features, the
performance verdict, the class probabilities, and the local SHAP top-5
contributions.

Uses Matplotlib's bundled DejaVu Sans font so that Unicode characters
(Greek letters, Turkish accents, superscripts) render
correctly.  No external font files are required.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.font_manager as fm
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from core.predictor import PredictionResult
from core.spectrum import INT_TO_SOIL
from core.formatting import fmt_prob


CLASS_DESCRIPTIONS = {
    "IO": "Immediate Occupancy",
    "LS": "Life Safety",
    "CP": "Collapse Prevention",
    "C":  "Collapse",
}
CLASS_COLOR_RGB = {
    "IO": (30, 132, 73),
    "LS": (214, 137, 16),
    "CP": (202, 111, 30),
    "C":  (146, 43, 33),
}

# Font family registered with FPDF (must match name passed to set_font).
PDF_FONT = "DejaVu"


def _find_dejavu(variant: str) -> str:
    """Locate a DejaVu Sans TTF file via Matplotlib's bundled fonts."""
    candidates = {
        "regular": "DejaVuSans.ttf",
        "bold":    "DejaVuSans-Bold.ttf",
        "italic":  "DejaVuSans-Oblique.ttf",
        "bolditalic": "DejaVuSans-BoldOblique.ttf",
    }
    target = candidates[variant]
    for font in fm.fontManager.ttflist:
        if Path(font.fname).name == target:
            return font.fname
    # Fallback to a font lookup by name + style
    props = fm.FontProperties(family="DejaVu Sans",
                               weight="bold" if "bold" in variant else "normal",
                               style="italic" if "italic" in variant else "normal")
    return fm.findfont(props)


class _PDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register DejaVu Sans in four variants. uni=True is the default in
        # fpdf2 for TTF fonts but we pass it explicitly for clarity.
        self.add_font(PDF_FONT, "",   _find_dejavu("regular"))
        self.add_font(PDF_FONT, "B",  _find_dejavu("bold"))
        self.add_font(PDF_FONT, "I",  _find_dejavu("italic"))
        self.add_font(PDF_FONT, "BI", _find_dejavu("bolditalic"))

    def header(self):
        self.set_font(PDF_FONT, "B", 13)
        self.set_text_color(20, 20, 20)
        self.cell(0, 8,
                   "Pushover Performance Prediction Report - RC Frame (TBEC 2018)",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        self.set_font(PDF_FONT, "", 9)
        self.set_text_color(90, 90, 90)
        self.cell(0, 5, f"Generated on {datetime.now():%Y-%m-%d %H:%M}",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        self.ln(1)

    def footer(self):
        self.set_y(-12)
        self.set_font(PDF_FONT, "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6,
                   "Decision-support tool. Not a substitute for engineering judgement.",
                   align="C")


def _section_title(pdf: FPDF, text: str) -> None:
    pdf.set_font(PDF_FONT, "B", 11)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(150, 150, 150)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(1)


def _two_column_table(pdf: FPDF, rows: Dict[str, str], col_w_label: float = 70.0,
                       col_w_value: float = 50.0, row_h: float = 4.0) -> None:
    pdf.set_font(PDF_FONT, "", 9)
    pdf.set_text_color(40, 40, 40)
    for label, value in rows.items():
        pdf.cell(col_w_label, row_h, label, border=0)
        pdf.cell(col_w_value, row_h, value, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def generate_report(result: PredictionResult, output_path: Path) -> Path:
    pdf = _PDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # 1. Building parameters
    _section_title(pdf, "1. Building parameters")
    p = result.inputs
    soil_name = INT_TO_SOIL.get(int(p["soil_ord"]), "?")
    params_rows = {
        "Number of stories":           f"{int(p['N_story'])}",
        "Number of bays":              f"{int(p['N_bay'])}",
        "Bay length (m)":              f"{p['Lbay']:.2f}",
        "First story height (m)":      f"{p['h1']:.2f}",
        "Typical story height (m)":    f"{p['htyp']:.2f}",
        "Total height (m)":            f"{result.Htot:.2f}",
        "Concrete cover (m)":          f"{p['cover']:.3f}",
        "Column section (m)":          f"{p['col_B']:.2f} x {p['col_H']:.2f}",
        "Beam section (m)":            f"{p['beam_b']:.2f} x {p['beam_h']:.2f}",
        "Column reinforcement ratio":  f"{p['rho_col']:.4f}",
        "Beam top / bottom reinf":     f"{p['rho_beam_top']:.4f} / {p['rho_beam_bot']:.4f}",
        "Concrete strength (MPa)":     f"{p['fc_MPa']:.1f}",
        "Dead load G (kN/m)":          f"{p['G']:.1f}",
        "Live load Q (kN/m)":          f"{p['Q']:.1f}",
        "Soil class (TBEC 2018)":      soil_name,
        "Site (design spectrum)":      "Nurdağı, Gaziantep (DD-2)",
    }
    _two_column_table(pdf, params_rows)
    pdf.ln(1)

    # 2. Predicted pushover quantities
    _section_title(pdf, "2. Predicted pushover quantities")
    reg = result.regression
    reg_rows = {
        "Yield base shear V_y (kN)":         f"{reg['Vy']:.1f}",
        "Yield displacement δy (m)":         f"{reg['delta_y']:.4f}",
        "Maximum base shear V_max (kN)":     f"{reg['Vmax']:.1f}",
        "Displacement at V_max (m)":         f"{reg['delta_max']:.4f}",
        "Target displacement δt (m)":        f"{reg['delta_t']:.4f}",
        "Base shear at δt (kN)":             f"{reg['V_at_dt']:.1f}",
        "Ultimate displacement δu (m)":      f"{reg['delta_u']:.4f}",
        "Fundamental period T1 (s)":         f"{reg['T1']:.3f}",
        "Sa(T1) [g]":                        f"{result.Sa_T1:.4f}",
        "Total seismic weight W (kN)":       f"{result.W_total:.1f}",
    }
    _two_column_table(pdf, reg_rows)
    pdf.ln(1)

    # 3. Engineered features
    _section_title(pdf, "3. Engineered features (used by the classifier)")
    aug = result.augmented
    aug_rows = {
        "θt = δt / Htot":                    f"{aug['theta_t']:.5f}",
        "DCRδ = δt / δu":                    f"{aug['DCR_delta']:.4f}",
        "μd = δt / δy":                      f"{aug['mu_d']:.3f}",
        "V(δt) / Vmax":                     f"{aug['V_at_dt_over_Vmax']:.4f}",
        "V_y / (W Sa(T1))":                f"{aug['Vy_over_W_Sa']:.5f}",
    }
    _two_column_table(pdf, aug_rows)
    pdf.ln(1)

    # 4. Performance verdict
    _section_title(pdf, "4. Performance level verdict")
    pdf.set_font(PDF_FONT, "B", 20)
    rgb = CLASS_COLOR_RGB.get(result.verdict, (90, 90, 90))
    pdf.set_text_color(*rgb)
    pdf.cell(0, 9,
              f"{result.verdict}: {CLASS_DESCRIPTIONS.get(result.verdict, '')}",
              new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(40, 40, 40)
    pdf.set_font(PDF_FONT, "", 10)
    pdf.cell(0, 6, f"Top-class probability: {fmt_prob(result.confidence)}",
              new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if result.notes:
        pdf.set_font(PDF_FONT, "", 8)
        pdf.multi_cell(0, 3.8, "Note: " + " ".join(result.notes))
        pdf.set_font(PDF_FONT, "", 10)
    pdf.ln(2)

    pdf.set_font(PDF_FONT, "B", 9)
    pdf.cell(0, 5, "Class probabilities", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(PDF_FONT, "", 9)
    for cls in ("IO", "LS", "CP", "C"):
        prob = result.class_probabilities.get(cls, 0.0)
        pdf.cell(40, 4.2, f"{cls} ({CLASS_DESCRIPTIONS[cls]})", border=0)
        pdf.cell(0, 4.2, fmt_prob(prob), border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # 5. Local SHAP top-5
    _section_title(pdf, "5. Top contributing features (local SHAP)")
    top = sorted(result.shap_local.items(),
                 key=lambda kv: abs(kv[1]), reverse=True)[:5]
    pdf.set_font(PDF_FONT, "", 9)
    for name, value in top:
        direction = "+" if value >= 0 else "-"
        pdf.cell(60, 4.2, name, border=0)
        pdf.cell(20, 4.2, direction, border=0, align="C")
        pdf.cell(0, 4.2, f"{value:+.4f}", border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(3)
    pdf.set_font(PDF_FONT, "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 3.6,
                    "This report is produced by the cascade ML framework "
                    "calibrated on 48,725 RC frame buildings designed per "
                    "TBEC 2018. The prediction supports preliminary "
                    "screening but does not replace a full nonlinear "
                    "analysis or the engineer's judgement.")

    output_path = Path(output_path)
    pdf.output(str(output_path))
    return output_path
