"""TBEC 2018 input validation rules.

Performs two layers of checks on the building parameters supplied by the
user. The first layer enforces engineering-reasonable ranges (matches the
LHS sampling domain used to generate the training dataset). The second
layer enforces a subset of TBEC 2018 design requirements (Articles 7.3
and 7.4 for column and beam minimum dimensions and reinforcement ratios,
and the rough strong-column weak-beam screen used to generate the training set,
labelled as a screening proxy because it is not the Article 7.3.5.1 strength check).

Each rule returns a ``RuleResult`` containing the rule name, a boolean
pass flag, and a human-readable message. Inputs that violate any rule are
flagged but never blocked, since the GUI uses the violations only for
informational display.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RuleResult:
    rule: str
    passed: bool
    message: str
    severity: str = "warning"  # "info", "warning", "error"


# Sampling ranges used during dataset generation (modules/lhs_sampling.py RANGES_RAW and
# DISCRETE_VALUES; manuscript Table 2). h1 was sampled as htyp + [0.0, 1.2] m, hence the
# h1 bounds below are the realised envelope and the h1 >= htyp condition is checked separately.
RANGES: Dict[str, tuple] = {
    "N_story":      (2, 12),
    "N_bay":        (2, 4),
    "Lbay":         (3.0, 7.0),
    "h1":           (2.7, 4.4),
    "htyp":         (2.7, 3.2),
    "cover":        (0.02, 0.05),
    "col_B":        (0.30, 0.80),
    "col_H":        (0.30, 0.80),
    "beam_b":       (0.25, 0.40),
    "beam_h":       (0.40, 0.80),
    "rho_col":      (0.010, 0.040),
    "rho_beam_top": (0.005, 0.010),
    "rho_beam_bot": (0.0033, 0.008),
    "fc_MPa":       (20.0, 40.0),
    "G":            (15.0, 50.0),
    "Q":            (5.0, 20.0),
    "soil_ord":     (0, 4),
}

# Discrete sampling sets of the dataset (values in between are extrapolations)
DISCRETE: Dict[str, tuple] = {
    "N_story": (2, 4, 6, 8, 10, 12),
    "N_bay":   (2, 3, 4),
}


def _within_range(name: str, value: float) -> RuleResult:
    low, high = RANGES[name]
    ok = low <= value <= high
    if ok:
        return RuleResult(name, True, f"{name} = {value:g} within [{low:g}, {high:g}]", "info")
    return RuleResult(
        name, False,
        f"{name} = {value:g} outside [{low:g}, {high:g}]",
        "warning",
    )


def check_ranges(params: Dict[str, float]) -> List[RuleResult]:
    """Check that every input is inside the training distribution."""
    return [_within_range(k, params[k]) for k in RANGES if k in params]


def check_domain(params: Dict[str, float]) -> List[RuleResult]:
    """Training-domain conditions that are not simple bounds: the discrete story and bay
    counts, square columns and the first-story height not below the typical one."""
    out: List[RuleResult] = []
    for key, allowed in DISCRETE.items():
        if key in params:
            v = params[key]
            ok = any(abs(v - a) < 1e-9 for a in allowed)
            out.append(RuleResult(
                f"{key} discrete", ok,
                f"{key} = {v:g} " + ("is one of the sampled values " if ok else "is not among the sampled values ")
                + f"{list(allowed)}", "info" if ok else "warning"))
    if "col_B" in params and "col_H" in params:
        ok = abs(params["col_B"] - params["col_H"]) < 1e-9
        out.append(RuleResult(
            "square column", ok,
            "col_B = col_H (as in the training set)" if ok
            else f"col_B = {params['col_B']:g} differs from col_H = {params['col_H']:g}; the training set has square columns only",
            "info" if ok else "warning"))
    if "h1" in params and "htyp" in params:
        ok = params["h1"] >= params["htyp"] - 1e-9
        out.append(RuleResult(
            "h1 >= htyp", ok,
            "first story not shorter than the typical story" if ok
            else f"h1 = {params['h1']:g} m is below htyp = {params['htyp']:g} m (never sampled)",
            "info" if ok else "warning"))
    return out


def check_tbdy_rules(params: Dict[str, float]) -> List[RuleResult]:
    """Selected TBEC 2018 design checks (Articles 7.3 and 7.4), used as a screening only."""
    out: List[RuleResult] = []

    # Minimum column dimension (TBEC 2018 Article 7.3.1.1)
    if params["col_B"] >= 0.30 and params["col_H"] >= 0.30:
        out.append(RuleResult(
            "TBEC 7.3.1.1 (column min dimension)", True,
            f"col_B = {params['col_B']:.2f} m, col_H = {params['col_H']:.2f} m >= 0.30 m",
            "info"))
    else:
        out.append(RuleResult(
            "TBEC 7.3.1.1 (column min dimension)", False,
            "Column shortest side must be at least 0.30 m.",
            "error"))

    # Minimum beam dimension (TBEC 2018 Article 7.4.1.1)
    if params["beam_b"] >= 0.25 and params["beam_h"] >= 0.30:
        out.append(RuleResult(
            "TBEC 7.4.1.1 (beam min dimension)", True,
            f"beam_b = {params['beam_b']:.2f} m, beam_h = {params['beam_h']:.2f} m",
            "info"))
    else:
        out.append(RuleResult(
            "TBEC 7.4.1.1 (beam min dimension)", False,
            "Beam minimum width 0.25 m, minimum depth 0.30 m.",
            "error"))

    # Beam aspect ratio (depth <= 3.5 x width): a common detailing heuristic, not a code clause.
    # Inside the sampled domain (beam_h <= 0.80 m, beam_b >= 0.25 m) it can never fail.
    if params["beam_h"] <= 3.5 * params["beam_b"]:
        out.append(RuleResult(
            "Beam aspect (heuristic)", True,
            f"beam_h / beam_b = {params['beam_h'] / params['beam_b']:.2f} <= 3.5",
            "info"))
    else:
        out.append(RuleResult(
            "Beam aspect (heuristic)", False,
            f"beam_h / beam_b = {params['beam_h'] / params['beam_b']:.2f} > 3.5 (heuristic, not a code check)",
            "warning"))

    # Column reinforcement ratio (TBEC 2018 7.3.3, 0.01 to 0.04)
    if 0.01 <= params["rho_col"] <= 0.04:
        out.append(RuleResult(
            "TBEC 7.3.3 (column reinf ratio)", True,
            f"rho_col = {params['rho_col']:.4f} within [0.010, 0.040]",
            "info"))
    else:
        out.append(RuleResult(
            "TBEC 7.3.3 (column reinf ratio)", False,
            "Column reinforcement ratio must be between 0.010 and 0.040.",
            "error"))

    # Beam reinforcement ratios (TBEC 2018 7.4.3.1, 0.0033 to 0.02)
    for tag in ("rho_beam_top", "rho_beam_bot"):
        if 0.0033 <= params[tag] <= 0.020:
            out.append(RuleResult(
                f"TBEC 7.4.3.1 ({tag})", True,
                f"{tag} = {params[tag]:.4f} within [0.0033, 0.020]",
                "info"))
        else:
            out.append(RuleResult(
                f"TBEC 7.4.3.1 ({tag})", False,
                f"{tag} must be in [0.0033, 0.020].",
                "error"))

    # Strong column - weak beam screening proxy. This is the same rough dimensional filter that was
    # applied when the training set was generated (modules/tbdy_filters.filter_scwb_rough): the beam
    # may not be deeper than the column, and a lightly reinforced column (rho_col < 0.012) may not
    # carry a beam deeper than 0.9 col_H. It is NOT the TBEC 2018 Article 7.3.5.1 strength check
    # (sum Mra >= 1.2 sum Mrb), which requires section moment capacities.
    bh, cH, rc = params["beam_h"], params["col_H"], params["rho_col"]
    ratio_hb = bh / max(cH, 1e-9)
    if bh > cH:
        out.append(RuleResult(
            "SCWB screening proxy (dataset filter)", False,
            f"beam_h = {bh:.2f} m > col_H = {cH:.2f} m: beam deeper than column (outside the training "
            f"screen; not a TBEC 7.3.5.1 strength check).",
            "warning"))
    elif rc < 0.012 and ratio_hb > 0.9:
        out.append(RuleResult(
            "SCWB screening proxy (dataset filter)", False,
            f"rho_col = {rc:.3f} < 0.012 with beam_h / col_H = {ratio_hb:.2f} > 0.9 (outside the "
            f"training screen; not a TBEC 7.3.5.1 strength check).",
            "warning"))
    else:
        out.append(RuleResult(
            "SCWB screening proxy (dataset filter)", True,
            f"beam_h / col_H = {ratio_hb:.2f} <= 1.0" + ("" if rc >= 0.012 else " and <= 0.9"),
            "info"))

    return out


def htot_from_params(params: Dict[str, float]) -> float:
    """Convenience helper used in several panels."""
    return float(params["h1"] + (params["N_story"] - 1) * params["htyp"])


def validate(params: Dict[str, float]) -> Dict[str, List[RuleResult]]:
    """Run both range and TBEC 2018 validation. Returns dict with two lists."""
    return {
        "ranges": check_ranges(params) + check_domain(params),
        "tbdy": check_tbdy_rules(params),
    }


def overall_status(results: Dict[str, List[RuleResult]]) -> str:
    """Aggregate severity: 'ok', 'warning', or 'error'."""
    rules = results["ranges"] + results["tbdy"]
    if any((not r.passed) and r.severity == "error" for r in rules):
        return "error"
    if any((not r.passed) and r.severity == "warning" for r in rules):
        return "warning"
    return "ok"
