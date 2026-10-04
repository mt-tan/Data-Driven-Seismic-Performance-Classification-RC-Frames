"""TBEC 2018 design spectrum for the Nurdağı (Gaziantep) seismic location.

Provides the elastic acceleration spectrum Sae(T) for the five soil classes
(ZA to ZE) at the DD-2 ground motion level. Spectral parameters Ss and S1
are obtained from the official Türkiye Earthquake Hazard Map for the
Nurdağı coordinates (37.180132°, 36.750172°).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# Site parameters (TBEC 2018, DD-2 level)
SS_NURDAGI: float = 1.159
S1_NURDAGI: float = 0.307
TL_TBDY: float = 6.0
G_CONST: float = 9.81

# Local soil amplification tables (TBEC 2018 Table 2.1 and 2.2)
FS_TABLE_SS = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50])
FS_TABLE: Dict[str, np.ndarray] = {
    "ZA": np.full(6, 0.8),
    "ZB": np.full(6, 0.9),
    "ZC": np.array([1.3, 1.3, 1.2, 1.2, 1.2, 1.2]),
    "ZD": np.array([1.6, 1.4, 1.2, 1.1, 1.0, 1.0]),
    "ZE": np.array([2.4, 1.7, 1.3, 1.1, 0.9, 0.8]),
}
F1_TABLE_S1 = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
F1_TABLE: Dict[str, np.ndarray] = {
    "ZA": np.full(6, 0.8),
    "ZB": np.full(6, 0.8),
    "ZC": np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.4]),
    "ZD": np.array([2.4, 2.2, 2.0, 1.9, 1.8, 1.7]),
    "ZE": np.array([4.2, 3.3, 2.8, 2.4, 2.2, 2.0]),
}

SOIL_TO_INT: Dict[str, int] = {"ZA": 0, "ZB": 1, "ZC": 2, "ZD": 3, "ZE": 4}
INT_TO_SOIL: Dict[int, str] = {v: k for k, v in SOIL_TO_INT.items()}


def get_amplification(soil: str) -> Tuple[float, float]:
    """Return the (FS, F1) amplification factors for the given soil class."""
    if soil not in FS_TABLE:
        raise ValueError(f"Unknown soil class '{soil}'. Use one of ZA..ZE.")
    FS = float(np.interp(SS_NURDAGI, FS_TABLE_SS, FS_TABLE[soil]))
    F1 = float(np.interp(S1_NURDAGI, F1_TABLE_S1, F1_TABLE[soil]))
    return FS, F1


def get_SDS_SD1(soil: str) -> Tuple[float, float]:
    """Return the (SDS, SD1) design spectral acceleration coefficients."""
    FS, F1 = get_amplification(soil)
    return FS * SS_NURDAGI, F1 * S1_NURDAGI


def get_corner_periods(soil: str) -> Tuple[float, float]:
    """Return the (TA, TB) corner periods of the design spectrum."""
    SDS, SD1 = get_SDS_SD1(soil)
    TB = SD1 / SDS
    TA = 0.2 * TB
    return TA, TB


def Sae(T: float, soil: str) -> float:
    """Elastic spectral acceleration in units of g."""
    SDS, SD1 = get_SDS_SD1(soil)
    TB = SD1 / SDS
    TA = 0.2 * TB
    if T <= TA:
        return SDS * (0.4 + 0.6 * T / TA)
    if T <= TB:
        return SDS
    if T <= TL_TBDY:
        return SD1 / T
    return SD1 * TL_TBDY / (T * T)


def Sae_vector(T_array: np.ndarray, soil_ord: np.ndarray) -> np.ndarray:
    """Vectorized helper used during batch prediction."""
    result = np.empty_like(T_array, dtype=float)
    for i, (T, ord_) in enumerate(zip(T_array, soil_ord)):
        result[i] = Sae(float(T), INT_TO_SOIL[int(ord_)])
    return result


def spectrum_curve(soil: str, T_max: float = 4.0, n: int = 600) -> Tuple[np.ndarray, np.ndarray]:
    """Sample the design spectrum on a logarithmic period grid for plotting."""
    T = np.linspace(0.01, T_max, n)
    Sa = np.array([Sae(t, soil) for t in T])
    return T, Sa
