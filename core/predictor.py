"""Cascade prediction pipeline used by the GUI.

Given a dictionary of 18 raw building parameters the :class:`Predictor`
class runs the full pipeline:

1. Eight regression models produce the seven primary pushover outputs and
   the fundamental period.
2. Five engineered features are derived from these regression outputs.
3. The cascade classifier produces a four-class performance verdict and
   class probabilities (IO / LS / CP / C).
4. A local SHAP explanation is obtained from the classifier's native
   TreeSHAP implementation.

The output is a :class:`PredictionResult` dataclass that the UI can render
without further processing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from catboost import Pool

from . import spectrum as sp
from .model_loader import (
    ALL_CLF_FEATURES, AUG_NAMES, CLASS_LABELS, INPUT_COLS,
    INT_TO_CLASS, REGRESSION_TARGETS, ModelBundle,
)

# Regression targets whose models are trained with a log(1 + y) transform.
LOG_TRANSFORM_TARGETS = {
    "delta_y", "delta_max", "delta_t", "delta_u",
    "Vy", "Vmax", "V_at_dt",
}


def _safe_expm1(y: np.ndarray) -> np.ndarray:
    return np.expm1(y)


@dataclass
class PredictionResult:
    """Container for everything the UI needs after a Predict click."""
    inputs: Dict[str, float] = field(default_factory=dict)
    regression: Dict[str, float] = field(default_factory=dict)
    augmented: Dict[str, float] = field(default_factory=dict)
    verdict: str = ""              # one of IO, LS, CP, C
    verdict_int: int = -1
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    shap_local: Dict[str, float] = field(default_factory=dict)
    base_value: float = 0.0
    confidence: float = 0.0        # top class probability
    Sa_T1: float = 0.0
    W_total: float = 0.0
    Htot: float = 0.0
    clipped_targets: List[str] = field(default_factory=list)   # regressor outputs that were negative and clipped at zero
    notes: List[str] = field(default_factory=list)             # human-readable caveats shown in the UI, PDF and batch output


TARGET_LABELS = {
    "Vy": "yield base shear", "delta_y": "yield displacement", "Vmax": "peak base shear",
    "delta_max": "displacement at peak base shear", "V_at_dt": "base shear at the target displacement",
    "delta_t": "target displacement", "delta_u": "ultimate displacement", "T1": "fundamental period",
}


def clipping_notes(clipped: List[str]) -> List[str]:
    """Notes for regressor outputs that were negative and clipped at zero. On the 9,745 hold-out frames of the
    training campaign this occurred only for the target displacement of 28 very stiff frames (0.3 percent) and
    did not change any verdict, because the engineered features are evaluated at the same floor value in the
    manuscript pipeline."""
    out = []
    for t in clipped:
        if t == "delta_t":
            out.append("Predicted target displacement clipped at zero (very stiff frame); "
                       "the drift-based features are evaluated at the floor value.")
        else:
            out.append(f"Predicted {TARGET_LABELS.get(t, t)} clipped at zero.")
    return out


class Predictor:
    def __init__(self, bundle: ModelBundle):
        if not bundle.ready:
            raise RuntimeError("ModelBundle is not loaded.")
        self.bundle = bundle

    # ----------------------------------------------------------------
    # Auxiliary derivations
    # ----------------------------------------------------------------
    @staticmethod
    def _Htot(params: Dict[str, float]) -> float:
        return params["h1"] + (params["N_story"] - 1) * params["htyp"]

    @staticmethod
    def _W_total(params: Dict[str, float]) -> float:
        Wpu = params["G"] + 0.3 * params["Q"]
        floor_w = Wpu * params["N_bay"] * params["Lbay"]
        return floor_w * params["N_story"]

    @staticmethod
    def _derive_augmented(reg_out: Dict[str, float], Htot: float,
                           W_total: float, Sa_T1: float) -> Dict[str, float]:
        eps = 1e-9
        Vy = max(reg_out["Vy"], eps)
        Vmax = max(reg_out["Vmax"], eps)
        Vat = max(reg_out["V_at_dt"], eps)
        dy = max(reg_out["delta_y"], eps)
        dt = max(reg_out["delta_t"], eps)
        du = max(reg_out["delta_u"], eps)
        return {
            "theta_t": dt / max(Htot, eps),
            "DCR_delta": dt / du,
            "mu_d": dt / dy,
            "V_at_dt_over_Vmax": Vat / Vmax,
            "Vy_over_W_Sa": Vy / (max(W_total, eps) * max(Sa_T1, eps)),
        }

    # ----------------------------------------------------------------
    # Main pipeline
    # ----------------------------------------------------------------
    def predict(self, params: Dict[str, float]) -> PredictionResult:
        # Derive Htot if not supplied.
        if "Htot" not in params or params["Htot"] <= 0:
            params = dict(params)
            params["Htot"] = self._Htot(params)

        # Build the raw input vector in the exact order expected by the models.
        X_raw = np.array([[params[col] for col in INPUT_COLS]], dtype=float)

        # Stage 1: regression
        reg_out: Dict[str, float] = {}
        clipped: List[str] = []
        for target in REGRESSION_TARGETS:
            reg = self.bundle.regressors[target]
            pred = float(reg.predict(X_raw)[0])
            if target in LOG_TRANSFORM_TARGETS:
                pred = float(np.expm1(pred))
            if pred < 0.0:
                clipped.append(target)
            reg_out[target] = max(pred, 0.0)

        # Stage 2: augmented features
        Htot = params["Htot"]
        W_total = self._W_total(params)
        soil_name = sp.INT_TO_SOIL[int(params["soil_ord"])]
        T1 = max(reg_out["T1"], 1e-3)
        Sa_T1 = sp.Sae(T1, soil_name)

        aug = self._derive_augmented(reg_out, Htot, W_total, Sa_T1)

        # Stage 3: classification
        X_aug = np.array(
            [[params[col] for col in INPUT_COLS] + [aug[k] for k in AUG_NAMES]],
            dtype=float)
        clf = self.bundle.classifier
        proba = clf.predict_proba(X_aug)[0]
        verdict_int = int(np.argmax(proba))
        verdict = INT_TO_CLASS[verdict_int]
        class_probabilities = {lbl: float(p)
                               for lbl, p in zip(CLASS_LABELS, proba)}

        # Stage 4: local SHAP
        pool = Pool(X_aug)
        shap_vals = clf.get_feature_importance(type="ShapValues", data=pool)
        # shape: (1, n_classes, n_features + 1) — last column is the base value
        shap_arr = np.asarray(shap_vals)
        base_value = 0.0
        if shap_arr.ndim == 3:
            class_arr = shap_arr[0, verdict_int]
            shap_features = class_arr[:-1]
            base_value = float(class_arr[-1])
        else:
            shap_features = shap_arr[0, :-1]
            base_value = float(shap_arr[0, -1])
        shap_local = {f: float(v) for f, v in zip(ALL_CLF_FEATURES, shap_features)}

        return PredictionResult(
            inputs=dict(params),
            regression=reg_out,
            augmented=aug,
            verdict=verdict,
            verdict_int=verdict_int,
            class_probabilities=class_probabilities,
            shap_local=shap_local,
            base_value=base_value,
            confidence=float(np.max(proba)),
            Sa_T1=float(Sa_T1),
            W_total=float(W_total),
            Htot=float(Htot),
            clipped_targets=clipped,
            notes=clipping_notes(clipped),
        )

    # ----------------------------------------------------------------
    # Batch helper used by the Excel batch window
    # ----------------------------------------------------------------
    def predict_many(self, params_list: List[Dict[str, float]]) -> List[PredictionResult]:
        return [self.predict(p) for p in params_list]


# Convenience top-level function used by tests and the batch window.
def quick_predict(bundle: ModelBundle, params: Dict[str, float]) -> PredictionResult:
    return Predictor(bundle).predict(params)
