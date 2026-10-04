"""Lazy loading wrapper for the CatBoost cascade models.

The GUI keeps a single :class:`ModelBundle` instance that holds eight
regression models (one per pushover output plus T1) and one classifier.
Models are loaded from CatBoost ``.cbm`` files produced by
``train_models.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from catboost import CatBoostClassifier, CatBoostRegressor


REGRESSION_TARGETS: List[str] = [
    "Vy", "delta_y", "Vmax", "delta_max",
    "V_at_dt", "delta_t", "delta_u", "T1",
]

INPUT_COLS: List[str] = [
    "N_story", "N_bay", "Lbay", "h1", "htyp", "Htot", "cover",
    "col_B", "col_H", "beam_b", "beam_h",
    "rho_col", "rho_beam_top", "rho_beam_bot",
    "fc_MPa", "G", "Q", "soil_ord",
]

AUG_NAMES: List[str] = [
    "theta_t", "DCR_delta", "mu_d",
    "V_at_dt_over_Vmax", "Vy_over_W_Sa",
]

ALL_CLF_FEATURES = INPUT_COLS + AUG_NAMES

CLASS_TO_INT = {"IO": 0, "LS": 1, "CP": 2, "C": 3}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}
CLASS_LABELS = ["IO", "LS", "CP", "C"]


@dataclass
class ModelBundle:
    regressors: Dict[str, CatBoostRegressor] = field(default_factory=dict)
    classifier: Optional[CatBoostClassifier] = None
    model_dir: Optional[Path] = None
    loaded: bool = False

    def load(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        for target in REGRESSION_TARGETS:
            path = self.model_dir / f"reg_{target}.cbm"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing regression model: {path}\n"
                    "Run train_models.py first.")
            reg = CatBoostRegressor()
            reg.load_model(str(path))
            self.regressors[target] = reg
        clf_path = self.model_dir / "clf_cascade.cbm"
        if not clf_path.exists():
            raise FileNotFoundError(
                f"Missing classifier model: {clf_path}")
        self.classifier = CatBoostClassifier()
        self.classifier.load_model(str(clf_path))
        self.loaded = True

    @property
    def ready(self) -> bool:
        return self.loaded and self.classifier is not None and \
            all(t in self.regressors for t in REGRESSION_TARGETS)
