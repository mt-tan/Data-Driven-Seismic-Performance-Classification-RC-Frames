"""Left-hand input panel of the GUI.

Provides labelled entry fields for the 18 raw building parameters, live
validation (TBEC 2018 and range checks), preset examples loaded from
``assets/examples``, and a Predict button.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import customtkinter as ctk

from core.validator import RANGES, RuleResult, overall_status, validate

# Field definitions: (column key, label text, unit, format)
FIELDS: List[tuple] = [
    ("N_story", "Number of stories", "", "{:.0f}"),
    ("N_bay", "Number of bays", "", "{:.0f}"),
    ("Lbay", "Bay length", "m", "{:.2f}"),
    ("h1", "First story height", "m", "{:.2f}"),
    ("htyp", "Typical story height", "m", "{:.2f}"),
    ("cover", "Concrete cover", "m", "{:.3f}"),
    ("col_B", "Column width", "m", "{:.2f}"),
    ("col_H", "Column depth", "m", "{:.2f}"),
    ("beam_b", "Beam width", "m", "{:.2f}"),
    ("beam_h", "Beam depth", "m", "{:.2f}"),
    ("rho_col", "Column reinf ratio", "—", "{:.4f}"),
    ("rho_beam_top", "Beam top reinf ratio", "—", "{:.4f}"),
    ("rho_beam_bot", "Beam bot reinf ratio", "—", "{:.4f}"),
    ("fc_MPa", "Concrete strength", "MPa", "{:.1f}"),
    ("G", "Dead load", "kN/m", "{:.1f}"),
    ("Q", "Live load", "kN/m", "{:.1f}"),
]

SOIL_OPTIONS = ["ZA", "ZB", "ZC", "ZD", "ZE"]
SOIL_TO_INT = {"ZA": 0, "ZB": 1, "ZC": 2, "ZD": 3, "ZE": 4}

# Default input = preset 02: hold-out frame LHS_31804 of the training campaign (4-story, 3-bay, soil ZD,
# reference class LS). Every value lies inside the sampled domain, so the start-up banner is clean.
DEFAULTS: Dict[str, float] = {
    "N_story": 4,
    "N_bay": 3,
    "Lbay": 3.8,
    "h1": 3.62,
    "htyp": 2.84,
    "cover": 0.049,
    "col_B": 0.54,
    "col_H": 0.54,
    "beam_b": 0.31,
    "beam_h": 0.47,
    "rho_col": 0.0161,
    "rho_beam_top": 0.0094,
    "rho_beam_bot": 0.0048,
    "fc_MPa": 34.8,
    "G": 46.5,
    "Q": 5.5,
    "soil_ord": 3,
}


class InputPanel(ctk.CTkFrame):
    def __init__(self, master, examples_dir: Path,
                 on_change: Optional[Callable[[Dict[str, float]], None]] = None,
                 on_predict: Optional[Callable[[Dict[str, float]], None]] = None,
                 on_batch: Optional[Callable[[], None]] = None):
        super().__init__(master, fg_color="#FFFFFF", corner_radius=10,
                          border_width=1, border_color="#E0E0E0")
        self.examples_dir = examples_dir
        self.on_change = on_change
        self.on_predict = on_predict
        self.on_batch = on_batch

        self.vars: Dict[str, ctk.StringVar] = {}
        self.entries: Dict[str, ctk.CTkEntry] = {}
        self.status_labels: Dict[str, ctk.CTkLabel] = {}

        # Suppress on_change callbacks while building widgets and loading
        # defaults. The flag is raised at the very end of __init__.
        self._ready = False

        self._build()
        self._load_defaults()
        self._validate()
        self._ready = True

    # ----------------------------------------------------------------
    def _build(self):
        header = ctk.CTkLabel(self, text="Building parameters",
                              font=ctk.CTkFont(size=15, weight="bold"),
                              text_color="#212121")
        header.pack(padx=12, pady=(12, 6), anchor="w")

        # Example selector
        ex_row = ctk.CTkFrame(self, fg_color="transparent")
        ex_row.pack(fill="x", padx=12, pady=(0, 8))
        ctk.CTkLabel(ex_row, text="Preset:",
                      text_color="#333333").pack(side="left", padx=(0, 6))
        self.example_var = ctk.StringVar(value="Custom")
        ex_options = ["Custom"] + sorted(
            [p.stem for p in self.examples_dir.glob("*.json")])
        self.example_menu = ctk.CTkOptionMenu(
            ex_row, values=ex_options, variable=self.example_var,
            command=self._apply_example, width=180,
            fg_color="#F5F5F5", button_color="#E0E0E0",
            button_hover_color="#CCCCCC", text_color="#212121",
            dropdown_fg_color="#FFFFFF", dropdown_text_color="#212121")
        self.example_menu.pack(side="left")

        # Scrollable entry area
        self.scroll = ctk.CTkScrollableFrame(self, fg_color="#FAFBFC",
                                              border_width=1,
                                              border_color="#E5E7EA",
                                              height=440)
        self.scroll.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        for key, label, unit, _fmt in FIELDS:
            self._add_entry(self.scroll, key, label, unit)
        self._add_soil_dropdown(self.scroll)

        # Overall validation banner
        self.banner = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=12),
                                    fg_color="#F0F2F5", text_color="#212121",
                                    corner_radius=8, height=32)
        self.banner.pack(fill="x", padx=12, pady=(4, 8))

        # Action buttons
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(0, 12))

        self.predict_btn = ctk.CTkButton(
            btn_row, text="Predict", height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#2980B9", hover_color="#1F618D",
            text_color="#FFFFFF",
            command=self._predict_clicked)
        self.predict_btn.pack(fill="x", pady=(0, 6))

        sub_row = ctk.CTkFrame(btn_row, fg_color="transparent")
        sub_row.pack(fill="x")
        ctk.CTkButton(sub_row, text="Reset", command=self._load_defaults,
                       fg_color="#E0E0E0", hover_color="#CCCCCC",
                       text_color="#212121", width=120).pack(
                           side="left", padx=(0, 4), expand=True, fill="x")
        ctk.CTkButton(sub_row, text="Batch (Excel)", command=self._batch_clicked,
                       fg_color="#E0E0E0", hover_color="#CCCCCC",
                       text_color="#212121", width=120).pack(
                           side="left", padx=(4, 0), expand=True, fill="x")

    def _add_entry(self, parent, key, label, unit):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=4, pady=3)
        ctk.CTkLabel(row, text=label, width=170, anchor="w",
                      text_color="#212121").pack(side="left")
        var = ctk.StringVar()
        ent = ctk.CTkEntry(row, textvariable=var, width=90,
                            fg_color="#FFFFFF", text_color="#212121",
                            border_color="#D0D0D0")
        ent.pack(side="left", padx=(4, 4))
        ctk.CTkLabel(row, text=unit, width=40, anchor="w",
                      text_color="#777777").pack(side="left")
        status = ctk.CTkLabel(row, text="●", width=12,
                                text_color="#1E8449",
                                font=ctk.CTkFont(size=14, weight="bold"))
        status.pack(side="left")
        var.trace_add("write", lambda *_: self._on_field_change())
        self.vars[key] = var
        self.entries[key] = ent
        self.status_labels[key] = status

    def _add_soil_dropdown(self, parent):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=4, pady=3)
        ctk.CTkLabel(row, text="Soil class (TBEC 2018)",
                      width=170, anchor="w",
                      text_color="#212121").pack(side="left")
        self.soil_var = ctk.StringVar(value="ZC")
        menu = ctk.CTkOptionMenu(row, values=SOIL_OPTIONS, variable=self.soil_var,
                                  command=lambda *_: self._on_field_change(),
                                  width=90,
                                  fg_color="#FFFFFF", button_color="#E0E0E0",
                                  button_hover_color="#CCCCCC",
                                  text_color="#212121",
                                  dropdown_fg_color="#FFFFFF",
                                  dropdown_text_color="#212121")
        menu.pack(side="left", padx=(4, 4))
        ctk.CTkLabel(row, text="—", width=40, anchor="w",
                      text_color="#777777").pack(side="left")

    # ----------------------------------------------------------------
    def _load_defaults(self):
        for key, _label, _unit, fmt in FIELDS:
            self.vars[key].set(fmt.format(DEFAULTS[key]))
        self.soil_var.set("ZC")
        self.example_var.set("Custom")
        self._validate()
        # Only notify the host after initial widget setup is finished.
        if getattr(self, "_ready", False) and self.on_change:
            self.on_change(self.get_values())

    def _apply_example(self, name: str):
        if name == "Custom":
            return
        path = self.examples_dir / f"{name}.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        self._applying_preset = True          # field traces must not reset the selector to "Custom"
        try:
            for key, _label, _unit, fmt in FIELDS:
                if key in data:
                    self.vars[key].set(fmt.format(data[key]))
            if "soil_ord" in data:
                for s, ord_ in SOIL_TO_INT.items():
                    if ord_ == int(data["soil_ord"]):
                        self.soil_var.set(s)
                        break
        finally:
            self._applying_preset = False
        self.example_var.set(name)
        self._validate()
        if self.on_change:
            self.on_change(self.get_values())

    # ----------------------------------------------------------------
    def get_values(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, _label, _unit, _fmt in FIELDS:
            try:
                out[key] = float(self.vars[key].get())
            except (ValueError, TypeError):
                out[key] = float("nan")
        out["soil_ord"] = float(SOIL_TO_INT[self.soil_var.get()])
        out["Htot"] = (out.get("h1", 0.0)
                       + (out.get("N_story", 0.0) - 1) * out.get("htyp", 0.0))
        return out

    # ----------------------------------------------------------------
    def _on_field_change(self):
        if not getattr(self, "_ready", False) or getattr(self, "_applying_preset", False):
            # Skip while widgets are still being initialized or a preset is being written.
            return
        self.example_var.set("Custom")
        self._validate()
        if self.on_change:
            self.on_change(self.get_values())

    def _validate(self):
        params = self.get_values()
        try:
            results = validate(params)
        except Exception:
            return
        rng_dict = {r.rule: r for r in results["ranges"]}
        for key in self.status_labels:
            r = rng_dict.get(key)
            if r is None:
                self.status_labels[key].configure(text="?", text_color="#B9770E")
                continue
            if r.passed:
                self.status_labels[key].configure(text="●", text_color="#1E8449")
            else:
                self.status_labels[key].configure(text="●", text_color="#C0392B")

        status = overall_status(results)
        errors = sum(1 for r in results["ranges"] + results["tbdy"]
                     if not r.passed and r.severity == "error")
        warns = sum(1 for r in results["ranges"] + results["tbdy"]
                    if not r.passed and r.severity == "warning")

        if status == "ok":
            self.banner.configure(text="All checks passed — inputs are valid.",
                                    fg_color="#D5F5E3", text_color="#0B5345")
        elif status == "warning":
            self.banner.configure(
                text=f"{warns} warning(s); prediction permitted but review inputs.",
                fg_color="#FCF3CF", text_color="#7E5109")
        else:
            self.banner.configure(
                text=f"{errors} error(s) and {warns} warning(s); fix inputs.",
                fg_color="#FADBD8", text_color="#922B21")
        self._last_validation = results

    def get_validation(self) -> Dict[str, List[RuleResult]]:
        return getattr(self, "_last_validation", validate(self.get_values()))

    # ----------------------------------------------------------------
    def _predict_clicked(self):
        if self.on_predict:
            self.on_predict(self.get_values())

    def _batch_clicked(self):
        if self.on_batch:
            self.on_batch()
