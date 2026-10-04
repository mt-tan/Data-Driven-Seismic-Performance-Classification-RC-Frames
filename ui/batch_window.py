"""Batch prediction window.

Reads a spreadsheet of building parameters, runs the cascade pipeline on
every row, and writes the results back to an Excel file. The expected
input layout mirrors the 18 raw inputs used in the GUI, with one row per
building.
"""
from __future__ import annotations

from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
import pandas as pd

from core.predictor import Predictor
from core.model_loader import INPUT_COLS

REQUIRED_COLS = INPUT_COLS  # the dataset's 18 columns


class BatchWindow(ctk.CTkToplevel):
    def __init__(self, master, predictor: Predictor):
        super().__init__(master)
        self.predictor = predictor
        self.title("Batch prediction")
        self.geometry("720x460")
        self.resizable(False, False)
        self.configure(fg_color="#F4F5F7")

        self._build()

    def _build(self):
        # Header card
        card = ctk.CTkFrame(self, fg_color="#FFFFFF", corner_radius=10,
                              border_width=1, border_color="#E0E0E0")
        card.pack(fill="both", expand=True, padx=14, pady=14)

        header = ctk.CTkLabel(
            card, text="Batch prediction from a spreadsheet",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#212121")
        header.pack(padx=18, pady=(18, 8), anchor="w")

        ctk.CTkLabel(
            card, justify="left", wraplength=640,
            text=(
                "Provide an Excel or CSV file with the following columns "
                "(one row per building):\n"
                + ", ".join(REQUIRED_COLS) + ".\n\n"
                "All 18 columns are mandatory. The output file contains the "
                "original inputs, the eight predicted pushover quantities, "
                "the five engineered features, the predicted performance "
                "class, and the four class probabilities."
            ),
            text_color="#555555",
        ).pack(padx=18, pady=4, anchor="w")

        in_row = ctk.CTkFrame(card, fg_color="transparent")
        in_row.pack(fill="x", padx=18, pady=(20, 6))
        ctk.CTkLabel(in_row, text="Input file:", width=80,
                      text_color="#212121").pack(side="left")
        self.in_path_var = ctk.StringVar()
        ctk.CTkEntry(in_row, textvariable=self.in_path_var,
                       width=450, fg_color="#FFFFFF", text_color="#212121",
                       border_color="#D0D0D0").pack(side="left", padx=6)
        ctk.CTkButton(in_row, text="Browse...",
                       command=self._browse_input,
                       fg_color="#E0E0E0", hover_color="#CCCCCC",
                       text_color="#212121").pack(side="left")

        out_row = ctk.CTkFrame(card, fg_color="transparent")
        out_row.pack(fill="x", padx=18, pady=6)
        ctk.CTkLabel(out_row, text="Output file:", width=80,
                      text_color="#212121").pack(side="left")
        self.out_path_var = ctk.StringVar()
        ctk.CTkEntry(out_row, textvariable=self.out_path_var,
                       width=450, fg_color="#FFFFFF", text_color="#212121",
                       border_color="#D0D0D0").pack(side="left", padx=6)
        ctk.CTkButton(out_row, text="Browse...",
                       command=self._browse_output,
                       fg_color="#E0E0E0", hover_color="#CCCCCC",
                       text_color="#212121").pack(side="left")

        self.run_btn = ctk.CTkButton(
            card, text="Run batch prediction", height=38,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2980B9", hover_color="#1F618D",
            text_color="#FFFFFF",
            command=self._run)
        self.run_btn.pack(fill="x", padx=18, pady=(20, 6))

        self.status_label = ctk.CTkLabel(
            card, text="", text_color="#555555",
            font=ctk.CTkFont(size=11))
        self.status_label.pack(padx=18, pady=4, anchor="w")

    # ----------------------------------------------------------------
    def _browse_input(self):
        path = filedialog.askopenfilename(
            filetypes=[("Spreadsheet", "*.xlsx *.csv"),
                       ("Excel", "*.xlsx"),
                       ("CSV", "*.csv")])
        if path:
            self.in_path_var.set(path)
            p = Path(path)
            self.out_path_var.set(str(p.with_name(p.stem + "_predictions.xlsx")))

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")])
        if path:
            self.out_path_var.set(path)

    # ----------------------------------------------------------------
    def _run(self):
        in_path = self.in_path_var.get().strip()
        out_path = self.out_path_var.get().strip()
        if not in_path or not out_path:
            self.status_label.configure(
                text="Please choose both input and output paths.",
                text_color="#B9770E")
            return
        try:
            if in_path.lower().endswith(".csv"):
                df = pd.read_csv(in_path)
            else:
                df = pd.read_excel(in_path)
        except Exception as exc:
            self.status_label.configure(
                text=f"Could not read input: {exc}", text_color="#C0392B")
            return

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            self.status_label.configure(
                text=f"Missing columns: {', '.join(missing)}",
                text_color="#C0392B")
            return

        self.run_btn.configure(state="disabled")
        self.status_label.configure(
            text=f"Predicting on {len(df)} rows...", text_color="#555555")
        self.update_idletasks()

        rows = []
        for _, row in df.iterrows():
            params = {col: float(row[col]) for col in REQUIRED_COLS}
            try:
                r = self.predictor.predict(params)
                out = {**params}
                out.update({f"pred_{k}": v for k, v in r.regression.items()})
                out.update({f"aug_{k}": v for k, v in r.augmented.items()})
                out["verdict"] = r.verdict
                out["confidence"] = r.confidence
                out["note"] = "; ".join(r.notes)
                for cls, p in r.class_probabilities.items():
                    out[f"prob_{cls}"] = p
                rows.append(out)
            except Exception as exc:
                rows.append({**params, "verdict": "ERROR", "error": str(exc)})

        out_df = pd.DataFrame(rows)
        try:
            out_df.to_excel(out_path, index=False)
        except Exception as exc:
            self.status_label.configure(
                text=f"Write failed: {exc}", text_color="#C0392B")
            self.run_btn.configure(state="normal")
            return

        self.status_label.configure(
            text=f"Done. Wrote {len(out_df)} rows to {out_path}",
            text_color="#1E8449")
        self.run_btn.configure(state="normal")
