"""Pushover Performance Predictor — main application.

A CustomTkinter desktop application that lets engineers enter the 18 raw
design parameters of an RC frame building (per TBEC 2018), runs the
cascade ML pipeline, and presents the predicted pushover quantities,
performance verdict, local SHAP explanation, and a downloadable PDF
report. Batch prediction from Excel is supported through a secondary
window.

Run after :file:`train_models.py` has populated ``assets/models``:

    python app.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.model_loader import ModelBundle
from core.predictor import Predictor, PredictionResult
from reports.pdf_template import generate_report
from ui.batch_window import BatchWindow
from ui.input_panel import InputPanel
from ui.result_panel import ResultPanel
from ui.structure_canvas import render_structure


APP_TITLE = "Pushover Performance Predictor - TBEC 2018"
APP_VERSION = "1.0.0"

APP_ROOT = Path(__file__).resolve().parent
MODELS_DIR = APP_ROOT / "assets" / "models"
EXAMPLES_DIR = APP_ROOT / "assets" / "examples"


class PushoverApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.title(APP_TITLE)
        self.geometry("1400x880")
        self.minsize(1200, 760)
        self.configure(fg_color="#F4F5F7")

        self.bundle = ModelBundle()
        self.predictor: Predictor | None = None
        self.last_result: PredictionResult | None = None

        self._build_menu()
        self._build_layout()

        # Defer model loading slightly so the UI appears responsive at start.
        self.after(150, self._load_models)

    # ------------------------------------------------------------------
    def _build_menu(self):
        # Top toolbar
        bar = ctk.CTkFrame(self, fg_color="#FFFFFF", height=46, corner_radius=0,
                            border_width=0)
        bar.pack(fill="x", side="top")

        ctk.CTkLabel(bar, text=APP_TITLE,
                      font=ctk.CTkFont(size=15, weight="bold"),
                      text_color="#212121").pack(side="left", padx=14)

        # Right-side buttons
        self.export_btn = ctk.CTkButton(
            bar, text="Export PDF report", width=170,
            command=self._export_pdf, fg_color="#1E5631",
            hover_color="#196F3D", text_color="#FFFFFF",
            state="disabled")
        self.export_btn.pack(side="right", padx=(6, 14), pady=6)

        ctk.CTkButton(
            bar, text="About", width=80,
            command=self._show_about, fg_color="#E0E0E0",
            hover_color="#CCCCCC", text_color="#212121"
            ).pack(side="right", padx=4, pady=6)

    def _build_layout(self):
        body = ctk.CTkFrame(self, fg_color="#F4F5F7")
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=0, minsize=380)
        body.grid_columnconfigure(1, weight=0, minsize=320)
        body.grid_columnconfigure(2, weight=1)
        body.grid_rowconfigure(0, weight=1)

        # Left: input panel
        self.input_panel = InputPanel(
            body, examples_dir=EXAMPLES_DIR,
            on_change=self._on_inputs_change,
            on_predict=self._on_predict,
            on_batch=self._on_batch)
        self.input_panel.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)

        # Middle: structure preview canvas
        self.struct_frame = ctk.CTkFrame(body, fg_color="#FFFFFF",
                                           corner_radius=10, border_width=1,
                                           border_color="#E0E0E0")
        self.struct_frame.grid(row=0, column=1, sticky="nsew", padx=4, pady=8)
        ctk.CTkLabel(self.struct_frame, text="Structure preview",
                      font=ctk.CTkFont(size=14, weight="bold"),
                      text_color="#212121"
                      ).pack(anchor="w", padx=12, pady=(10, 4))
        self.struct_fig = Figure(figsize=(3.6, 8.0), dpi=100, facecolor="#FFFFFF")
        self.struct_canvas = FigureCanvasTkAgg(self.struct_fig, master=self.struct_frame)
        self.struct_canvas.draw()
        self.struct_canvas.get_tk_widget().pack(fill="both", expand=True,
                                                  padx=8, pady=(0, 10))

        # Right: result panel
        self.result_panel = ResultPanel(body)
        self.result_panel.grid(row=0, column=2, sticky="nsew", padx=(4, 8), pady=8)

        # Status bar
        self.status_bar = ctk.CTkLabel(
            self, text="Initializing…",
            anchor="w", fg_color="#FFFFFF", text_color="#666666",
            font=ctk.CTkFont(size=11), height=26)
        self.status_bar.pack(fill="x", side="bottom")

        # Initial structure render with default values
        self._render_structure(self.input_panel.get_values())

    # ------------------------------------------------------------------
    def _load_models(self):
        if not MODELS_DIR.exists() or not any(MODELS_DIR.glob("*.cbm")):
            self._set_status(
                "Models not found. Run train_models.py first "
                "(see README.md).", "warn")
            return
        self._set_status("Loading models…")
        try:
            self.bundle.load(MODELS_DIR)
            self.predictor = Predictor(self.bundle)
            self._set_status(f"Models loaded from {MODELS_DIR}. Ready.", "ok")
        except Exception as exc:
            self.predictor = None
            self._set_status(f"Model loading failed: {exc}", "err")
            traceback.print_exc()

    # ------------------------------------------------------------------
    def _on_inputs_change(self, params: dict):
        # Re-render the structure preview on any change
        try:
            self._render_structure(params)
        except Exception:
            traceback.print_exc()

    def _render_structure(self, params: dict):
        render_structure(self.struct_fig, params)
        self.struct_canvas.draw_idle()

    # ------------------------------------------------------------------
    def _on_predict(self, params: dict):
        if self.predictor is None:
            messagebox.showwarning(
                "Models not loaded",
                "ML models are not loaded. Run train_models.py first.")
            return
        try:
            self._set_status("Running cascade pipeline…")
            self.update_idletasks()
            result = self.predictor.predict(params)
            self.last_result = result
            self.result_panel.show(result)
            self.export_btn.configure(state="normal")
            self._set_status(
                f"Verdict: {result.verdict} "
                f"({result.confidence * 100:.1f}% confidence).", "ok")
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Prediction failed",
                                  f"An error occurred:\n{exc}")
            self._set_status(f"Prediction failed: {exc}", "err")

    def _on_batch(self):
        if self.predictor is None:
            messagebox.showwarning(
                "Models not loaded",
                "ML models are not loaded. Run train_models.py first.")
            return
        BatchWindow(self, self.predictor)

    # ------------------------------------------------------------------
    def _export_pdf(self):
        if self.last_result is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF document", "*.pdf")],
            initialfile="pushover_report.pdf")
        if not path:
            return
        try:
            generate_report(self.last_result, Path(path))
            self._set_status(f"PDF report saved to {path}", "ok")
            messagebox.showinfo("Report saved",
                                  f"PDF report saved to:\n{path}")
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Export failed",
                                  f"Could not save report:\n{exc}")

    # ------------------------------------------------------------------
    def _show_about(self):
        messagebox.showinfo(
            "About",
            f"{APP_TITLE}\n"
            f"Version {APP_VERSION}\n\n"
            "Cascade machine learning framework for predicting the "
            "pushover-based performance level of RC frame buildings "
            "designed per the Turkish Building Earthquake Code "
            "(TBEC 2018). Calibrated on 48,725 nonlinear pushover "
            "analyses generated in OpenSeesPy.\n\n"
            "Use as a decision-support tool only. Final assessment "
            "requires full nonlinear analysis and engineering judgement.")

    # ------------------------------------------------------------------
    def _set_status(self, text: str, level: str = "info"):
        colors = {"info": "#666666", "ok": "#1E8449",
                  "warn": "#B9770E", "err": "#C0392B"}
        self.status_bar.configure(text=text,
                                    text_color=colors.get(level, "#666666"))


def main():
    app = PushoverApp()
    app.mainloop()


if __name__ == "__main__":
    main()
