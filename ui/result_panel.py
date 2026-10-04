"""Right-hand result panel: verdict, pushover curve, numerical table, SHAP."""
from __future__ import annotations

from typing import Optional

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core.predictor import PredictionResult
from core.formatting import fmt_prob


CLASS_COLORS = {
    "IO": "#1E8449",
    "LS": "#D68910",
    "CP": "#CA6F1E",
    "C":  "#922B21",
}
CLASS_DESCRIPTIONS = {
    "IO": "Immediate Occupancy",
    "LS": "Life Safety",
    "CP": "Collapse Prevention",
    "C":  "Collapse",
}
IDLE_COLOR = "#F0F2F5"
PANEL_BG = "#FFFFFF"
PANEL_BORDER = "#E0E0E0"
INNER_BG = "#FAFBFC"
TEXT_PRIMARY = "#212121"
TEXT_SECONDARY = "#555555"
TEXT_MUTED = "#888888"


class ResultPanel(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color=PANEL_BG,
                          corner_radius=10, border_width=1,
                          border_color=PANEL_BORDER)
        self.last_result: Optional[PredictionResult] = None
        self._build()

    # ----------------------------------------------------------------
    def _build(self):
        # Top row: verdict | pushover plot
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="both", expand=True, padx=12, pady=(12, 6))
        top.grid_columnconfigure(0, weight=1)
        top.grid_columnconfigure(1, weight=2)
        top.grid_rowconfigure(0, weight=1)

        # Verdict box
        self.verdict_frame = ctk.CTkFrame(top, fg_color=IDLE_COLOR,
                                           corner_radius=12,
                                           border_width=1,
                                           border_color=PANEL_BORDER)
        self.verdict_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.verdict_label = ctk.CTkLabel(
            self.verdict_frame, text="—",
            font=ctk.CTkFont(size=64, weight="bold"),
            text_color=TEXT_MUTED)
        self.verdict_label.pack(pady=(28, 4))

        self.verdict_desc = ctk.CTkLabel(
            self.verdict_frame, text="Press Predict to evaluate",
            font=ctk.CTkFont(size=13),
            text_color=TEXT_SECONDARY)
        self.verdict_desc.pack(pady=(0, 8))

        self.confidence_label = ctk.CTkLabel(
            self.verdict_frame, text="",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED)
        self.confidence_label.pack(pady=(0, 6))

        self.safety_badge = ctk.CTkLabel(
            self.verdict_frame, text="",
            font=ctk.CTkFont(size=11, weight="bold"),
            corner_radius=8, fg_color="#E0E0E0",
            text_color=TEXT_PRIMARY,
            width=160, height=24)
        self.safety_badge.pack(pady=(2, 6))

        # Caveat line (e.g. a regressor output clipped at zero); empty when there is nothing to report
        self.note_label = ctk.CTkLabel(
            self.verdict_frame, text="",
            font=ctk.CTkFont(size=10),
            text_color="#FFF3CD", wraplength=250, justify="left")
        self.note_label.pack(padx=12, pady=(0, 8))

        # Class probability bar chart (Tkinter canvas)
        self.proba_canvas = ctk.CTkCanvas(
            self.verdict_frame, height=120, bg=IDLE_COLOR,
            highlightthickness=0)
        self.proba_canvas.pack(fill="x", padx=12, pady=(0, 12))

        # Pushover curve
        self.curve_frame = ctk.CTkFrame(top, fg_color=INNER_BG,
                                          corner_radius=8,
                                          border_width=1,
                                          border_color=PANEL_BORDER)
        self.curve_frame.grid(row=0, column=1, sticky="nsew")

        self.curve_fig = Figure(figsize=(6.0, 3.6), dpi=100, facecolor=INNER_BG)
        self.curve_ax = self.curve_fig.add_subplot(111)
        self._style_axes(self.curve_ax)
        self.curve_canvas = FigureCanvasTkAgg(self.curve_fig, master=self.curve_frame)
        self.curve_canvas.draw()
        self.curve_canvas.get_tk_widget().pack(fill="both", expand=True,
                                                  padx=8, pady=8)

        # Bottom row: numerical table | SHAP waterfall
        bot = ctk.CTkFrame(self, fg_color="transparent")
        bot.pack(fill="both", expand=True, padx=12, pady=(6, 12))
        bot.grid_columnconfigure(0, weight=1)
        bot.grid_columnconfigure(1, weight=1)
        bot.grid_rowconfigure(0, weight=1)

        # Numerical results table
        self.table_frame = ctk.CTkFrame(bot, fg_color=INNER_BG,
                                          corner_radius=8,
                                          border_width=1,
                                          border_color=PANEL_BORDER)
        self.table_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ctk.CTkLabel(self.table_frame, text="Predicted pushover quantities",
                      font=ctk.CTkFont(size=12, weight="bold"),
                      text_color=TEXT_PRIMARY
                      ).pack(anchor="w", padx=10, pady=(8, 4))
        self.table_text = ctk.CTkTextbox(
            self.table_frame, fg_color="#FFFFFF", text_color=TEXT_PRIMARY,
            border_width=1, border_color=PANEL_BORDER,
            font=ctk.CTkFont(family="Consolas", size=11), wrap="none")
        self.table_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.table_text.configure(state="disabled")

        # SHAP waterfall
        self.shap_frame = ctk.CTkFrame(bot, fg_color=INNER_BG,
                                         corner_radius=8,
                                         border_width=1,
                                         border_color=PANEL_BORDER)
        self.shap_frame.grid(row=0, column=1, sticky="nsew")
        self.shap_fig = Figure(figsize=(5.0, 3.6), dpi=100, facecolor=INNER_BG)
        self.shap_ax = self.shap_fig.add_subplot(111)
        self._style_axes(self.shap_ax)
        self.shap_canvas = FigureCanvasTkAgg(self.shap_fig, master=self.shap_frame)
        self.shap_canvas.draw()
        self.shap_canvas.get_tk_widget().pack(fill="both", expand=True,
                                                padx=8, pady=8)

    # ----------------------------------------------------------------
    @staticmethod
    def _style_axes(ax):
        """Light-theme axis styling with larger fonts."""
        ax.set_facecolor(INNER_BG)
        for s in ax.spines.values():
            s.set_color("#C8CCD1")
            s.set_linewidth(0.8)
        ax.tick_params(colors="#333333", labelsize=11)
        ax.yaxis.label.set_color("#333333")
        ax.yaxis.label.set_size(12)
        ax.xaxis.label.set_color("#333333")
        ax.xaxis.label.set_size(12)
        ax.title.set_color(TEXT_PRIMARY)
        ax.title.set_size(12)
        ax.grid(alpha=0.20, color="#888888", linestyle=":")

    # ----------------------------------------------------------------
    def show(self, result: PredictionResult):
        self.last_result = result
        self._update_verdict(result)
        self._update_pushover_curve(result)
        self._update_table(result)
        self._update_shap(result)

    # ----------------------------------------------------------------
    def _update_verdict(self, r: PredictionResult):
        color = CLASS_COLORS.get(r.verdict, "#666666")
        self.verdict_frame.configure(fg_color=color, border_color=color)
        self.verdict_label.configure(text=r.verdict, text_color="#FFFFFF")
        self.verdict_desc.configure(
            text=CLASS_DESCRIPTIONS.get(r.verdict, ""),
            text_color="#FFFFFF")
        self.confidence_label.configure(
            text=f"Confidence  {fmt_prob(r.confidence)}",
            text_color="#FFFFFF")

        # Safety badge: direction of a possible error. Probability mass on MORE severe
        # classes than the verdict means the verdict may be optimistic (unsafe side);
        # mass on LESS severe classes means the verdict is conservative (safe side).
        order = {"IO": 0, "LS": 1, "CP": 2, "C": 3}
        opt = sum(p for cls, p in r.class_probabilities.items()
                  if order[cls] > r.verdict_int)      # mass on worse classes
        cons = sum(p for cls, p in r.class_probabilities.items()
                   if order[cls] < r.verdict_int)     # mass on better classes
        if opt > cons and opt > 0.05:
            self.safety_badge.configure(
                text="Optimistic risk  (review)",
                fg_color="#7B241C", text_color="#FFFFFF")
        elif cons > 0.05:
            self.safety_badge.configure(
                text="Conservative (safe side)",
                fg_color="#1B4F72", text_color="#FFFFFF")
        else:
            self.safety_badge.configure(
                text="Sharp classification",
                fg_color="#117A65", text_color="#FFFFFF")
        self.note_label.configure(text=("Note: " + " ".join(r.notes)) if r.notes else "")

        # Probability bars — match canvas background to verdict frame color
        canvas = self.proba_canvas
        canvas.configure(bg=color)
        canvas.delete("all")
        canvas.update_idletasks()
        w = max(canvas.winfo_width(), 240)
        margin = 14
        bar_h = 18
        for i, cls in enumerate(["IO", "LS", "CP", "C"]):
            y = margin + i * (bar_h + 6)
            canvas.create_text(margin, y + bar_h / 2,
                                text=cls, fill="#FFFFFF",
                                anchor="w", font=("Arial", 10, "bold"))
            p = r.class_probabilities[cls]
            x0 = margin + 26
            x1 = x0 + int((w - x0 - margin - 60) * max(p, 0.01))
            # White-tinted bar fills, semi-transparent feel
            canvas.create_rectangle(x0, y, x1, y + bar_h,
                                      fill="#FFFFFF", outline="",
                                      stipple="gray50")
            canvas.create_text(w - margin - 30, y + bar_h / 2,
                                text=fmt_prob(p),
                                fill="#FFFFFF",
                                anchor="w", font=("Arial", 10))

    def _update_pushover_curve(self, r: PredictionResult):
        ax = self.curve_ax
        ax.clear()
        self._style_axes(ax)

        reg = r.regression
        x = [0.0, reg["delta_y"], reg["delta_max"], reg["delta_u"]]
        y = [0.0, reg["Vy"], reg["Vmax"], reg["Vmax"] * 0.8]

        ax.plot(x, y, "-", lw=2.5, color="#2980B9", label="Capacity curve")
        ax.scatter([reg["delta_y"]], [reg["Vy"]], c="#1E8449", s=100,
                    zorder=5, label="Yield", edgecolors="white", linewidths=1)
        ax.scatter([reg["delta_max"]], [reg["Vmax"]], c="#F39C12", s=100,
                    zorder=5, label="Peak", edgecolors="white", linewidths=1)
        ax.scatter([reg["delta_t"]], [reg["V_at_dt"]], c="#E67E22", s=140,
                    marker="*", zorder=5, label=r"Target $\delta_t$",
                    edgecolors="white", linewidths=0.8)
        ax.scatter([reg["delta_u"]], [reg["Vmax"] * 0.8], c="#C0392B", s=100,
                    zorder=5, label="Ultimate", edgecolors="white", linewidths=1)

        ax.axvline(reg["delta_t"], ls=":", lw=0.9, color="#888888")
        ax.set_xlabel("Roof displacement [m]")
        ax.set_ylabel("Base shear [kN]")
        ax.set_title(f"Predicted pushover capacity   (T1 = {reg['T1']:.2f} s)")
        ax.legend(loc="lower right", fontsize=10, facecolor="#FFFFFF",
                  labelcolor=TEXT_PRIMARY, edgecolor=PANEL_BORDER,
                  framealpha=0.95)
        self.curve_fig.tight_layout()
        self.curve_canvas.draw_idle()

    def _update_table(self, r: PredictionResult):
        text = (
            f"{'Quantity':<38}{'Value':>14}\n"
            f"{'-' * 54}\n"
            f"{'Yield base shear V_y [kN]':<38}{r.regression['Vy']:>14.1f}\n"
            f"{'Yield displacement delta_y [m]':<38}{r.regression['delta_y']:>14.4f}\n"
            f"{'Maximum base shear V_max [kN]':<38}{r.regression['Vmax']:>14.1f}\n"
            f"{'Disp at peak delta_max [m]':<38}{r.regression['delta_max']:>14.4f}\n"
            f"{'Base shear at target V@dt [kN]':<38}{r.regression['V_at_dt']:>14.1f}\n"
            f"{'Target displacement delta_t [m]':<38}{r.regression['delta_t']:>14.4f}\n"
            f"{'Ultimate displacement delta_u [m]':<38}{r.regression['delta_u']:>14.4f}\n"
            f"{'Fundamental period T1 [s]':<38}{r.regression['T1']:>14.3f}\n"
            f"\n{'Augmented feature':<38}{'Value':>14}\n"
            f"{'-' * 54}\n"
            f"{'Roof drift ratio (dt/H)':<38}{r.augmented['theta_t']:>14.5f}\n"
            f"{'Disp demand/capacity (dt/du)':<38}{r.augmented['DCR_delta']:>14.4f}\n"
            f"{'Displacement ductility (dt/dy)':<38}{r.augmented['mu_d']:>14.3f}\n"
            f"{'Post-peak strength (V@dt/Vmax)':<38}{r.augmented['V_at_dt_over_Vmax']:>14.4f}\n"
            f"{'Yield coefficient Vy/(W Sa(T1))':<38}{r.augmented['Vy_over_W_Sa']:>14.5f}\n"
            f"\n{'Context':<38}{'Value':>14}\n"
            f"{'-' * 54}\n"
            f"{'Total height H_tot [m]':<38}{r.Htot:>14.2f}\n"
            f"{'Total seismic weight W [kN]':<38}{r.W_total:>14.1f}\n"
            f"{'Spectral acceleration Sa(T1) [g]':<38}{r.Sa_T1:>14.4f}\n"
        )
        self.table_text.configure(state="normal")
        self.table_text.delete("0.0", "end")
        self.table_text.insert("0.0", text)
        self.table_text.configure(state="disabled")

    def _update_shap(self, r: PredictionResult):
        ax = self.shap_ax
        ax.clear()
        self._style_axes(ax)
        items = sorted(r.shap_local.items(), key=lambda kv: abs(kv[1]),
                       reverse=True)[:10]
        items = list(reversed(items))
        names = [k for k, _ in items]
        vals = [v for _, v in items]
        colors = ["#C0392B" if v < 0 else "#2980B9" for v in vals]
        y_pos = list(range(len(names)))
        ax.barh(y_pos, vals, color=colors, edgecolor="#FFFFFF", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=11, color=TEXT_PRIMARY)
        ax.axvline(0, color="#444444", lw=0.7)
        ax.set_xlabel("SHAP value")
        ax.set_title(f"Local explanation for class {r.verdict}")
        self.shap_fig.tight_layout()
        self.shap_canvas.draw_idle()
