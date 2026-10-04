"""Display helpers shared by the result panel, the PDF report and the batch window."""
from __future__ import annotations


def fmt_prob(p: float) -> str:
    """Probability for display. Saturated values are shown as bounds so that the GUI never
    claims 100.0 % or 0.0 % confidence (the classifier outputs are calibrated only within
    its test-set accuracy)."""
    if p > 0.9995:
        return ">= 99.9 %"
    if p < 0.0005:
        return "<= 0.1 %"
    return f"{p * 100:.1f} %"
