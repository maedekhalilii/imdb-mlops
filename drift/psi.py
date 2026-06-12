import numpy as np


def compute_psi(reference: list, current: list, bins: int = 10) -> float:
    """
    Population Stability Index (PSI) between two distributions.
    PSI < 0.10  → stable
    PSI 0.10-0.25 → warning
    PSI > 0.25  → alarm, retraining needed
    """
    ref = np.array(reference, dtype=float)
    cur = np.array(current, dtype=float)

    min_val = min(ref.min(), cur.min())
    max_val = max(ref.max(), cur.max())

    # Avoid zero-width bins when all values are identical
    if min_val == max_val:
        return 0.0

    breakpoints = np.linspace(min_val, max_val, bins + 1)

    ref_counts = np.histogram(ref, bins=breakpoints)[0].astype(float)
    cur_counts = np.histogram(cur, bins=breakpoints)[0].astype(float)

    # Smooth to avoid log(0)
    ref_pct = (ref_counts + 1e-6) / (len(ref) + 1e-6 * bins)
    cur_pct = (cur_counts + 1e-6) / (len(cur) + 1e-6 * bins)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 4)


def interpret_psi(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    elif psi < 0.25:
        return "warning"
    return "alarm - retrain needed"
