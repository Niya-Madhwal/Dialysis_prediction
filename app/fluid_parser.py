# fluid_parser.py

import math
from typing import Dict, List, Optional

def predict_dilution_spike(
    baseline_kft: Dict[str, float],
    intake_totals: Dict[str, float],
    weight_kg: float,
    breakdown: Optional[List[Dict]] = None,
    tbw_fraction: float = 0.5,
    rate_cap_na_per_h: float = 2.0,
    k_shift_tau_h: float = 1.2,
    k_ecf_fraction_initial: float = 0.4,
) -> Dict[str, float]:
    """
    Oral-adapted 1-hour estimate using per-item precomputed fields:
      - item['na_conc_mmol_per_L'] and item['absorption_t_half_h']
      - item['Water_ml'], item['Potassium_mg']
    """
    TBW = weight_kg * tbw_fraction                     # L
    ECF = max(0.2 * weight_kg, 1e-6)                   # L

    Na_b = float(baseline_kft.get("Sodium", 138.0))
    K_b  = float(baseline_kft.get("Potassium", 4.5))

    # ---------- Sodium: per-item mixing over ~1 hour ----------
    V_abs_total = 0.0   # L absorbed
    Na_added_mEq = 0.0  # mEq from absorbed volumes at item drink concentration

    # For K absorption accumulation
    K_abs_mEq_total_1h = 0.0

    if breakdown:
        for item in breakdown:
            water_ml = float(item.get("Water_ml", 0.0) or 0.0)
            water_L  = water_ml / 1000.0
            # Allow K absorption even for solids
            t_half_h = float(item.get("absorption_t_half_h", 1.2) or 1.2)
            k_abs = math.log(2) / max(t_half_h, 1e-6)
            frac_abs_1h = 1.0 - math.exp(-k_abs * 1.0)
            K_abs_mEq_total_1h += (float(item.get("Potassium_mg", 0.0) or 0.0) / 39.1) * frac_abs_1h

            if water_L <= 0.0:
                continue

            # Per-item sodium mixing
            na_conc = float(item.get("na_conc_mmol_per_L", 5.0) or 5.0)  # mmol/L
            V_abs = water_L * frac_abs_1h  # 1-hour absorbed volume
            V_abs_total += V_abs
            Na_added_mEq += na_conc * V_abs  # mmol/L * L = mEq

    else:
        # Fallback to totals if breakdown is missing
        V_in_L = max(float(intake_totals.get("Water_ml", 0.0) or 0.0), 0.0) / 1000.0
        # Assume hypotonic drink ~5 mmol/L and ~60% absorption in 1 h (t½ ~40m)
        frac_abs_1h = 1.0 - math.exp(-1.04 * 1.0)
        V_abs_total = V_in_L * frac_abs_1h
        Na_added_mEq = 5.0 * V_abs_total
        K_abs_mEq_total_1h = (float(intake_totals.get("Potassium_mg", 0.0) or 0.0) / 39.1) * frac_abs_1h

    # Mix Na with TBW and cap the hourly change
    if V_abs_total > 0.0:
        Na_est = (TBW * Na_b + Na_added_mEq) / (TBW + V_abs_total)
        Na_est = max(Na_b - rate_cap_na_per_h, min(Na_b + rate_cap_na_per_h, Na_est))
    else:
        Na_est = Na_b

    # ---------- Potassium: partial ECF appearance, then insulin-driven shift ----------
    K_mEq_to_ecf = K_abs_mEq_total_1h * k_ecf_fraction_initial
    dK_ecf = K_mEq_to_ecf / ECF  # mmol/L

    shift_fraction_1h = 1.0 - math.exp(-1.0 / k_shift_tau_h)  # ~0.56 if tau=1.2h
    dK_shift = dK_ecf * shift_fraction_1h
    K_est = K_b + (dK_ecf - dK_shift)

    V_in_total_ml = float(intake_totals.get("Water_ml", 0.0) or 0.0)
    return {
        "Water_ml": round(V_in_total_ml, 2),
        "Sodium_new": round(Na_est, 2),
        "ΔSodium": round(Na_est - Na_b, 2),
        "Potassium_new": round(K_est, 2),
        "ΔPotassium": round(K_est - K_b, 2),
    }

# ----------------
# Quick test/demo
# ----------------
if __name__ == "__main__":
    # Example usage with food_lookup_new-style breakdown
    breakdown = [
        {'food': 'Coconut Water', 'quantity_g': 300.0, 'matched': True, 'Water_ml': 285.0, 'Sodium_mg': 315.0,
         'Potassium_mg': 750.0, 'na_conc_mmol_per_L': 15.0, 'absorption_t_half_h': 0.67},
        {'food': 'Tap Water', 'quantity_g': 1000.0, 'matched': True, 'Water_ml': 999.0, 'Sodium_mg': 40.0,
         'Potassium_mg': 0.0, 'na_conc_mmol_per_L': 2.0, 'absorption_t_half_h': 0.25}
    ]
    totals = {"Water_ml": 1284.0, "Potassium_mg": 750.0}
    baseline = {"Sodium": 138.0, "Potassium": 4.5}
    out = predict_dilution_spike(baseline, totals, weight_kg=70.0, breakdown=breakdown)
    print(out)