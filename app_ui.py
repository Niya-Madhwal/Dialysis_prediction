# app_ui.py

import os
import json
import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st



# --- Normal reference ranges for KFT values ---
KFT_NORMAL_RANGES = {
    "Sodium":      (135, 145),   # mEq/L
    "Potassium":   (3.5, 5.1),   # mEq/L
    "Creatinine":  (0.7, 1.2),   # mg/dL
    "Urea":        (10, 50),     # mg/dL
    "Phosphorus":  (2.5, 4.5),   # mg/dL
}


# ---------- Paths ----------
APP_DIR = os.path.join(os.path.dirname(__file__), "app")
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_CSV = os.path.join(DATA_DIR, "diet_history.csv")

# ---------- Imports from your modules ----------
# Food lookup (prefer newer)
breakdown_food_items = None
try:
    from app.food_lookup_new import breakdown_food_items as _bf_new
    breakdown_food_items = _bf_new
except Exception:
    try:
        from app.food_lookup import breakdown_food_items as _bf_old
        breakdown_food_items = _bf_old
    except Exception:
        pass

# Fluid parser
predict_dilution_spike = None
try:
    from app.fluid_parser import predict_dilution_spike as _pds
    predict_dilution_spike = _pds
except Exception:
    pass

# KFT parser (fallback: return dummy values)
def _fallback_parse_kft(_: bytes) -> dict:
    return {"Sodium": 138.0, "Potassium": 4.5, "Creatinine": 7.2, "Urea": 95.0}

parse_kft_report = _fallback_parse_kft
estimate_parse_quality = lambda _v: 0.0  # type: ignore
try:
    from app.kft_parser import parse_kft_report as _parse_kft_report, estimate_parse_quality as _estimate_parse_quality
    parse_kft_report = _parse_kft_report
    estimate_parse_quality = _estimate_parse_quality
except Exception:
    try:
        from app.kft_parser import parse_kft as _parse_kft_report
        parse_kft_report = _parse_kft_report
    except Exception:
        pass

# Optional: Load trained Keras model and scaler
from tensorflow import keras
import joblib

MODEL_PATH = os.path.join("app", "Kidney_dialysis_datamodel.h5")
SCALER_PATH = os.path.join("app", "scaler.pkl")
model = None
scaler = None
if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH, compile=False)
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)

# ---------- UI ----------
st.set_page_config(page_title="Kidney Monitor – E2E", layout="wide")
st.title("🩺 Kidney Monitor – End-to-End Tester")

with st.sidebar:
    st.header("Session")
    patient_id = st.text_input("Patient ID", value="TEST001")
    weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.5)
    tbw_fraction = st.selectbox("TBW fraction", options=[0.5, 0.6], index=0)
    session_date = st.date_input("Session date", value=dt.date.today())

# ------------------- 1) Upload & parse KFT -------------------
st.subheader("1) Upload KFT report")
uploaded = st.file_uploader("Upload KFT PDF/PNG/JPG", type=["pdf", "png", "jpg", "jpeg"])

# Parsed KFT values (may be empty)
baseline_kft: dict = {}
parse_quality = None
PARSE_QUALITY_THRESHOLD = 0.6
if uploaded:
    b = uploaded.read()
    try:
        baseline_kft = parse_kft_report(b)
        st.success("KFT parsed.")
        try:
            parse_quality = float(estimate_parse_quality(baseline_kft))
        except Exception:
            parse_quality = None
    except Exception as e:
        st.error(f"KFT parsing failed: {e}")
        baseline_kft = {"Sodium": 138.0, "Potassium": 4.5}

# Manual override UI
st.markdown("**Baseline labs (parsed) — you may edit manually below**")
st.json(baseline_kft or {"Sodium": 138.0, "Potassium": 4.5})
if parse_quality is not None:
    st.markdown(f"**Parse quality:** {parse_quality:.0%} • Threshold: {PARSE_QUALITY_THRESHOLD:.0%}")
    if parse_quality < PARSE_QUALITY_THRESHOLD:
        st.warning("Parse quality below threshold. You can enter exact values manually below.")

with st.expander("Enter/adjust KFT values manually"):
    # Prefill with parsed if available
    def _pref(key: str, default: float) -> float:
        try:
            return float(baseline_kft.get(key, default))
        except Exception:
            return float(default)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        manual_na = st.number_input("Sodium (mEq/L)", min_value=100.0, max_value=200.0,
                                    value=_pref("Sodium", 138.0), step=0.1)
    with col2:
        manual_k = st.number_input("Potassium (mEq/L)", min_value=2.0, max_value=10.0,
                                   value=_pref("Potassium", 4.5), step=0.1)
    with col3:
        manual_cr = st.number_input("Creatinine (mg/dL)", min_value=0.1, max_value=25.0,
                                     value=_pref("Creatinine", 7.2), step=0.1)
    with col4:
        manual_urea = st.number_input("Urea (mg/dL)", min_value=1.0, max_value=300.0,
                                      value=_pref("Urea", 95.0), step=1.0)

    use_manual_kft = st.checkbox("Use manual values for calculations", value=True)

effective_kft = baseline_kft.copy() if isinstance(baseline_kft, dict) else {}
if 'use_manual_kft' in locals() and use_manual_kft:
    effective_kft["Sodium"] = float(manual_na)
    effective_kft["Potassium"] = float(manual_k)
    effective_kft["Creatinine"] = float(manual_cr)
    effective_kft["Urea"] = float(manual_urea)

st.markdown("**KFT values that will be used**")
st.json(effective_kft or {"Sodium": 138.0, "Potassium": 4.5})

# ------------------- 2) Intake logger -------------------
st.subheader("2) Log intake items (one per line)")
st.caption("Examples: `300g watermelon`, `1100ml Apple Juice`, `1 cup soy milk`")
items_text = st.text_area("Items", value="300g watermelon\n1100ml Apple Juice\n1 cup soy milk", height=140)

do_lookup = st.button("Lookup foods & compute nutrients")
breakdown, totals = [], {}

if do_lookup:
    if breakdown_food_items is None:
        st.error("Could not import food lookup. Ensure `food_lookup_new.py` or `food_lookup.py` has `breakdown_food_items`.")
    else:
        items = [ln.strip() for ln in items_text.splitlines() if ln.strip()]
        try:
            breakdown = breakdown_food_items(items)
        except Exception as e:
            st.error(f"Food lookup failed: {e}")
            breakdown = []

        if breakdown:
            df_b = pd.DataFrame(breakdown)
            st.markdown("**Per-item breakdown**")
            st.dataframe(df_b, use_container_width=True)

            def _sum(col): return round(float(df_b[col].fillna(0).sum()), 2) if col in df_b else 0.0
            totals = {
                "Water_ml": _sum("Water_ml"),
                "Sodium_mg": _sum("Sodium_mg"),
                "Potassium_mg": _sum("Potassium_mg"),
                "Phosphorus_mg": _sum("Phosphorus_mg"),
                "Protein_g": _sum("Protein_g"),
            }
            st.markdown("**Intake totals**")
            st.json(totals)

# ------------------- 3) Fluid projection -------------------
st.subheader("3) Fluid dilution & K spike (~1h)")
proj = None
if breakdown and totals:
    if predict_dilution_spike is None:
        st.warning("Fluid parser not found (`fluid_parser.predict_dilution_spike`). Skipping step.")
    else:
        try:
            proj = predict_dilution_spike(
                baseline_kft=effective_kft or {"Sodium": 138.0, "Potassium": 4.5},
                intake_totals=totals,
                weight_kg=float(weight_kg),
                breakdown=breakdown,
                tbw_fraction=float(tbw_fraction),
            )
            st.json(proj)
        except Exception as e:
            st.error(f"Projection failed: {e}")

# ------------------- 4) Save to history CSV -------------------
st.subheader("4) Save this intake to history CSV")
if breakdown:
    if not os.path.isfile(HISTORY_CSV):
        pd.DataFrame(columns=["session_id", "date", "items_json"]).to_csv(HISTORY_CSV, index=False)

    if st.button("Save meal"):
        try:
            session_id = f"{patient_id}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            row = {
                "session_id": session_id,
                "date": session_date.isoformat(),
                "items_json": pd.Series(breakdown).to_json(orient="values"),
            }
            pd.DataFrame([row]).to_csv(HISTORY_CSV, mode="a", index=False, header=False)
            st.success(f"Saved to {HISTORY_CSV}")
        except Exception as e:
            st.error(f"Save failed: {e}")

    with st.expander("Show last 5 saved sessions"):
        try:
            df_hist = pd.read_csv(HISTORY_CSV)
            st.dataframe(df_hist.tail(5), use_container_width=True)
        except Exception as e:
            st.info("No history yet.")

# ------------------- 5) Predict severity / next date -------------------
st.subheader("5) Predict dialysis severity / next date")

# Allow selecting past intakes for this patient to include in prediction
history_df = None
selected_past_n = 0
include_current_intake = False
agg_totals = None
agg_proj = None

try:
    if os.path.isfile(HISTORY_CSV):
        history_df = pd.read_csv(HISTORY_CSV)
except Exception:
    history_df = None

with st.expander("Use past saved intakes for this patient (optional)"):
    if history_df is None or history_df.empty:
        st.caption("No saved intakes found.")
    else:
        # Filter rows belonging to this patient. Our `session_id` is stored as `<patient>-<timestamp>`
        dfp = history_df.copy()
        if 'session_id' in dfp.columns:
            dfp = dfp[dfp['session_id'].astype(str).str.startswith(f"{patient_id}-", na=False)]
        else:
            dfp = dfp.iloc[0:0]

        if dfp.empty:
            st.caption("No past intakes for this patient yet.")
        else:
            # Normalize a date column for display and sorting
            date_col = 'date' if 'date' in dfp.columns else ('data' if 'data' in dfp.columns else None)
            if date_col is not None:
                with pd.option_context('mode.chained_assignment', None):
                    dfp['__date'] = pd.to_datetime(dfp[date_col], errors='coerce')
                dfp = dfp.sort_values(by='__date').reset_index(drop=True)
            else:
                dfp['__date'] = pd.NaT

            max_n = int(min(14, len(dfp)))
            selected_past_n = st.number_input(
                "Include last N saved intakes (days)", min_value=0, max_value=max_n, value=0, step=1
            )
            if breakdown:
                include_current_intake = st.checkbox("Include current unsaved intake as well", value=True)
            else:
                include_current_intake = False

            # Build aggregated totals across the selected sessions + optionally current
            if selected_past_n > 0 or include_current_intake:
                totals_list: list[dict] = []

                # Helper to compute per-session totals from various schemas
                def compute_totals_from_row(row: pd.Series) -> dict:
                    # Preferred: items_json saved by this UI
                    try:
                        if 'items_json' in row and isinstance(row['items_json'], str) and row['items_json']:
                            items = json.loads(row['items_json'])
                            df_items = pd.DataFrame(items)
                            def _sum(col):
                                return round(float(df_items[col].fillna(0).sum()), 2) if col in df_items else 0.0
                            return {
                                "Water_ml": _sum("Water_ml"),
                                "Sodium_mg": _sum("Sodium_mg"),
                                "Potassium_mg": _sum("Potassium_mg"),
                                "Phosphorus_mg": _sum("Phosphorus_mg"),
                                "Protein_g": _sum("Protein_g"),
                            }
                    except Exception:
                        pass

                    # Legacy: items string from simple_diet_logger.py
                    try:
                        if 'items' in row and isinstance(row['items'], str) and row['items']:
                            raw_items = [s.strip() for s in str(row['items']).split('|') if s.strip()]
                            if breakdown_food_items is not None and raw_items:
                                items = breakdown_food_items(raw_items)
                                df_items = pd.DataFrame(items)
                                def _sum(col):
                                    return round(float(df_items[col].fillna(0).sum()), 2) if col in df_items else 0.0
                                return {
                                    "Water_ml": _sum("Water_ml"),
                                    "Sodium_mg": _sum("Sodium_mg"),
                                    "Potassium_mg": _sum("Potassium_mg"),
                                    "Phosphorus_mg": _sum("Phosphorus_mg"),
                                    "Protein_g": _sum("Protein_g"),
                                }
                    except Exception:
                        pass

                    return {"Water_ml": 0.0, "Sodium_mg": 0.0, "Potassium_mg": 0.0, "Phosphorus_mg": 0.0, "Protein_g": 0.0}

                if selected_past_n > 0:
                    recent = dfp.tail(selected_past_n)
                    for _, r in recent.iterrows():
                        totals_list.append(compute_totals_from_row(r))

                if include_current_intake and breakdown:
                    # Compute totals from current breakdown already shown above
                    df_b = pd.DataFrame(breakdown)
                    def _sum(col):
                        return round(float(df_b[col].fillna(0).sum()), 2) if col in df_b else 0.0
                    totals_list.append({
                        "Water_ml": _sum("Water_ml"),
                        "Sodium_mg": _sum("Sodium_mg"),
                        "Potassium_mg": _sum("Potassium_mg"),
                        "Phosphorus_mg": _sum("Phosphorus_mg"),
                        "Protein_g": _sum("Protein_g"),
                    })

                # Aggregate totals
                if totals_list:
                    agg_totals = {
                        "Water_ml": round(sum(t.get("Water_ml", 0.0) for t in totals_list), 2),
                        "Sodium_mg": round(sum(t.get("Sodium_mg", 0.0) for t in totals_list), 2),
                        "Potassium_mg": round(sum(t.get("Potassium_mg", 0.0) for t in totals_list), 2),
                        "Phosphorus_mg": round(sum(t.get("Phosphorus_mg", 0.0) for t in totals_list), 2),
                        "Protein_g": round(sum(t.get("Protein_g", 0.0) for t in totals_list), 2),
                    }
                    st.markdown("**Combined intake totals (selected days)**")
                    st.json(agg_totals)

                    # Project effect on Na/K using the fluid parser
                    if predict_dilution_spike is not None:
                        try:
                            agg_proj = predict_dilution_spike(
                                baseline_kft=effective_kft or {"Sodium": 138.0, "Potassium": 4.5},
                                intake_totals=agg_totals,
                                weight_kg=float(weight_kg),
                                breakdown=None,
                                tbw_fraction=float(tbw_fraction),
                            )
                            st.markdown("**Projected Na/K after combined intake (~1h)**")
                            st.json(agg_proj)
                        except Exception as e:
                            st.warning(f"Multi-day projection failed: {e}")

if st.button("Run prediction"):
    # Use effective KFT, optionally overridden by the aggregated projection
    kft_for_prediction = dict(effective_kft) if isinstance(effective_kft, dict) else {"Sodium": 138.0, "Potassium": 4.5}
    if isinstance(agg_proj, dict):
        if "Sodium_new" in agg_proj:
            kft_for_prediction["Sodium"] = float(agg_proj["Sodium_new"])
        if "Potassium_new" in agg_proj:
            kft_for_prediction["Potassium"] = float(agg_proj["Potassium_new"])
    st.markdown("**KFT values used for prediction**")
    st.json(kft_for_prediction)
    # These are the exact columns your model expects:
    required_features = [
        'Age', 'Gender', 'Weight', 'Diabetes', 'Hypertension', 'Creatinine', 'Urea', 'Potassium', 'Hemoglobin',
        'Kt/V', 'URR', 'Urine Output (ml/day)', 'Dry Weight (kg)',
        'Dialyzer Type_High-flux', 'Dialyzer Type_Low-flux',
        'Disease Severity_Mild', 'Disease Severity_Moderate', 'Disease Severity_Severe'
    ]

    # TODO: Get these from user inputs/UI or set dummy/defaults
    # Here, use sidebar fields for demo; replace as needed!
    input_row = {
        "Age": st.sidebar.number_input("Age", value=55),
        "Gender": st.sidebar.selectbox("Gender (0=Male, 1=Female)", options=[0,1], index=0),
        "Weight": float(weight_kg),
        "Diabetes": st.sidebar.selectbox("Diabetes (0=No, 1=Yes)", options=[0,1], index=0),
        "Hypertension": st.sidebar.selectbox("Hypertension (0=No, 1=Yes)", options=[0,1], index=0),
        "Creatinine": (kft_for_prediction.get("Creatinine") if isinstance(kft_for_prediction, dict) else None) or np.nan,
        "Urea": (kft_for_prediction.get("Urea") if isinstance(kft_for_prediction, dict) else None) or np.nan,
        "Potassium": (kft_for_prediction.get("Potassium") if isinstance(kft_for_prediction, dict) else None) or np.nan,
        "Hemoglobin": st.sidebar.number_input("Hemoglobin", value=10.0),
        "Kt/V": st.sidebar.number_input("Kt/V", value=1.2),
        "URR": st.sidebar.number_input("URR", value=70.0),
        "Urine Output (ml/day)": st.sidebar.number_input("Urine Output (ml/day)", value=500),
        "Dry Weight (kg)": st.sidebar.number_input("Dry Weight (kg)", value=weight_kg),
        "Dialyzer Type_High-flux": st.sidebar.selectbox("Dialyzer Type: High-flux", options=[0,1], index=1),
        "Dialyzer Type_Low-flux": st.sidebar.selectbox("Dialyzer Type: Low-flux", options=[0,1], index=0),
        "Disease Severity_Mild": st.sidebar.selectbox("Disease Severity: Mild", options=[0,1], index=0),
        "Disease Severity_Moderate": st.sidebar.selectbox("Disease Severity: Moderate", options=[0,1], index=0),
        "Disease Severity_Severe": st.sidebar.selectbox("Disease Severity: Severe", options=[0,1], index=1),
    }

    # Always build DataFrame with all required columns
    X = pd.DataFrame([{k: input_row.get(k, 0) for k in required_features}], columns=required_features)

    st.markdown("**Features sent to model:**")
    st.dataframe(X)

    # Allow prediction when manual values are explicitly chosen
    if (
        (parse_quality is not None) and (parse_quality < PARSE_QUALITY_THRESHOLD)
        and (not ('use_manual_kft' in locals() and use_manual_kft))
    ):
        st.warning("Skipping prediction due to low KFT parse quality. Enable 'Use manual values' above to proceed.")
    elif model is None:
        st.warning("No model found at `models/dialysis_model.h5`.")
    else:
        try:
            # Scale if scaler exists
            X_scaled = scaler.transform(X) if scaler is not None else X

            y = model.predict(X_scaled)
            st.success("Prediction complete.")
            st.write("**Model output:**", y)

            try:
                freq_per_week = float(y.flatten()[0])
                days_between = int(round(7.0 / max(freq_per_week, 1e-6)))
                next_date = session_date + dt.timedelta(days=days_between)
                st.info(f"Estimated dialysis frequency: {freq_per_week:.2f}/week → next ~ **{next_date}**")
            except Exception:
                pass
        except Exception as e:
            st.error(f"Model inference failed: {e}")

# ------------------- Disclaimer -------------------
st.divider()
st.caption(
    "Disclaimer: These insights are generated by AI and are for informational purposes only. "
    "They are not a substitute for professional medical advice, diagnosis, or treatment. Always consult your doctor."
)
