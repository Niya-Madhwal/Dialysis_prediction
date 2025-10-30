# app_ui.py

import os
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

# Master CSV (legacy/aggregate)
HISTORY_CSV = os.path.join(DATA_DIR, "diet_history.csv")

# New hierarchical history storage: data/history/<PATIENT_ID>/<YYYY-MM-DD>.csv
HISTORY_DIR = os.path.join(DATA_DIR, "history")
os.makedirs(HISTORY_DIR, exist_ok=True)

def _ensure_patient_history_paths(patient_id: str, session_date: dt.date) -> str:
    """Ensure directories exist and return the CSV path for patient's date file."""
    safe_patient_id = str(patient_id).strip().replace("/", "_").replace("\\", "_")
    patient_dir = os.path.join(HISTORY_DIR, safe_patient_id)
    os.makedirs(patient_dir, exist_ok=True)
    date_csv_path = os.path.join(patient_dir, f"{session_date.isoformat()}.csv")
    # Initialize file with header if missing
    if not os.path.isfile(date_csv_path):
        import pandas as _pd
        _pd.DataFrame(columns=["session_id", "date_time", "items_json"]).to_csv(date_csv_path, index=False)
    return date_csv_path

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

            # Persist in session_state for later saving across reruns
            st.session_state["breakdown"] = breakdown
            st.session_state["totals"] = totals

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
st.subheader("4) Save this intake to history")
# Use persisted breakdown if available (so save works after rerun)
persisted_breakdown = st.session_state.get("breakdown", [])
if persisted_breakdown:
    # Ensure master CSV exists (for backward compatibility)
    if not os.path.isfile(HISTORY_CSV):
        pd.DataFrame(columns=["session_id", "date", "items_json"]).to_csv(HISTORY_CSV, index=False)

    if st.button("Save meal"):
        try:
            # Build identifiers
            now_ts = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
            session_id = f"{patient_id}-{now_ts}"

            # Prepare rows for master and patient-specific files
            items_json = pd.Series(persisted_breakdown).to_json(orient="values")
            master_row = {
                "session_id": session_id,
                "date": session_date.isoformat(),
                "items_json": items_json,
            }

            # Append to master CSV (aggregate)
            pd.DataFrame([master_row]).to_csv(HISTORY_CSV, mode="a", index=False, header=False)

            # Append to patient-specific date CSV under data/history/<PATIENT_ID>/<YYYY-MM-DD>.csv
            patient_csv_path = _ensure_patient_history_paths(patient_id, session_date)
            patient_row = {
                "session_id": session_id,
                "date_time": dt.datetime.now().isoformat(timespec="seconds"),
                "items_json": items_json,
            }
            pd.DataFrame([patient_row]).to_csv(patient_csv_path, mode="a", index=False, header=False)

            st.success(f"Saved. Patient history: {patient_csv_path}")
        except Exception as e:
            st.error(f"Save failed: {e}")

# Always show patient history viewer
with st.expander("Show last saved sessions for this patient"):
    try:
        # Collect the patient's records across recent dates
        safe_patient_id = str(patient_id).strip().replace("/", "_").replace("\\", "_")
        patient_dir = os.path.join(HISTORY_DIR, safe_patient_id)
        if not os.path.isdir(patient_dir):
            st.info("No history yet for this patient.")
        else:
            # Read up to last 7 date files, newest first
            date_files = sorted(
                [p for p in os.listdir(patient_dir) if p.endswith('.csv')],
                reverse=True
            )[:7]
            frames = []
            for fname in date_files:
                fpath = os.path.join(patient_dir, fname)
                try:
                    dfp = pd.read_csv(fpath)
                    dfp.insert(1, "date", fname.replace('.csv', ''))
                    frames.append(dfp)
                except Exception:
                    continue
            if frames:
                df_patient = pd.concat(frames, ignore_index=True)
                # Show last 10 sessions
                st.dataframe(df_patient.tail(10), use_container_width=True)
            else:
                st.info("No history yet for this patient.")
    except Exception:
        st.info("No history yet for this patient.")

# ------------------- 5) Predict severity / next date -------------------
st.subheader("5) Predict dialysis severity / next date")
if st.button("Run prediction"):
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
        "Creatinine": (effective_kft.get("Creatinine") if isinstance(effective_kft, dict) else None) or np.nan,
        "Urea": (effective_kft.get("Urea") if isinstance(effective_kft, dict) else None) or np.nan,
        "Potassium": (effective_kft.get("Potassium") if isinstance(effective_kft, dict) else None) or np.nan,
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
