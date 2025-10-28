# app_ui.py

import os
import json
import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Any



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
HISTORY_DIR = os.path.join(DATA_DIR, "history")
os.makedirs(HISTORY_DIR, exist_ok=True)

# ---------- History helpers (per-patient, per-date) ----------
def _sanitize_patient_id(pid: str) -> str:
    # Avoid path traversal; keep simple safe characters
    pid = (pid or "").strip()
    return "".join(c for c in pid if c.isalnum() or c in ("-", "_", ".")).strip(".-_") or "UNKNOWN"

def _patient_dir(pid: str) -> str:
    d = os.path.join(HISTORY_DIR, _sanitize_patient_id(pid))
    os.makedirs(d, exist_ok=True)
    return d

def save_patient_day_log(patient_id: str, session_date: dt.date, session_id: str, breakdown: List[Dict[str, Any]]) -> None:
    """Append a session breakdown under app/data/history/{patient_id}/{YYYY-MM-DD}.json"""
    path = os.path.join(_patient_dir(patient_id), f"{session_date.isoformat()}.json")
    day_payload: List[Dict[str, Any]] = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    day_payload = data
        except Exception:
            day_payload = []
    day_payload.append({
        "session_id": session_id,
        "timestamp": dt.datetime.now().isoformat(),
        "items": breakdown,
    })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(day_payload, f, ensure_ascii=False, indent=2)

def list_patient_dates(patient_id: str) -> List[str]:
    d = _patient_dir(patient_id)
    try:
        files = [fn for fn in os.listdir(d) if fn.endswith(".json")]
    except FileNotFoundError:
        files = []
    dates = []
    for fn in files:
        name = fn[:-5]
        try:
            # validate format
            _ = dt.date.fromisoformat(name)
            dates.append(name)
        except Exception:
            continue
    dates.sort(reverse=True)
    return dates

def load_breakdown_for_dates(patient_id: str, dates_iso: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    base = _patient_dir(patient_id)
    for d_iso in dates_iso:
        path = os.path.join(base, f"{d_iso}.json")
        day_items: List[Dict[str, Any]] = []
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    sessions = json.load(f)
                    if isinstance(sessions, list):
                        for s in sessions:
                            if isinstance(s, dict) and isinstance(s.get("items"), list):
                                day_items.extend(s["items"])
            except Exception:
                pass
        out[d_iso] = day_items
    return out

def _fallback_list_dates_from_csv(patient_id: str) -> List[str]:
    if not os.path.exists(HISTORY_CSV):
        return []
    try:
        df = pd.read_csv(HISTORY_CSV)
        if df.empty:
            return []
        # Filter by session_id prefix if present
        if "session_id" in df.columns:
            mask = df["session_id"].astype(str).str.startswith(f"{patient_id}-")
            df = df[mask]
        if "date" in df.columns:
            dates = sorted(df["date"].dropna().astype(str).unique().tolist(), reverse=True)
        elif "data" in df.columns:  # legacy column name
            dates = sorted(df["data"].dropna().astype(str).unique().tolist(), reverse=True)
        else:
            dates = []
        return dates
    except Exception:
        return []

def _fallback_load_breakdown_for_dates(patient_id: str, dates_iso: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not os.path.exists(HISTORY_CSV):
        return out
    try:
        df = pd.read_csv(HISTORY_CSV)
        if df.empty:
            return out
        if "session_id" in df.columns:
            df = df[df["session_id"].astype(str).str.startswith(f"{patient_id}-")]
        date_col = "date" if "date" in df.columns else ("data" if "data" in df.columns else None)
        if date_col is None:
            return out
        for d_iso in dates_iso:
            sub = df[df[date_col].astype(str) == d_iso]
            items: List[Dict[str, Any]] = []
            for _, row in sub.iterrows():
                v = row.get("items_json")
                if isinstance(v, str) and v.strip():
                    try:
                        parsed = json.loads(v)
                        if isinstance(parsed, list):
                            # items are dicts
                            items.extend(parsed)
                    except Exception:
                        pass
            out[d_iso] = items
    except Exception:
        return out
    return out

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

baseline_kft = {}
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

st.markdown("**Baseline labs**")
st.json(baseline_kft or {"Sodium": 138.0, "Potassium": 4.5})
if parse_quality is not None:
    st.markdown(f"**Parse quality:** {parse_quality:.0%} • Threshold: {PARSE_QUALITY_THRESHOLD:.0%}")
    if parse_quality < PARSE_QUALITY_THRESHOLD:
        st.warning("Parse quality below threshold. Predictions will be skipped.")

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
                baseline_kft=baseline_kft or {"Sodium": 138.0, "Potassium": 4.5},
                intake_totals=totals,
                weight_kg=float(weight_kg),
                breakdown=breakdown,
                tbw_fraction=float(tbw_fraction),
            )
            st.json(proj)
        except Exception as e:
            st.error(f"Projection failed: {e}")

# ------------------- 3b) Projection from saved history (last N days) -------------------
st.subheader("3b) Project using saved history for this patient")
available_dates = list_patient_dates(patient_id)
if not available_dates:
    # Try CSV fallback for legacy entries
    available_dates = _fallback_list_dates_from_csv(patient_id)

if available_dates:
    default_n = 5
    default_sel = available_dates[:default_n]
    selected_dates = st.multiselect(
        "Select dates (latest first)", options=available_dates, default=default_sel,
        help="Last 5 dates are preselected by default"
    )

    if st.button("Compute projection from selected saved dates"):
        try:
            data_by_date = load_breakdown_for_dates(patient_id, selected_dates)
            if not any(data_by_date.get(d) for d in selected_dates):
                # Legacy fallback
                data_by_date = _fallback_load_breakdown_for_dates(patient_id, selected_dates)

            results_rows = []
            combined_items = []
            for d_iso in selected_dates:
                items = data_by_date.get(d_iso, [])
                if not items:
                    continue
                df_b = pd.DataFrame(items)
                def _sum(col):
                    return round(float(df_b[col].fillna(0).sum()), 2) if col in df_b else 0.0
                totals_d = {
                    "Water_ml": _sum("Water_ml"),
                    "Sodium_mg": _sum("Sodium_mg"),
                    "Potassium_mg": _sum("Potassium_mg"),
                    "Phosphorus_mg": _sum("Phosphorus_mg"),
                    "Protein_g": _sum("Protein_g"),
                }
                # Per-date projection
                try:
                    proj_d = predict_dilution_spike(
                        baseline_kft=baseline_kft or {"Sodium": 138.0, "Potassium": 4.5},
                        intake_totals=totals_d,
                        weight_kg=float(weight_kg),
                        breakdown=items,
                        tbw_fraction=float(tbw_fraction),
                    ) if predict_dilution_spike is not None else None
                except Exception:
                    proj_d = None

                row = {"date": d_iso, **totals_d}
                if proj_d:
                    row.update({
                        "Sodium_new": proj_d.get("Sodium_new"),
                        "ΔSodium": proj_d.get("ΔSodium"),
                        "Potassium_new": proj_d.get("Potassium_new"),
                        "ΔPotassium": proj_d.get("ΔPotassium"),
                    })
                results_rows.append(row)
                combined_items.extend(items)

            if results_rows:
                st.markdown("**Per-date projections**")
                st.dataframe(pd.DataFrame(results_rows), use_container_width=True)

            # Combined across all selected dates
            if combined_items:
                df_c = pd.DataFrame(combined_items)
                def _sumc(col):
                    return round(float(df_c[col].fillna(0).sum()), 2) if col in df_c else 0.0
                totals_c = {
                    "Water_ml": _sumc("Water_ml"),
                    "Sodium_mg": _sumc("Sodium_mg"),
                    "Potassium_mg": _sumc("Potassium_mg"),
                    "Phosphorus_mg": _sumc("Phosphorus_mg"),
                    "Protein_g": _sumc("Protein_g"),
                }
                try:
                    proj_c = predict_dilution_spike(
                        baseline_kft=baseline_kft or {"Sodium": 138.0, "Potassium": 4.5},
                        intake_totals=totals_c,
                        weight_kg=float(weight_kg),
                        breakdown=combined_items,
                        tbw_fraction=float(tbw_fraction),
                    ) if predict_dilution_spike is not None else None
                except Exception as e:
                    proj_c = None
                st.markdown("**Combined across selected dates**")
                st.json({"totals": totals_c, "projection": proj_c or {}})
        except Exception as e:
            st.error(f"Failed to compute from history: {e}")
else:
    st.info("No saved history for this patient yet.")

# ------------------- 4) Save to history CSV -------------------
st.subheader("4) Save this intake to history CSV + per-patient/day JSON")
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
            # Also save to structured per-patient/day file
            try:
                save_patient_day_log(patient_id, session_date, session_id, breakdown)
            except Exception as _e:
                st.warning(f"Saved CSV, but JSON history failed: {_e}")
            st.success(f"Saved to {HISTORY_CSV} and per-patient history")
        except Exception as e:
            st.error(f"Save failed: {e}")

    with st.expander("Show last 5 saved sessions/dates"):
        try:
            # Show per-patient dates first
            dates = list_patient_dates(patient_id)
            if not dates:
                dates = _fallback_list_dates_from_csv(patient_id)
            st.write("Patient dates:", ", ".join(dates[:5]) or "(none)")
            # Then show raw CSV last rows for legacy visibility
            if os.path.exists(HISTORY_CSV):
                df_hist = pd.read_csv(HISTORY_CSV)
                st.dataframe(df_hist.tail(5), use_container_width=True)
        except Exception as e:
            st.info("No history yet.")

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
        "Creatinine": baseline_kft.get("Creatinine", np.nan),
        "Urea": baseline_kft.get("Urea", np.nan),
        "Potassium": baseline_kft.get("Potassium", np.nan),
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

    if (parse_quality is not None) and (parse_quality < PARSE_QUALITY_THRESHOLD):
        st.warning("Skipping prediction due to low KFT parse quality.")
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
