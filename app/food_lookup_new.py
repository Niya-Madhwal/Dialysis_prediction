# food_lookup_new.py

import os
import re
import math
import pandas as pd
from difflib import get_close_matches
from typing import List, Dict, Tuple, Any, Optional

# ---------------------------------------------------------------------------
# Config & CSV Columns
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "MyFoodData-Nutrition.csv")
PER_100G = True  # True if your CSV is per-100g, False if per-serving

UNIT_TO_G = {
    "g": 1.0, "gram": 1.0, "grams": 1.0,
    "kg": 1000.0,
    "ml": 1.0, "l": 1000.0, "liter": 1000.0, "litre": 1000.0,
    "cup": 240.0, "cups": 240.0,
    "glass": 240.0, "glasses": 240.0,
    "tbsp": 15.0, "tablespoon": 15.0, "tablespoons": 15.0,
    "tsp": 5.0, "teaspoon": 5.0, "teaspoons": 5.0,
    "bottle": 330.0, "bottles": 330.0,
    "katori": 150.0, "bowl": 250.0,
    "piece": 150.0, "pieces": 150.0,   # average apple ~150g
    "egg": 50.0, "eggs": 50.0,
    "slice": 30.0, "slices": 30.0,
    "small": 100.0, "medium": 150.0, "large": 200.0
}

CSV_COLS = {
    "name": ["name", "Name", "Food", "Food Name", "Dish Name", "description", "Description"],
    "water_g": ["Water (g)", "Water", "Water_g"],
    "sodium_mg": ["Sodium (mg)", "Sodium, Na", "Na (mg)"],
    "potassium_mg": ["Potassium, K (mg)", "Potassium (mg)", "K (mg)"],
    "phosphorus_mg": ["Phosphorus, P (mg)", "Phosphorus (mg)", "P (mg)"],
    "protein_g": ["Protein (g)", "Protein"],
    "calcium_mg": ["Calcium (mg)"],
    "iron_mg": ["Iron, Fe (mg)", "Iron (mg)"],
    "calories_kcal": ["Calories", "Energy (kcal)"]
}

# ---------------------------------------------------------------------------
# Load & canonicalize dataset
# ---------------------------------------------------------------------------
def _find_first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for canon, candidates in CSV_COLS.items():
        actual = _find_first_present(df, candidates)
        if actual:
            rename_map[actual] = canon
    df = df.rename(columns=rename_map)
    if "name" not in df.columns:
        raise ValueError(f"Could not find a name column. Found: {list(df.columns)}")

    for k in ["water_g","sodium_mg","potassium_mg","phosphorus_mg",
              "protein_g","calcium_mg","iron_mg","calories_kcal"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce").fillna(0.0)
        else:
            df[k] = 0.0

    df["name_lower"] = df["name"].astype(str).str.lower().str.strip()
    return df

def _load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nutrition file not found at: {path}")
    df = pd.read_csv(path)
    return _canonicalize_columns(df)

# ---------------------------------------------------------------------------
# Parsing & Matching
# ---------------------------------------------------------------------------
_QTY_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>kg|g|ml|l|liter|litre|cup|cups|glass|glasses|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons|bottle|bottles|katori|bowl)\b",
    re.IGNORECASE
)

def parse_quantity_and_food(item: str) -> Tuple[float, str]:
    text = item.strip().lower()
    qty_g_total = 0.0
    for m in _QTY_RE.finditer(text):
        num = float(m.group("num"))
        unit = m.group("unit").lower()
        qty_g_total += num * UNIT_TO_G.get(unit, 0.0)
    # fallback: simple "<num> g"
    if qty_g_total == 0.0:
        m = re.search(r"(\d+(?:\.\d+)?)\s*g\b", text)
        if m:
            qty_g_total = float(m.group(1))
    # strip quantities/units to get the food key
    food_key = _QTY_RE.sub("", text)
    food_key = re.sub(r"\d+(?:\.\d+)?\s*g\b", "", food_key)
    food_key = re.sub(r"\([^)]*\)", "", food_key)  # remove parentheticals
    food_key = re.sub(r"\s+", " ", food_key).strip()
    return qty_g_total, food_key

def lookup_row(df: pd.DataFrame, food_key: str, cutoff: float = 0.85) -> Tuple[bool, str, Optional[pd.Series]]:
    if not food_key:
        return False, food_key, None
    # 1. Exact match
    exact = df[df["name_lower"] == food_key]
    if not exact.empty:
        row = exact.iloc[0]
        return True, row["name"], row
    # 2. Startswith match
    sw = df[df["name_lower"].str.startswith(food_key)]
    if not sw.empty:
        row = sw.iloc[0]
        return True, row["name"], row
    # 3. Contains match
    cont = df[df["name_lower"].str.contains(food_key)]
    if not cont.empty:
        row = cont.iloc[0]
        return True, row["name"], row
    # 4. Fuzzy match (stricter)
    matches = get_close_matches(food_key, df["name_lower"].tolist(), n=1, cutoff=cutoff)
    if matches:
        row = df[df["name_lower"] == matches[0]].iloc[0]
        return True, row["name"], row
    return False, food_key, None


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------
_BEVERAGE_PAT = re.compile(
    r"\b("
    r"water|coconut water|lemon water|lemonade|juice|milk|tea|coffee|broth|soup|soda|shake|lassi|buttermilk"
    r")\b",
    flags=re.IGNORECASE
)
_NON_BEVERAGE_EXCEPTIONS = {"watermelon", "water chestnut", "watercress"}

def _norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())

def _guess_is_liquid(qty_g: float, water_ml: float, name: str) -> bool:
    n = _norm_name(name)
    if n in _NON_BEVERAGE_EXCEPTIONS:
        return False
    if _BEVERAGE_PAT.search(n):
        return True
    return False

def _guess_is_sugary(name: str) -> bool:
    n = _norm_name(name)
    return any(k in n for k in ["juice", "soda", "coconut water", "shake", "lassi", "sweet"])

def _phos_absorption_fraction(name: str) -> float:
    n = _norm_name(name)
    if "oil" in n:
        return 0.0
    if any(k in n for k in ["phosphate", "fortified", "trisodium", "disodium", "dipotassium"]):
        return 0.9
    if any(k in n for k in ["soy milk", "almond milk", "oat milk", "rice milk"]):
        return 0.7
    if any(k in n for k in ["milk", "yogurt", "curd", "paneer", "cheese", "dairy", "buttermilk"]):
        return 0.6
    return 0.35

def _drink_category(name: str) -> str:
    n = _norm_name(name)
    # Specific → general order
    if any(k in n for k in ["ors", "electral", "oral rehydration", "rehydration salts"]):
        return "ors"
    if any(k in n for k in ["gatorade", "powerade", "electrolyte drink", "isotonic drink", "sports drink"]):
        return "sports"
    if any(k in n for k in ["broth", "soup", "rasam", "sambar", "dal soup"]):
        return "broth"
    if any(k in n for k in ["shake", "smoothie", "falooda", "thick"]):
        return "thick_sugary"
    if any(k in n for k in ["lassi", "buttermilk", "chaas", "chhaas"]):
        return "buttermilk"
    if any(k in n for k in ["coconut water"]):
        return "coconut"
    if any(k in n for k in ["juice", "lemonade", "nimbu pani", "jaljeera", "shikanji"]):
        return "juice"
    if any(k in n for k in ["milk", "soy milk", "almond milk", "oat milk", "rice milk"]):
        return "milk"
    if any(k in n for k in ["tea", "coffee"]):
        return "tea_coffee"
    if any(k in n for k in ["soda", "cola", "soft drink", "aerated"]):
        return "soda"
    if "water" in n:
        return "water"
    return "other"

def _na_bounds_and_thalf_for_category(cat: str, is_sugary: bool) -> tuple[tuple[float, float], float]:
    # (na_min, na_max) in mmol/L, absorption half-life (hours)
    if cat == "broth":
        return (5.0, 70.0), 0.5
    if cat == "ors":
        return (30.0, 90.0), 0.5
    if cat == "sports":
        return (10.0, 40.0), 0.5
    if cat == "milk":
        return (2.0, 25.0), 0.67
    if cat == "buttermilk":
        return (2.0, 25.0), 0.5
    if cat == "coconut":
        return (2.0, 20.0), 0.67
    if cat == "juice":
        return (1.0, 15.0), 0.67
    if cat == "thick_sugary":
        return (1.0, 15.0), 1.2
    if cat == "tea_coffee":
        return (0.5, 5.0), 0.25
    if cat == "soda":
        return (1.0, 10.0), 0.5
    if cat == "water":
        return (0.0, 3.0), 0.25
    return (1.0, 15.0), (0.67 if is_sugary else 0.25)

# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------
def breakdown_food_items(items: List[str], data_path: str = DATA_PATH) -> List[Dict[str, Any]]:
    df = _load_dataset(data_path)
    out: List[Dict[str, Any]] = []

    for raw in items:
        qty_g, key = parse_quantity_and_food(raw)
        matched, canon_name, row = lookup_row(df, key)

        if not matched or row is None:
            out.append({
                "food": key or raw,
                "matched": False,
                "quantity_g": round(qty_g, 2),
                "Water_ml": 0.0, "Sodium_mg": 0.0, "Potassium_mg": 0.0,
                "Phosphorus_mg": 0.0, "Protein_g": 0.0,
                "is_liquid": False,
                "is_sugary": False,
                "p_absorption_fraction": 0.35,
                "na_conc_mmol_per_L": 0.0,
                "absorption_t_half_h": 1.2,
            })
            continue

        # Scale factor
        if PER_100G:
            factor = (qty_g / 100.0) if qty_g > 0 else 0.0
        else:
            factor = (qty_g / 100.0)

        water_ml      = float(row["water_g"])      * factor
        sodium_mg     = float(row["sodium_mg"])    * factor
        potassium_mg  = float(row["potassium_mg"]) * factor
        phosphorus_mg = float(row["phosphorus_mg"])* factor
        protein_g     = float(row["protein_g"])    * factor

        is_liquid = _guess_is_liquid(qty_g, water_ml, str(canon_name))
        is_sugary = _guess_is_sugary(str(canon_name))
        p_abs_frac = _phos_absorption_fraction(str(canon_name))

        # --- Precompute drink Na concentration + absorption half-life ---
        if is_liquid and water_ml > 0:
            cat = _drink_category(str(canon_name))
            (na_lo, na_hi), cat_t_half = _na_bounds_and_thalf_for_category(cat, is_sugary)
            na_mEq = sodium_mg / 23.0
            na_conc = na_mEq / (water_ml / 1000.0) if water_ml > 0 else 0.0
            na_conc_mmol_per_L = max(na_lo, min(na_hi, na_conc))
            absorption_t_half_h = cat_t_half
        else:
            na_conc_mmol_per_L = 0.0
            absorption_t_half_h = 1.2

        out.append({
            "food": canon_name,
            "matched": True,
            "quantity_g": round(qty_g, 2),
            "Water_ml": round(water_ml, 2),
            "Sodium_mg": round(sodium_mg, 2),
            "Potassium_mg": round(potassium_mg, 2),
            "Phosphorus_mg": round(phosphorus_mg, 2),
            "Protein_g": round(protein_g, 2),
            "is_liquid": is_liquid,
            "is_sugary": is_sugary,
            "p_absorption_fraction": p_abs_frac,
            "na_conc_mmol_per_L": round(na_conc_mmol_per_L, 2),
            "absorption_t_half_h": round(absorption_t_half_h, 2),
        })

    return out

# ----------------
# Quick test/demo
# ----------------
if __name__ == "__main__":
    items = ["200g apples"
,"100ml Almond Milk"
,"100g Curd"]
    results = breakdown_food_items(items)
    for r in results:
        print(r)
