import os
import re
import pandas as pd
from difflib import get_close_matches
from typing import List, Dict, Any

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
HISTORY_CSV = os.path.join(BASE_DIR, "data", "diet_history.csv")
NUTRITION_CSV = os.path.join(BASE_DIR, "data", "Indian_Food_Nutrition_Processed.csv")
FEATURES_CSV = os.path.join(BASE_DIR, "data", "diet_features.csv")

# Load CSVs
history_df = pd.read_csv(HISTORY_CSV)
nutrition_df = pd.read_csv(NUTRITION_CSV)
nutrition_df['desc_lower'] = nutrition_df['Dish Name'].str.lower()

# Default nutrient values
DEFAULTS = {
    "Sodium, Na": 0.0,
    "Potassium, K": 0.0,
    "Protein": 0.0,
    "Phosphorus, P": 0.0
}

def parse_quantity_and_food(item: str) -> (float, str):
    text = item.lower()
    m = re.search(r'(\d+(?:\.\d+)?)\s*g', text)
    qty = float(m.group(1)) if m else 0.0
    key = re.sub(r'\([^)]*\)', '', text)
    key = re.sub(r'(\d+(?:\.\d+)?\s*g)', '', key).strip()
    return qty, key

def lookup_nutrition(key: str) -> Dict[str, float]:
    match = nutrition_df[nutrition_df['desc_lower'] == key]
    if match.empty:
        best = get_close_matches(key, nutrition_df['desc_lower'], n=1, cutoff=0.9)
        if not best:
            return DEFAULTS.copy()
        match = nutrition_df[nutrition_df['desc_lower'] == best[0]]
    row = match.iloc[0]
    return {
        "Sodium, Na": row.get("Sodium, Na", DEFAULTS["Sodium, Na"]),
        "Potassium, K": row.get("Potassium, K", DEFAULTS["Potassium, K"]),
        "Protein": row.get("Protein", DEFAULTS["Protein"]),
        "Phosphorus, P": row.get("Phosphorus, P", DEFAULTS["Phosphorus, P"])
    }

def process_session(session_id: str) -> List[Dict[str, Any]]:
    session_rows = history_df[history_df['session_id'] == session_id]
    if session_rows.empty:
        raise ValueError(f"No session found with session_id: {session_id}")
    
    items_combined = []
    for _, row in session_rows.iterrows():
        items = str(row['items']).split("|")
        items_combined.extend(items)

    results = []
    for item in items_combined:
        qty_g, food_key = parse_quantity_and_food(item)
        nutrients = lookup_nutrition(food_key)
        factor = qty_g / 100.0

        result = {
            "session_id": session_id,
            "date": row['data'],
            "food": food_key,
            "quantity_g": qty_g,
            "Sodium_mg": round(nutrients["Sodium, Na"] * factor, 2),
            "Potassium_mg": round(nutrients["Potassium, K"] * factor, 2),
            "Protein_g": round(nutrients["Protein"] * factor, 2),
            "Phosphorus_mg": round(nutrients["Phosphorus, P"] * factor, 2)
        }
        results.append(result)

    return results

# Run and write to features CSV
if not history_df.empty:
    latest_session_id = history_df['session_id'].iloc[-1]
    features = process_session(latest_session_id)
    features_df = pd.DataFrame(features)
    features_df.to_csv(FEATURES_CSV, index=False)
    print(f"Saved {len(features_df)} feature rows to {FEATURES_CSV}")
else:
    print("No sessions found in diet_history.csv.")
