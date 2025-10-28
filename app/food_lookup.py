# food_lookup_new.py

import os
import re
import pandas as pd
from difflib import get_close_matches
from typing import List, Dict, Tuple, Any

# Load the dataset
DATA_PATH = os.path.join("data", "MyFoodData-Nutrition.csv")  
df = pd.read_csv(DATA_PATH)
df['name_lower'] = df['name'].str.lower().str.strip()

# Relevant columns to extract
FEATURES = [
    "Sodium (mg)",
    "Potassium, K (mg)",
    "Phosphorus, P (mg)",
    "Protein (g)",
    "Water (g)"
]

# Clean column names to match our keys
df.rename(columns={
    "Potassium, K (mg)": "Potassium_mg",
    "Sodium (mg)": "Sodium_mg",
    "Phosphorus, P (mg)": "Phosphorus_mg",
    "Protein (g)": "Protein_g",
    "Water (g)": "Water_g"
}, inplace=True)

# Fill missing values with 0
df.fillna(0, inplace=True)

def parse_quantity_and_food(item: str) -> Tuple[float, str]:
    text = item.lower()
    
    # Match grams first
    m = re.search(r"(\d+(?:\.\d+)?)\s*g", text)
    if m:
        qty = float(m.group(1))
    else:
        # Match ml as ≈ grams (for fluids)
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*ml", text)
        if m2:
            qty = float(m2.group(1))  # assume 1ml = 1g
        else:
            # Fallback: 1 cup = 240g
            m3 = re.search(r"(\d+(?:\.\d+)?)\s*cup", text)
            qty = float(m3.group(1)) * 240 if m3 else 0.0

    # Clean text
    no_paren = re.sub(r"\([^)]*\)", '', text)
    food_key = re.sub(r"\d+(?:\.\d+)?\s*(g|ml|cup)s?", '', no_paren).strip()
    return qty, food_key


def lookup_food_nutrients(food_name: str) -> Dict[str, float]:
    row = df[df['name_lower'] == food_name]
    if row.empty:
        match = get_close_matches(food_name, df['name_lower'], n=1, cutoff=0.7)
        if not match:
            print(f"⚠️ No match found for '{food_name}'")
            return {k: 0.0 for k in ["Sodium_mg", "Potassium_mg", "Phosphorus_mg", "Protein_g", "Water_g"]}
        row = df[df['name_lower'] == match[0]]
        print(f"[INFO] Matched '{food_name}' → '{match[0]}'")

    row = row.iloc[0]
    return {
        "Sodium_mg": row["Sodium_mg"],
        "Potassium_mg": row["Potassium_mg"],
        "Phosphorus_mg": row["Phosphorus_mg"],
        "Protein_g": row["Protein_g"],
        "Water_g": row["Water_g"]
    }

def breakdown_food_items(items: List[str]) -> List[Dict[str, Any]]:
    breakdown = []

    for item in items:
        qty, key = parse_quantity_and_food(item)
        nutrients = lookup_food_nutrients(key)
        factor = qty / 100.0

        breakdown.append({
            "food": key,
            "quantity_g": qty,
            "Sodium_mg": round(nutrients["Sodium_mg"] * factor, 2),
            "Potassium_mg": round(nutrients["Potassium_mg"] * factor, 2),
            "Phosphorus_mg": round(nutrients["Phosphorus_mg"] * factor, 2),
            "Protein_g": round(nutrients["Protein_g"] * factor, 2),
            "Water_ml": round(nutrients["Water_g"] * factor, 2)  # Water_g ≈ fluid ml
        })

    return breakdown

# Example usage
if __name__ == "__main__":
    items = ["300g watermelon", "1100ml Apple Juice"]
    results = breakdown_food_items(items)
    for r in results:
        print(r)
