import os
import csv
import uuid
from datetime import datetime, date
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List


DATA_DIR= os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


HISTORY_CSV = os.path.join(DATA_DIR, "diet_history.csv")

if not os.path.isfile(HISTORY_CSV):
    with open(HISTORY_CSV, mode='w', newline="", encoding="utf-8") as f:
        writer= csv.writer(f)
        writer.writerow(["session_id","data", "items"])

class DietLog(BaseModel):
    items : List[str]

app = FastAPI(title="Simple Diet Logger")

@app.post("/log-diet")
def log_diet(diet: DietLog):
    session_id = uuid.uuid4().hex
    log_date = date.today().isoformat()
    items_str = "|".join(diet.items)

    print(f"[DEBUG] Received items: {diet.items}")
    print(f"[DEBUG] Writing to {HISTORY_CSV}: {session_id}, {log_date}, {items_str}")

    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([session_id, log_date, items_str])

    return {"session_id": session_id, "date": log_date, "items_logged": diet.items}
    