def suggest_dialysis(kft):
    if kft.get('Creatinine', 0) > 10 or kft.get('Potassium', 0) > 6:
        return "Dialysis recommended immediately"
    elif kft.get('Creatinine', 0) > 7:
        return "Dialysis may be needed in 2 days"
    else:
        return "No immediate dialysis needed"
