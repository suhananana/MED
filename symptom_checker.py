SYMPTOM_DB = {
    "fever": "Possible infection or flu.",
    "headache": "Possible migraine or stress.",
    "cough": "Possible respiratory infection.",
    "chest pain": "Possible heart issue. Seek immediate care."
}

def check_symptoms(symptom: str):
    symptom = symptom.lower()

    for key in SYMPTOM_DB:
        if key in symptom:
            return SYMPTOM_DB[key]

    return "No symptom match found."
