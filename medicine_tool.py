MEDICINE_DB = {
    "paracetamol": "Used for fever and pain relief.",
    "ibuprofen": "NSAID pain reliever.",
    "cetirizine": "Used for allergies."
}

def medicine_info(medicine: str):
    medicine = medicine.lower()

    for med in MEDICINE_DB:
        if med in medicine:
            return MEDICINE_DB[med]

    return "Medicine not found."
