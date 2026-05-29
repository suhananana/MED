import wikipedia

def medical_wikipedia_search(query: str):
    try:
        return wikipedia.summary(query, sentences=5)
    except Exception as e:
        return str(e)
