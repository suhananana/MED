import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

def scrape_medical_web(query: str):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(f"medical {query}", max_results=3))

        if not results:
            return "No results found."

        url = results[0]["href"]

        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers)

        soup = BeautifulSoup(response.text, "html.parser")

        paragraphs = soup.find_all("p")

        text = " ".join([p.get_text() for p in paragraphs[:10]])

        return text[:3000]

    except Exception as e:
        return str(e)
