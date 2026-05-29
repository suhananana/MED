"""
RAG Router: Classifies the user query and delegates to the appropriate RAG tool.
Uses keyword + semantic scoring — no LLM needed for routing (fast & free).
"""

import re
from typing import Dict, Any


class RAGRouter:
    """Routes medical queries to the most relevant RAG tool."""

    ROUTING_RULES = {
        "emergency": [
            r"\b(emergency|urgent|911|ambulance|heart attack|stroke|choking|bleed|unconscious|"
            r"overdose|poison|faint|collapse|severe|critical|immediately|first.?aid|cpr)\b"
        ],
        "drug": [
            r"\b(drug|medication|medicine|pill|tablet|capsule|dose|dosage|prescription|"
            r"paracetamol|ibuprofen|aspirin|amoxicillin|metformin|lisinopril|omeprazole|"
            r"antibiotic|antidepressant|painkiller|side.?effect|interaction|overdose|"
            r"mg|mcg|ml|pharmacy|generic|brand)\b"
        ],
        "symptom": [
            r"\b(symptom|feel|feeling|pain|ache|hurt|nausea|vomit|fever|cough|rash|"
            r"itching|swelling|headache|dizziness|fatigue|tired|shortness.of.breath|"
            r"chest.pain|abdominal|diagnosis|diagnose|condition|disease|disorder|"
            r"what.do.i.have|i.have|i.am.having|experiencing)\b"
        ],
        "anatomy": [
            r"\b(anatomy|organ|body|system|heart|lung|liver|kidney|brain|bone|muscle|"
            r"nerve|vein|artery|cell|tissue|function|how.does.the|what.is.the.*made.of|"
            r"structure|physiology|biology|endocrine|immune|lymph|spine|joint)\b"
        ],
    }

    def __init__(self, tools: dict):
        self.tools = tools

    def _score_query(self, query: str) -> str:
        """Return the tool key that best matches the query."""
        q = query.lower()
        scores = {tool: 0 for tool in self.ROUTING_RULES}

        for tool, patterns in self.ROUTING_RULES.items():
            for pattern in patterns:
                matches = re.findall(pattern, q)
                scores[tool] += len(matches)

        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "general"

    def query(self, user_query: str, temperature: float = 0.3) -> Dict[str, Any]:
        """Route the query and return a unified response dict."""
        tool_key = self._score_query(user_query)
        tool = self.tools.get(tool_key, self.tools["general"])

        try:
            result = tool.run(user_query, temperature=temperature)
            result["tool"] = tool_key
            return result
        except Exception as e:
            # Fallback to general
            result = self.tools["general"].run(user_query, temperature=temperature)
            result["tool"] = "general"
            result["error"] = str(e)
            return result
