"""Ollama HTTP adapter; endpoint and model are configured through environment."""
import json
import os
from urllib.request import Request, urlopen


class LLM:
    def json(self, system, payload):
        body = {
            "model": os.getenv("RAG_LLM_MODEL", "llama3.2"),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload)},
            ],
        }
        request = Request(os.getenv("RAG_LLM_URL", "http://localhost:11434/api/chat"),
                          data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read())
        parsed = json.loads(result["message"]["content"])
        if not isinstance(parsed, dict):
            raise ValueError("LLM must return a JSON object")
        return parsed
