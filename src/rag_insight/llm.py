"""Ollama HTTP adapter; endpoint and model are configured through environment."""
import json
import os
from urllib.request import Request, urlopen


class LLM:
    def __init__(self, model: str = "llama3.2:latest", url: str = "http://localhost:11434/api/chat"):
        self.model = model
        self.url = url

    def json(self, system, payload, max_tokens=96, response_format=None):
        body = {
            "model": os.getenv("RAG_LLM_MODEL", self.model),
            "stream": False,
            "format": response_format or "json",
            # Keep responses bounded and the model resident between the grader
            # and generator calls; both materially reduce local Ollama latency.
            "keep_alive": os.getenv("RAG_OLLAMA_KEEP_ALIVE", "10m"),
            "options": {"temperature": 0, "num_predict": max_tokens},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload)},
            ],
        }
        request = Request(os.getenv("RAG_LLM_URL", self.url),
                          data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read())
        parsed = json.loads(result["message"]["content"])
        if not isinstance(parsed, dict):
            raise ValueError("LLM must return a JSON object")
        return parsed
