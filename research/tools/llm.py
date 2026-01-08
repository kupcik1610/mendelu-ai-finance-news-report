"""
Ollama LLM wrapper for local language model interactions.
"""

import json
import requests
from django.conf import settings


class LLM:
    """Unified interface to local Ollama LLM."""

    def __init__(self, model: str = None):
        self.model = model or getattr(settings, 'OLLAMA_MODEL', 'mistral')
        self.base_url = getattr(settings, 'OLLAMA_HOST', 'http://localhost:11434')

    def generate(self, prompt: str, system: str = "") -> str:
        """Generate text response from LLM."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system,
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"LLM request failed: {e}")
            return ""

    def generate_json(self, prompt: str, system: str = "") -> dict | list:
        """Generate JSON response from LLM."""
        full_prompt = prompt + "\n\nRespond with valid JSON only. No markdown, no explanation."
        response = self.generate(full_prompt, system)

        # Extract JSON from response (handle markdown code blocks)
        text = response.strip()
        if text.startswith("```"):
            # Remove markdown code block
            lines = text.split("\n")
            # Find the content between ``` markers
            start_idx = 1 if lines[0].startswith("```") else 0
            end_idx = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            text = "\n".join(lines[start_idx:end_idx])
            # Remove 'json' language identifier if present
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON within the response
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass

            # Try array
            start = text.find('[')
            end = text.rfind(']')
            if start != -1 and end != -1:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass

            return {}

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False

            # Check if our model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            return self.model in model_names or any(self.model in name for name in model_names)
        except requests.exceptions.RequestException:
            return False
