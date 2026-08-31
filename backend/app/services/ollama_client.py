
import requests


class OllamaClient:

    BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str = "phi3:latest",
        timeout: int = 300
    ):
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        prompt: str
    ) -> str:

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 2048
            }
        }

        try:

            response = requests.post(
                f"{self.BASE_URL}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            if not response.ok:
                print()
                print("=" * 80)
                print("OLLAMA ERROR")
                print("=" * 80)
                print("Status:", response.status_code)
                print("Response:", response.text)
                print("=" * 80)

            response.raise_for_status()

            data = response.json()

            if "response" not in data:
                raise RuntimeError(
                    f"Ollama response did not contain "
                    f"'response': {data}"
                )

            return data["response"].strip()

        except requests.exceptions.Timeout:
            raise RuntimeError(
                "Ollama request timed out. "
                "The model may need more time or "
                "the prompt may be too large."
            )

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Could not connect to Ollama. "
                "Make sure Ollama is running."
            )