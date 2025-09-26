# Thin wrapper for Gemini; no logic yet to keep it simple
class GeminiClient:
    def __init__(self, api_key: str | None):
        self.api_key = api_key
