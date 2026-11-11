import openai

from src.config import Config
from src.observability.metrics import LLM_REQUESTS, LLM_TOKENS


class TigerGatewayClient:
    def __init__(self):
        self.url = Config.AI_GATEWAY_URL
        self.api_key = Config.AI_GATEWAY_API_KEY
        self.model = Config.AI_GATEWAY_MODEL

    def call_llm(self, messages, temperature=0):
        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.url,
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
        except Exception:
            LLM_REQUESTS.labels(self.model or "unknown", "error").inc()
            raise

        LLM_REQUESTS.labels(self.model or "unknown", "success").inc()
        usage = getattr(response, "usage", None)
        if usage is not None:
            for token_type in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ):
                value = getattr(usage, token_type, 0) or 0
                LLM_TOKENS.labels(self.model or "unknown", token_type).inc(
                    value
                )

        try:
            return response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Unexpected response format from AI gateway"
            ) from exc
