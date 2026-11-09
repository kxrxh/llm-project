"""Вспомогательный запускатель для выполнения задания через OpenAI API.

Запуск: python -m assistant.graphrag.openai_eval index.
Перед запуском экспортируйте переменные из .env. Проверен на gpt-6-luna.
"""

import ssl

import httpx

from assistant.infra import cli
from assistant.infra.config import Settings, make_erp
from assistant.infra.llm import LLMClient


class _OpenAIHTTP(httpx.Client):
    def post(self, url, **kwargs) -> httpx.Response:
        if url == "chat/completions":
            body = kwargs["json"].copy()
            body["max_completion_tokens"] = body.pop("max_tokens")
            body["reasoning_effort"] = "none"
            kwargs["json"] = body
        return super().post(url, **kwargs)


def main() -> None:
    """Запускает CLI с настройками OpenAI из окружения.

    Raises:
        SystemExit: LLM_BASE_URL не указывает на OpenAI.
    """
    settings = Settings.from_env()
    if settings.llm_base_url.rstrip("/") != "https://api.openai.com/v1":
        raise SystemExit("Для этого запуска укажите LLM_BASE_URL=https://api.openai.com/v1")

    with _OpenAIHTTP(
        base_url=settings.llm_base_url,
        headers={"Authorization": f"Bearer {settings.llm_api_key}"},
        timeout=600,
        verify=ssl.create_default_context(),
    ) as http:
        llm = LLMClient(
            settings.llm_base_url,
            settings.llm_model,
            settings.embed_model,
            api_key=settings.llm_api_key,
            cache_dir=settings.artifacts_dir / "llm_cache",
            http=http,
        )
        cli.main(context=cli.Context(settings, llm, make_erp(settings)))


if __name__ == "__main__":
    main()
