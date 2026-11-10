"""Инструменты агента. Требование FR-6."""

import re
from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
PART = {"type": "string", "pattern": "^P-[0-9]{4}[A-Z]?$"}
WAREHOUSE = {"type": "string", "enum": WAREHOUSES}
QUESTION = {"type": "string", "minLength": 1}
MODEL = {"type": "string", "pattern": "^(AR|BZ|GS)-[0-9]{3}$"}


def _schema(name: str, description: str, properties: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _schema(
        "search_knowledge",
        "Поиск по конкретным ошибкам, моделям, деталям, заменам, регламентам и поставщикам. "
        "Для обзора линейки используй summarize_topic.",
        {"question": QUESTION},
    ),
    _schema(
        "summarize_topic",
        "Обзор: частота отказов, сравнение линеек, перечни деталей линейки.",
        {"question": QUESTION},
    ),
    _schema(
        "check_stock",
        "Остаток известной детали на указанном складе и признак снятия с производства.",
        {"part_id": PART, "warehouse": WAREHOUSE},
    ),
    _schema(
        "get_price",
        "Цена и статус известной детали.",
        {"part_id": PART},
    ),
    _schema(
        "create_service_order",
        "Создать заказ только по запросу пользователя. Количество должно быть указано пользователем.",
        {"part_id": PART, "warehouse": WAREHOUSE, "qty": {"type": "integer", "minimum": 1}, "model": MODEL},
    ),
]


def validate_tool_arguments(name: str, arguments: dict) -> str | None:
    """Проверка аргументов по тем же ограничениям, которые получает модель."""
    schema = next(
        (item["function"]["parameters"] for item in TOOL_SCHEMAS if item["function"]["name"] == name),
        None,
    )
    if schema is None:
        return f"Неизвестный инструмент: {name}"
    if not isinstance(arguments, dict):
        return "Аргументы должны быть объектом JSON"
    if set(arguments) != set(schema["required"]):
        return f"Ожидаются только аргументы: {', '.join(schema['required'])}"

    for key, rule in schema["properties"].items():
        value = arguments[key]
        if rule["type"] == "integer":
            if type(value) is not int or value < rule["minimum"]:
                return f"{key}: требуется целое количество больше нуля"
        else:
            if not isinstance(value, str) or not value.strip():
                return f"{key}: требуется непустая строка"
            if "enum" in rule and value not in rule["enum"]:
                return f"{key}: допустимые значения: {', '.join(rule['enum'])}"
            if "pattern" in rule and re.fullmatch(rule["pattern"], value) is None:
                return f"{key}: неверный формат кода"

    return None


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` возвращаются с `error` и `error_type`, чтобы различать исходы.
    `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """

    def search_knowledge(question: str) -> dict:
        result = local_search(question, kb, llm)
        return {"answer": result.text, "facts": result.facts, "sources": result.sources, "refused": result.refused}

    def summarize_topic(question: str) -> dict:
        result = global_search(question, kb, llm)
        return {"answer": result.text, "sources": result.sources, "refused": result.refused}

    def checked(name: str, function: Callable[..., dict]) -> Callable[..., dict]:
        def tool(**arguments) -> dict:
            error = validate_tool_arguments(name, arguments)
            if error:
                return {"error": error}

            try:
                return function(**arguments)
            except ErpNotFound as exc:
                return {"error": f"Данных в ERP нет: {exc}", "error_type": "not_found"}
            except ErpRejected as exc:
                return {"error": f"ERP отклонила запрос: {exc}", "error_type": "rejected"}

        return tool

    functions = {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": erp.get_stock,
        "get_price": erp.get_price,
        "create_service_order": erp.create_order,
    }

    return {name: checked(name, function) for name, function in functions.items()}
