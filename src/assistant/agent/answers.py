from assistant.agent.validation import unsupported_facts
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import AgentResult

_NOT_FOUND = "В базе знаний не найдено"
_NO_DATA = f"{_NOT_FOUND} достаточно подтвержденных данных для ответа."


def grounded_answer(
    text: str, question: str, observations: list[dict], messages: list[dict], llm: LLMClient
) -> tuple[str, bool]:
    if not text.strip():
        return _NO_DATA, True

    if not unsupported_facts(text, question, observations):
        return text, False

    messages.append({"role": "assistant", "content": text})
    messages.append(
        {
            "role": "user",
            "content": "Исправь ответ: убери неподтвержденные коды, цены и остатки. Инструменты не вызывай.",
        }
    )

    correction = llm.chat(messages)
    if correction.tool_calls or not correction.content.strip():
        return _NO_DATA, True

    if unsupported_facts(correction.content, question, observations):
        return _NO_DATA, True

    return correction.content, False


def partial_answer(observations: list[dict]) -> str:
    parts = []
    for result in observations:
        if result.get("answer"):
            parts.append(result["answer"])
        elif result.get("order_id"):
            parts.append(f"Создан заказ {result['order_id']}.")
        elif "price_rub" in result:
            parts.append(f"Цена {result['part_id']}: {result['price_rub']} руб.")
        elif "qty" in result:
            parts.append(f"Остаток {result['part_id']} на складе {result['warehouse']}: {result['qty']} шт.")

    return "\n".join(dict.fromkeys(parts))


def finalize_result(result: AgentResult) -> AgentResult:
    observations = [
        step.result for step in result.steps if step.result and not step.error and not step.result.get("refused")
    ]
    result.sources = sorted({source for observation in observations for source in observation.get("sources", [])})
    errors = [step.result for step in result.steps if step.result and step.result.get("error_type")]
    missing_data = any(step.result and step.result.get("refused") for step in result.steps)
    missing_data |= any(error["error_type"] == "not_found" for error in errors)

    if not result.refused and _NOT_FOUND.lower() in result.answer.lower():
        if observations:
            if not missing_data:
                result.answer = partial_answer(observations)
        elif errors and not missing_data:
            result.answer = "\n".join(dict.fromkeys(error["error"] for error in errors))
        else:
            result.refused = True

    needs_input = result.needs_confirmation or bool(result.steps and result.steps[-1].result is None)
    if missing_data and not observations and not needs_input:
        if _NOT_FOUND.lower() not in result.answer.lower():
            result.answer += f"\n{_NOT_FOUND}."
        result.refused = True

    for error in errors:
        if error["error"].lower() not in result.answer.lower():
            result.answer += f"\n{error['error']}"

    for observation in observations:
        order_id = observation.get("order_id")
        if order_id and order_id not in result.answer:
            result.answer += f"\nСоздан заказ {order_id}."

    return result
