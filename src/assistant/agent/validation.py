import json
import re
from collections.abc import Callable, Iterator
from decimal import Decimal

from assistant.agent.tools import validate_tool_arguments
from assistant.infra.grounding import find_codes, normalize_code
from assistant.infra.llm import ToolCall

_CODE_CANDIDATE = re.compile(r"(?<!\w)[A-Za-zА-Яа-я]{1,2}[- ]?[0-9]{2,4}[A-Za-zА-Яа-я]?(?!\w)")
_ORDER_ID = re.compile(r"\bSO-[0-9]+\b")
_ORDER_CREATED = re.compile(
    r"\b(?:заказ\s+(?:SO-[0-9]+\s+)?(?:успешно\s+)?(?:создан|оформлен)|(?:создал|оформил)[аи]?\s+заказ)",
    re.IGNORECASE,
)

_WORD_VALUES: dict[str, int] = dict(
    zip(
        "ноль один два три четыре пять шесть семь восемь девять десять одиннадцать двенадцать тринадцать "
        "четырнадцать пятнадцать шестнадцать семнадцать восемнадцать девятнадцать".split(),
        range(20),
        strict=True,
    )
)
_WORD_VALUES.update({"одна": 1, "одно": 1, "одну": 1, "две": 2})
_WORD_VALUES.update(
    dict(
        zip(
            "двадцать тридцать сорок пятьдесят шестьдесят семьдесят восемьдесят девяносто".split(),
            range(20, 100, 10),
            strict=True,
        )
    )
)
_WORD_VALUES.update(
    dict(
        zip(
            "сто двести триста четыреста пятьсот шестьсот семьсот восемьсот девятьсот".split(),
            range(100, 1000, 100),
            strict=True,
        )
    )
)
_SCALES = {"тысяча": 1000, "тысячи": 1000, "тысяч": 1000, "миллион": 1000000, "миллиона": 1000000, "миллионов": 1000000}
_WORDS = "|".join(sorted((*_WORD_VALUES, *_SCALES, "минус", "полтора", "полторы"), key=len, reverse=True))
_DIGITS = r"[+-]?(?:[0-9]{1,3}(?:[ \u00a0\u202f][0-9]{3})+|[0-9]+)(?:[.,][0-9]+)?"
_NUMBER = rf"(?<![\w.,-])(?:{_DIGITS}|(?:{_WORDS})(?:[ \t\u00a0\u202f]+(?:{_WORDS}))*)(?![\w]|[.,][0-9])"
# Не принимай окончания слов «Север.» и «труб.» за денежные сокращения.
_PRICE_UNIT = re.compile(r"(?:(?<![^\W\d])(?:руб(?:ль|ля|лей|\.)?|р\.)|₽)(?!\w)", re.IGNORECASE)
_STOCK_UNIT = re.compile(r"(?:шт(?:ук(?:а|и)?)?\.?|единиц(?:а|ы)?)(?!\w)", re.IGNORECASE)
_PRICE = re.compile(rf"({_NUMBER})\s*{_PRICE_UNIT.pattern}", re.IGNORECASE)
_STOCK = re.compile(rf"({_NUMBER})\s*(?:шт\w*|единиц\w*|детал\w*)", re.IGNORECASE)
_PRICE_LABEL = re.compile(rf"(?:цена|стоимость|стоит)\s*[:=—-]?\s*({_NUMBER})", re.IGNORECASE)
_STOCK_LABEL = re.compile(rf"(?:остаток|осталось|в наличии|на складе)\s*[:=—-]?\s*({_NUMBER})", re.IGNORECASE)
_QUANTITY_LABEL = re.compile(rf"(?:закажи(?:те)?|заказать|заказ|количество)\s*[:=—-]?\s*({_NUMBER})", re.IGNORECASE)
_QUANTITY_PART = re.compile(rf"({_NUMBER})\s+(?:P|Р)-[0-9]{{4}}[A-Z]?(?!\w)", re.IGNORECASE)
_QUANTITY_RANGES = (
    re.compile(rf"(?<!\w)({_DIGITS})\s*[-–—]\s*({_DIGITS})(?!\w)"),
    re.compile(rf"(?:от\s+)?({_NUMBER})\s+до\s+({_NUMBER})", re.IGNORECASE),
)
_WAREHOUSE_NAMES = {
    "Москва": r"\bмоскв(?:а|е|ы|у)\b",
    "Санкт-Петербург": r"\bсанкт-петербург(?:е|а)?\b",
    "Казань": r"\bказан(?:ь|и)\b",
    "Новосибирск": r"\bновосибирск(?:е|а)?\b",
}


def validate_tool_call(
    call: ToolCall, tools: dict[str, Callable[..., dict]], question: str, observations: list[dict]
) -> str | None:
    if call.name not in tools:
        return f"Неизвестный инструмент: {call.name}"

    error = validate_tool_arguments(call.name, call.arguments)
    if error:
        return error

    if "part_id" in call.arguments:
        known_codes = find_codes(evidence_text(question, observations))
        if call.arguments["part_id"] not in known_codes:
            return "Номер детали не подтвержден. Сначала вызови search_knowledge."
        if call.name == "create_service_order" and call.arguments["model"] not in known_codes:
            return "Модель оборудования не подтверждена. Уточни модель у пользователя."

    return None


def _without_codes(text: str) -> str:
    return _ORDER_ID.sub(" ", _CODE_CANDIDATE.sub(lambda m: " " if normalize_code(m[0]) else m[0], text))


def _number(raw: str) -> Decimal | None:
    if re.fullmatch(_DIGITS, raw):
        return Decimal(re.sub(r"\s", "", raw).replace(",", "."))

    words = raw.casefold().split()
    sign = -1 if words[0] == "минус" else 1
    if sign < 0:
        words = words[1:]

    total = group = 0
    previous_rank = 4
    previous_scale = 1000000000
    for word in words:
        if word in _SCALES:
            scale = _SCALES[word]
            if scale >= previous_scale:
                return None
            total += (group or 1) * scale
            group = 0
            previous_rank = 4
            previous_scale = scale
            continue

        value = _WORD_VALUES.get(word)
        if value is None:
            return None
        rank = 3 if value >= 100 else 2 if value >= 20 else 1
        if rank >= previous_rank or (previous_rank == 2 and value >= 10) or (value == 0 and len(words) > 1):
            return None
        group += value
        previous_rank = rank

    return Decimal(sign * (total + group)) if words else None


def _measurements(text: str, *patterns: re.Pattern) -> set[Decimal]:
    text = re.sub(r"[*_`]", "", text)
    return {
        value for pattern in patterns for match in pattern.finditer(text) if (value := _number(match[1])) is not None
    }


def _numbers(text: str) -> set[Decimal]:
    text = re.sub(r"[*_`]", "", text)
    text = re.sub(r"(?m)^\s*[0-9]+[.)]\s+|\[[0-9]+\]", " ", _without_codes(text))
    return {value for match in re.finditer(_NUMBER, text, re.IGNORECASE) if (value := _number(match[0])) is not None}


def _clauses(text: str) -> list[str]:
    return re.split(r"[,;.!?]\s+|\n", text)


def _measurement_clauses(text: str) -> Iterator[tuple[str, set[str], set[str]]]:
    for sentence in re.split(r"[;.!?]\s+|\n", text):
        parts: set[str] = set()
        warehouses: set[str] = set()
        for clause in re.split(r",\s+", sentence):
            parts = {code for code in find_codes(clause) if code.startswith("P-")} or parts
            warehouses = {
                name for name, pattern in _WAREHOUSE_NAMES.items() if re.search(pattern, clause, re.IGNORECASE)
            } or warehouses
            yield clause, parts, warehouses


def evidence_text(question: str, observations: list[dict]) -> str:
    results = [{key: value for key, value in result.items() if key != "sources"} for result in observations]
    return question + "\n" + json.dumps(results, ensure_ascii=False)


def _unsupported_measurements(
    text: str, question: str, observations: list[dict], field: str, patterns: tuple[re.Pattern, ...], unit: re.Pattern
) -> bool:
    order_quantity = False
    for clause, parts, warehouses in _measurement_clauses(re.sub(r"[*_`]", "", text)):
        order_quantity = (
            field == "qty"
            and not _STOCK_LABEL.search(clause)
            and ("заказ" in clause.casefold() or (order_quantity and _QUANTITY_LABEL.search(clause) is not None))
        )
        allowed = set()
        for question_clause, question_parts, question_warehouses in _measurement_clauses(question):
            if parts and question_parts and parts.isdisjoint(question_parts):
                continue
            if field == "qty" and warehouses and question_warehouses and warehouses.isdisjoint(question_warehouses):
                continue
            if field == "qty" and not order_quantity and _QUANTITY_LABEL.search(question_clause):
                continue
            allowed |= _measurements(question_clause, *patterns)
        for result in observations:
            if parts and result.get("part_id") not in parts:
                continue
            if field == "qty" and ("order_id" in result) != order_quantity:
                continue
            if field == "qty" and warehouses and result.get("warehouse") not in warehouses:
                continue
            if field in result:
                allowed.add(Decimal(str(result[field])))

        matches = [match for pattern in patterns for match in pattern.finditer(clause)]
        if any(_number(match[1]) not in allowed for match in matches):
            return True
        if any(
            not any(match.start() <= marker.start() < match.end() for match in matches)
            for marker in unit.finditer(clause)
        ):
            return True

    return False


def unsupported_facts(text: str, question: str, observations: list[dict]) -> bool:
    evidence = evidence_text(question, observations)
    if set(find_codes(text)) - set(find_codes(evidence)):
        return True

    if set(_ORDER_ID.findall(text)) - set(_ORDER_ID.findall(evidence)):
        return True
    if _ORDER_CREATED.search(text) and not any(result.get("order_id") for result in observations):
        return True

    numbers = _numbers(question)
    for result in observations:
        numbers |= _numbers(result.get("answer", ""))
        numbers |= _numbers("\n".join(result.get("facts", [])))
        numbers |= {Decimal(str(result[field])) for field in ("price_rub", "qty") if field in result}

    return bool(
        _numbers(text) - numbers
        or _unsupported_measurements(text, question, observations, "price_rub", (_PRICE, _PRICE_LABEL), _PRICE_UNIT)
        or _unsupported_measurements(text, question, observations, "qty", (_STOCK, _STOCK_LABEL), _STOCK_UNIT)
    )


def question_quantities(question: str) -> set[Decimal]:
    quantities: set[Decimal] = set()
    for clause in _clauses(question):
        if _STOCK_LABEL.search(clause) and not _QUANTITY_LABEL.search(clause):
            continue
        clause_quantities = _measurements(clause, _STOCK, _QUANTITY_LABEL, _QUANTITY_PART)
        if clause_quantities:
            for pattern in _QUANTITY_RANGES:
                for match in pattern.finditer(clause):
                    for raw in match.groups():
                        value = _number(raw)
                        if value is None:
                            return set()
                        clause_quantities.add(value)
        quantities |= clause_quantities

    return quantities
