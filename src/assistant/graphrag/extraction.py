"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json

from pydantic import BaseModel

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    ENTITY_TYPES,
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)

SYSTEM_PROMPT = """Извлеки все сущности и связи из чанка, включая списки и таблицы.
Верни JSON с entities и relations по схеме, без пояснений.
Используй только факты чанка и названия из словаря. Факты примера не переноси.
Нормализуй коды: ar 210 -> AR-210, Е-17 -> E-17.

Модель оборудования (Model) и линейка (ProductLine) — разные сущности.
HAS_PART связывает модель с артикулом детали (Part), а IS_A — деталь с видом узла (Component).
Для каждой строки состава сохрани обе связи и SUPPLIED_BY, если указан поставщик.
INDICATES: ошибка указывает на узел. FIXED_BY: ошибка устраняется по регламенту.
FAILED: отказ узла в модели из отчёта о выезде.
REPLACED_BY: снятая деталь -> новая; обычный ремонт узла этой связи не создаёт.
Укажи IS_A для обеих деталей замены, если у них один вид узла.
Если связей схемы нет, верни пустые массивы.

Пример текста: «Модель AR-123 входит в линейку Арктика.
В ней установлен компрессор — деталь P-1234, поставщик КомпрессорМаш.
Деталь P-1234 снята с производства. Вместо неё устанавливается P-1234B».
Пример ответа:
{"entities": [{"type": "Model", "name": "AR-123"},
              {"type": "ProductLine", "name": "Арктика"},
              {"type": "Part", "name": "P-1234"},
              {"type": "Part", "name": "P-1234B"},
              {"type": "Component", "name": "компрессор"},
              {"type": "Supplier", "name": "КомпрессорМаш"}],
 "relations": [{"subject_type": "Model", "subject": "AR-123", "type": "BELONGS_TO",
                "object_type": "ProductLine", "object": "Арктика"},
               {"subject_type": "Model", "subject": "AR-123", "type": "HAS_PART",
                "object_type": "Part", "object": "P-1234"},
               {"subject_type": "Part", "subject": "P-1234", "type": "IS_A",
                "object_type": "Component", "object": "компрессор"},
               {"subject_type": "Part", "subject": "P-1234", "type": "SUPPLIED_BY",
                "object_type": "Supplier", "object": "КомпрессорМаш"},
               {"subject_type": "Part", "subject": "P-1234", "type": "REPLACED_BY",
                "object_type": "Part", "object": "P-1234B"},
               {"subject_type": "Part", "subject": "P-1234B", "type": "IS_A",
                "object_type": "Component", "object": "компрессор"}]}
"""


class _EntityData(BaseModel):
    type: str
    name: str


class _RelationData(BaseModel):
    subject_type: str
    subject: str
    type: str
    object_type: str
    object: str


class _ExtractionData(BaseModel):
    entities: list[_EntityData]
    relations: list[_RelationData]


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Извлекает сущности и связи чанка по EXTRACTION_SCHEMA (FR-1).

    Args:
        chunk: Текст с идентификаторами чанка и документа.
        llm: Клиент модели с кешированием ответов.

    Returns:
        Сущности с каноническими именами, связи и число отброшенных
        элементов в dropped. После двух некорректных ответов результат пуст.

    Raises:
        LLMError: Сервис модели недоступен или вернул ошибку.
    """
    names = vocabulary()
    codes_in_text = set(find_codes(chunk.text))
    names_in_text = set(find_names(chunk.text, names))
    signatures = "\n".join(
        f"{subject} -{kind}-> {object_type}" for kind, (subject, object_type) in RELATION_SIGNATURES.items()
    )
    messages = [
        {
            "role": "system",
            "content": f"{SYSTEM_PROMPT}\nНаправления связей:\n{signatures}\n"
            f"Словарь названий:\n{json.dumps(names, ensure_ascii=False)}",
        },
        {"role": "user", "content": chunk.text},
    ]

    for _ in range(2):
        response = llm.chat(messages, json_schema=EXTRACTION_SCHEMA, max_tokens=4096)
        try:
            data = _parse_response(response.content)
            break
        except ValueError:
            # Меняем запрос, чтобы кеш не вернул тот же ответ.
            messages.append(
                {"role": "user", "content": "Предыдущий ответ не соответствует схеме. Верни корректный JSON."}
            )
    else:
        return ExtractionResult(chunk.chunk_id, chunk.doc_id)

    def grounded_entity(kind: str, raw: str) -> Entity | None:
        name = canonical_name(kind, raw, names)
        if name is None:
            return None
        if kind in CODE_TYPES:
            if name not in codes_in_text:
                return None
        elif (kind, name) not in names_in_text:
            return None
        return Entity(kind, name)

    entities: dict[str, Entity] = {}
    relations: set[Relation] = set()
    dropped = 0

    for item in data.entities:
        entity = grounded_entity(item.type, item.name)
        if entity is None:
            dropped += 1
        else:
            entities[entity.node_id] = entity

    for item in data.relations:
        subject = grounded_entity(item.subject_type, item.subject)
        target = grounded_entity(item.object_type, item.object)
        if subject is None or target is None:
            dropped += 1
            continue

        relation = Relation(subject, item.type, target)
        if not relation.matches_schema():
            dropped += 1
            continue

        entities[subject.node_id] = subject
        entities[target.node_id] = target
        relations.add(relation)

    return ExtractionResult(
        chunk.chunk_id,
        chunk.doc_id,
        [entities[node] for node in sorted(entities)],
        sorted(relations, key=lambda relation: (relation.subject.node_id, relation.type, relation.object.node_id)),
        dropped,
    )


def _parse_response(content: str) -> _ExtractionData:
    data = _ExtractionData.model_validate_json(content, strict=True)
    if any(entity.type not in ENTITY_TYPES for entity in data.entities):
        raise ValueError("Неизвестный тип сущности")
    for relation in data.relations:
        if (
            relation.subject_type not in ENTITY_TYPES
            or relation.object_type not in ENTITY_TYPES
            or relation.type not in RELATION_SIGNATURES
        ):
            raise ValueError("Неизвестный тип сущности или связи")
    return data
