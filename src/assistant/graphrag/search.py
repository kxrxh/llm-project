"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from itertools import combinations, pairwise

import networkx as nx

from assistant.graphrag.communities import community_evidence
from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

Edge = tuple[str, str]

NOT_FOUND = "в базе знаний не найдено"
MAX_FACT_CHARS = 8000

LOCAL_PROMPT = (
    "Ответь кратко по-русски только по пронумерованным фактам. Стрелка идёт от субъекта к объекту. "
    "REPLACED_BY ведёт от старой детали к новой: для заказа укажи последнюю деталь цепочки. "
    "Используй точные коды. Если вопрос составной, сообщи подтверждённую часть. "
    f"Если подходящих фактов нет, ответь ровно «{NOT_FOUND}»."
)

MAP_PROMPT = (
    "Ответь на вопрос только по данной сводке. Верни JSON с частичным ответом answer "
    "и целой оценкой полезности score от 0 до 10. Если нужных сведений нет, поставь 0. "
    "Для вопросов о частоте в линейке нужны подсчёты именно по указанной линейке. "
    "В answer приведи краткое утверждение вместе с основанием: для агрегатов "
    "укажи название линейки, узел или модель и готовое число из сводки. "
    "Ответ «недостаточно данных» всегда получает score 0."
)

REDUCE_PROMPT = (
    "Дай краткий ответ по-русски, опираясь только на частичные ответы. Учитывай указанную "
    "линейку и готовые итоги, не складывай повторяющиеся цифры. Сохраняй точные названия. "
    f"Если данных для вывода нет, ответь ровно «{NOT_FOUND}»."
)


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    anchors = _find_anchors(question, kb)
    if not anchors:
        return Answer(NOT_FOUND, refused=True)

    graph = kb.graph
    lowered = question.casefold()
    include_suppliers = "поставщик" in lowered or "поставля" in lowered
    if len(anchors) == 1:
        edges = _neighborhood_edges(graph, anchors[0])
    else:
        edges = _edges_between_anchors(graph, anchors, include_suppliers)

    facts: list[str] = []
    numbered_facts: list[str] = []
    documents: set[str] = set()
    remaining_chars = MAX_FACT_CHARS

    for left, right in edges:
        relation = graph[left][right]
        fact = f"{graph.nodes[left]['name']} -{relation['type']}-> {graph.nodes[right]['name']}"
        line = f"{len(facts) + 1}. {fact}"
        if len(line) > remaining_chars:
            continue

        facts.append(fact)
        numbered_facts.append(line)
        remaining_chars -= len(line) + 1  # перевод строки перед следующим фактом
        documents.update(chunk.partition("#")[0] for chunk in relation["sources"])

    if not facts:
        return Answer(NOT_FOUND, refused=True)

    context = "\n".join(numbered_facts)
    system_prompt = LOCAL_PROMPT
    if include_suppliers:
        system_prompt += (
            " В ответе о поставщике укажи точную пару «артикул — поставщик»"
            " из соответствующего факта SUPPLIED_BY."
        )
    response = llm.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Факты:\n{context}\n\nВопрос: {question}"},
        ]
    )
    return _answer(response.content, documents, facts)


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    partial_answers: list[str] = []
    documents: set[str] = set()

    for community in kb.communities:
        response = llm.chat(
            [
                {"role": "system", "content": MAP_PROMPT},
                {"role": "user", "content": f"Сводка:\n{community.summary}\n\nВопрос: {question}"},
            ],
            json_schema=MAP_SCHEMA,
        )
        partial_answer = _parse_useful_answer(response.content)
        if partial_answer is None:
            continue

        partial_answers.append(partial_answer)
        for _, chunks in community_evidence(community, kb.graph):
            documents.update(chunk.partition("#")[0] for chunk in chunks)

    if not partial_answers:
        return Answer(NOT_FOUND, refused=True)

    context = "\n".join(partial_answers)
    response = llm.chat(
        [
            {"role": "system", "content": REDUCE_PROMPT},
            {"role": "user", "content": f"Частичные ответы:\n{context}\n\nВопрос: {question}"},
        ]
    )
    return _answer(response.content, documents, partial_answers)


def _find_anchors(question: str, kb: KnowledgeBase) -> list[str]:
    """Неизвестный код делает ответ по остальным опорам ненадёжным."""
    anchors = []
    for code in find_codes(question):
        node = f"{code_type(code)}:{code}"
        if node not in kb.graph:
            return []
        anchors.append(node)

    for kind, name in find_names(question, kb.vocabulary):
        node = f"{kind}:{name}"
        if node in kb.graph:
            anchors.append(node)

    return anchors


def _neighborhood_edges(graph: nx.DiGraph, anchor: str) -> list[Edge]:
    undirected = graph.to_undirected(as_view=True)
    distances = nx.single_source_shortest_path_length(undirected, anchor, cutoff=2)

    def nearest_endpoint(edge: Edge) -> tuple[int, Edge]:
        left, right = edge
        return min(distances[left], distances[right]), edge

    return sorted(graph.subgraph(distances).edges, key=nearest_endpoint)


def _edges_between_anchors(graph: nx.DiGraph, anchors: list[str], include_suppliers: bool) -> list[Edge]:
    undirected = graph.to_undirected(as_view=True)
    edges: set[Edge] = set()

    for start, end in combinations(anchors, 2):
        for path in nx.all_simple_paths(undirected, start, end, cutoff=3):
            # Идти по пути можно в обе стороны. Факт сохраняет направление из исходного графа.
            for left, right in pairwise(path):
                if graph.has_edge(left, right):
                    edges.add((left, right))
                if graph.has_edge(right, left):
                    edges.add((right, left))

    _add_part_replacements(graph, edges, include_suppliers)

    def fact_priority(edge: Edge) -> tuple[bool, bool, Edge]:
        left, right = edge
        is_replacement = graph.edges[edge]["type"] == "REPLACED_BY"
        touches_anchor = left in anchors or right in anchors
        return not is_replacement, not touches_anchor, edge

    # Если все факты не поместятся в запрос, первыми попадут замены, затем связи самих опор.
    return sorted(edges, key=fact_priority)


def _add_part_replacements(graph: nx.DiGraph, edges: set[Edge], include_suppliers: bool) -> None:
    path_nodes = graph.edge_subgraph(edges).nodes
    pending = {node for node in path_nodes if graph.nodes[node]["type"] == "Part"}
    visited: set[str] = set()

    while pending:
        part = pending.pop()
        visited.add(part)
        outgoing = graph[part]

        replacements = [target for target, relation in outgoing.items() if relation["type"] == "REPLACED_BY"]
        for replacement in replacements:
            edges.add((part, replacement))
            if replacement not in visited:
                pending.add(replacement)

        if include_suppliers and not replacements:
            for supplier, relation in outgoing.items():
                if relation["type"] == "SUPPLIED_BY":
                    edges.add((part, supplier))


def _parse_useful_answer(content: str) -> str | None:
    """Возвращает текст map-ответа с положительной оценкой, если JSON соответствует схеме."""
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return None

    if not isinstance(result, dict):
        return None

    score = result.get("score")
    if type(score) is not int or not 1 <= score <= 10:
        return None

    answer = result.get("answer")
    if not isinstance(answer, str):
        return None
    return answer.strip() or None


def _answer(text: str, sources: set[str], facts: list[str]) -> Answer:
    text = text.strip()
    if not text or text.casefold().startswith(NOT_FOUND):
        return Answer(NOT_FOUND, refused=True)

    return Answer(text, sources=sorted(sources), facts=facts)
