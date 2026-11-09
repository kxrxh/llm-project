"""Сообщества графа и их сводки. Требование FR-3."""

from collections import Counter
from collections.abc import Iterator

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community

SUMMARY_PROMPT = (
    "Напиши по-русски сводку сообщества до 150 слов. Используй только данные ниже. "
    "В первую очередь сохрани готовые итоги отказов по линейкам, моделям и узлам и снятые детали. "
    "Затем опиши основные связи. Сохрани точные коды, названия и числа. Вес — число чанков. "
    "Если общие итоги по линейке не приведены, не делай вывод обо всей линейке."
)


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    undirected = nx.Graph()
    undirected.add_nodes_from(sorted(graph))

    # Для Лувена встречные связи образуют одно ребро с суммарным весом.
    for left, right, weight in sorted(graph.edges(data="weight", default=1)):
        if undirected.has_edge(left, right):
            weight += undirected[left][right]["weight"]
        undirected.add_edge(left, right, weight=weight)

    groups = nx.community.louvain_communities(undirected, weight="weight", seed=7)
    ordered_groups = [sorted(nodes) for nodes in groups]
    ordered_groups.sort()
    return [Community(community_id, nodes) for community_id, nodes in enumerate(ordered_groups)]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    members = ", ".join(community.nodes)
    facts = "\n".join(fact for fact, _ in community_evidence(community, graph))

    response = llm.chat(
        [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": f"Узлы сообщества: {members}\n{facts}"},
        ]
    )
    return " ".join(response.content.split()[:150])


def community_evidence(community: Community, graph: nx.DiGraph) -> Iterator[tuple[str, set[str]]]:
    """Для каждого факта сводки возвращает его текст и подтверждающие чанки."""
    subgraph = graph.subgraph(community.nodes)

    for node in sorted(subgraph):
        if graph.nodes[node]["type"] == "ProductLine":
            yield from _line_evidence(node, graph)

        if subgraph.degree(node) == 0:
            yield node, set(graph.nodes[node]["sources"])

    for left, right, relation in sorted(subgraph.edges(data=True)):
        fact = f"{left} -{relation['type']}-> {right} (вес {relation['weight']})"
        yield fact, set(relation["sources"])


def _outgoing_relations(graph: nx.DiGraph, node: str, kind: str) -> Iterator[tuple[str, dict]]:
    for target, relation in sorted(graph[node].items()):
        if relation["type"] == kind:
            yield target, relation


def _line_evidence(line: str, graph: nx.DiGraph) -> Iterator[tuple[str, set[str]]]:
    """Собирает замены и отказы по всем моделям линейки, даже попавшим в другие сообщества."""
    line_name = graph.nodes[line]["name"]
    failures_by_model: Counter[str] = Counter()
    failures_by_component: Counter[str] = Counter()
    failure_sources: set[str] = set()

    for model, _, membership in sorted(graph.in_edges(line, data=True)):
        if membership["type"] != "BELONGS_TO":
            continue
        model_name = graph.nodes[model]["name"]

        for component, failure in _outgoing_relations(graph, model, "FAILED"):
            component_name = graph.nodes[component]["name"]
            failures_by_model[model_name] += failure["weight"]
            failures_by_component[component_name] += failure["weight"]
            failure_sources.update(membership["sources"])
            failure_sources.update(failure["sources"])

        for part, installation in _outgoing_relations(graph, model, "HAS_PART"):
            part_name = graph.nodes[part]["name"]
            for replacement, change in _outgoing_relations(graph, part, "REPLACED_BY"):
                replacement_name = graph.nodes[replacement]["name"]
                fact = f"{line_name}, {model_name}: деталь {part_name} снята с производства, замена {replacement_name}."
                sources = set(membership["sources"]).union(installation["sources"], change["sources"])
                yield fact, sources

    for grouping, counts in [("моделям", failures_by_model), ("узлам", failures_by_component)]:
        if counts:
            totals = ", ".join(f"{name} — {count}" for name, count in counts.most_common())
            yield f"{line_name}, отказы по {grouping} (суммы весов FAILED): {totals}.", failure_sources
