"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Собирает граф знаний из результатов извлечения (FR-2).

    Args:
        results: Сущности и связи с идентификаторами исходных чанков.

    Returns:
        DiGraph с атрибутами type, name и sources у узлов, type, weight
        и sources у рёбер. Вес равен числу разных чанков. Связи вне схемы
        исключаются. Из встречных замен остаётся связь с большим весом,
        при равенстве удаляются обе. Число конфликтов записывается
        в graph.graph["conflicts"].
    """
    graph = nx.DiGraph(conflicts=0)
    for result in results:
        relations = [relation for relation in result.relations if relation.matches_schema()]
        entities = result.entities + [
            entity for relation in relations for entity in (relation.subject, relation.object)
        ]
        for entity in entities:
            if entity.node_id not in graph:
                graph.add_node(entity.node_id, type=entity.type, name=entity.name, sources=set())
            graph.nodes[entity.node_id]["sources"].add(result.chunk_id)

        for relation in relations:
            source, target = relation.subject.node_id, relation.object.node_id
            if not graph.has_edge(source, target):
                graph.add_edge(source, target, type=relation.type, sources=set())
            graph[source][target]["sources"].add(result.chunk_id)

    for _, attributes in graph.nodes(data=True):
        attributes["sources"] = sorted(attributes["sources"])
    for _, _, attributes in graph.edges(data=True):
        attributes["sources"] = sorted(attributes["sources"])
        attributes["weight"] = len(attributes["sources"])

    for source, target, forward in list(graph.edges(data=True)):
        if forward["type"] != "REPLACED_BY" or source >= target:
            continue
        backward = graph.get_edge_data(target, source)
        if backward is None or backward["type"] != "REPLACED_BY":
            continue

        graph.graph["conflicts"] += 1
        if forward["weight"] <= backward["weight"]:
            graph.remove_edge(source, target)
        if forward["weight"] >= backward["weight"]:
            graph.remove_edge(target, source)

    return graph
