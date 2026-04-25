"""
LangGraph assembly — wires all nodes into a compiled graph.

Graph flow:
    retrieve → reason → [conditional] → answer → evaluate → finalize
                          |
                          └─ needs_clarification? → finalize (skip answer/eval)

To add new nodes: import your node function, add it with graph.add_node(),
and connect it with graph.add_edge() or graph.add_conditional_edges().
"""
from langgraph.graph import END, StateGraph

from app.nodes.answer import answer_node
from app.nodes.evaluate import evaluate_node
from app.nodes.execute_plan import execute_plan_node
from app.nodes.finalize import finalize_node
from app.nodes.reason import reason_node
from app.nodes.retrieve import retrieve_node
from app.state import GraphState


def _route_after_reason(state: GraphState) -> str:
    """Route to 'execute_plan' normally, or 'finalize' if clarification is needed."""
    if state.get("needs_clarification"):
        return "finalize"
    return "execute_plan"


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reason", reason_node)
    graph.add_node("execute_plan", execute_plan_node)
    graph.add_node("answer", answer_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_conditional_edges(
        "reason",
        _route_after_reason,
        {"execute_plan": "execute_plan", "finalize": "finalize"},
    )
    graph.add_edge("execute_plan", "answer")
    graph.add_edge("answer", "evaluate")
    graph.add_edge("evaluate", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


# Module-level compiled graph (imported by run.py and scripts)
compiled_graph = build_graph()
