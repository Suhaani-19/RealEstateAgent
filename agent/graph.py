from langgraph.graph import StateGraph, END
from agent.state import PropertyState
from agent.nodes import node_predict_price, node_retrieve_market, node_generate_advisory

def build_graph():
    graph = StateGraph(PropertyState)

    # Add the 3 nodes
    graph.add_node("predict_price", node_predict_price)
    graph.add_node("retrieve_market", node_retrieve_market)
    graph.add_node("generate_advisory", node_generate_advisory)

    # Connect them in order
    graph.set_entry_point("predict_price")
    graph.add_edge("predict_price", "retrieve_market")
    graph.add_edge("retrieve_market", "generate_advisory")
    graph.add_edge("generate_advisory", END)

    return graph.compile()