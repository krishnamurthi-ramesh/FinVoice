from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.agent.nodes import stt_node, intent_node, tool_execution_node, handoff_node, generation_node, tts_node

def route_intent(state: AgentState):
    if state.get("requires_human"):
        return "handoff"
    return "tools"

def build_bfsi_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("stt", stt_node)
    workflow.add_node("intent", intent_node)
    workflow.add_node("tools", tool_execution_node)
    workflow.add_node("handoff", handoff_node)
    workflow.add_node("generation", generation_node)
    workflow.add_node("tts", tts_node)
    
    workflow.set_entry_point("stt")
    workflow.add_edge("stt", "intent")
    
    workflow.add_conditional_edges("intent", route_intent, {
        "handoff": "handoff",
        "tools": "tools"
    })
    
    workflow.add_edge("tools", "generation")
    workflow.add_edge("handoff", "tts")
    workflow.add_edge("generation", "tts")
    workflow.add_edge("tts", END)
    
    return workflow.compile()

bfsi_agent = build_bfsi_graph()
