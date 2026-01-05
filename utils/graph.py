from pathlib import Path
from typing import Any
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from agents.supervisor.supervisor_agent import supervisor_node
from agents.orchestrator.orchestrator_agent import orchestrator_node
from agents.generator.generator_agent import generator_node
from agents.compiler.compiler_agent import compiler_node
from agents.tester.tester_agent import tester_node
from agents.assessor.assessor_agent import assessor_node
from models import AgentType, AgentState, BenchmarkMetrics



def route_next(state: AgentState) -> AgentType:
    """Route to the next node based on the state."""
    return state["next_step"]

def build_workflow():
    """Build and return the workflow graph."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("Orchestrator", orchestrator_node)
    workflow.add_node("Generator", generator_node)
    workflow.add_node("Compiler", compiler_node)
    workflow.add_node("Tester", tester_node)
    workflow.add_node("Assessor", assessor_node)
    
    # Set the entry point
    workflow.add_edge(START, "Supervisor")

    # Add conditional edges
    workflow.add_conditional_edges(
        "Supervisor",
        route_next,
        {
            "Orchestrator": "Orchestrator",
            "FINISH": END
        }
    )

    workflow.add_conditional_edges(
        "Orchestrator", 
        route_next,
        {
            "Supervisor": "Supervisor",
            "Generator": "Generator",
            "Compiler": "Compiler",
            "Tester": "Tester",
            "Assessor": "Assessor"
        }
    )
    
    workflow.add_conditional_edges(
        "Generator", 
        route_next,
        {
            "Orchestrator": "Orchestrator"
        }
    )
    
    workflow.add_conditional_edges(
        "Compiler", 
        route_next,
        {
            "Orchestrator": "Orchestrator"
        }
    )

    workflow.add_conditional_edges(
        "Tester", 
        route_next,
        {
            "Orchestrator": "Orchestrator"
        }
    )

    workflow.add_conditional_edges(
        "Assessor", 
        route_next,
        {
            "Orchestrator": "Orchestrator"
        }
    )
    
    return workflow.compile()

def start_workflow(
        graph, config: RunnableConfig, 
        user_action: str, user_request: str, file_format: str, round: int, max_iterations: int, 
        model_source: str, session_dir: Path, benchmark_metrics: BenchmarkMetrics, last_parser: dict[str, str] = {}
    ) -> dict[str, Any]:
    """Start the workflow graph."""
    user_message = f"{user_action}: {user_request}"
    
    initial_state = {
        "messages": [ HumanMessage(content=user_message) ],
        "user_action": user_action,
        "user_request": user_request,
        "file_format": file_format,
        "supervisor_specifications": None,
        "generator_code": last_parser.get("code"),
        "compiler_result": None,
        "tester_result": None,
        "code_assessment": last_parser.get("assessment"),
        "round": round,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "model_source": model_source,
        "session_dir": session_dir,
        "next_step": "Supervisor",
        "benchmark_metrics": benchmark_metrics
    }

    return graph.invoke(initial_state, config)
