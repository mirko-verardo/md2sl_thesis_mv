from langgraph.graph import START, END, StateGraph
from agents.supervisor.supervisor_agent import supervisor_node
from agents.orchestrator.orchestrator_agent import orchestrator_node
from agents.generator.generator_agent import generator_node
from agents.compiler.compiler_agent import compiler_node
from agents.tester.tester_agent import tester_node
from agents.assessor.assessor_agent import assessor_node
from models import AgentType, AgentState



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
