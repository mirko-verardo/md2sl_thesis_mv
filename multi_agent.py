from datetime import datetime
from pathlib import Path
from typing import Literal
from traceback import format_exc
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from agents.supervisor.supervisor_agent import supervisor_node
from agents.orchestrator.orchestrator_agent import orchestrator_node
from agents.generator.generator_agent import generator_node
from agents.validator.validator_agent import validator_node
from agents.assessor.assessor_agent import assessor_node
from models import AgentState, SystemMetrics
from utils import colors
from utils.general import get_model_source_from_input, get_file_format_from_input, print_colored
from utils.multi_agent import get_action_from_input



def create_session_directory(path: str) -> tuple[Path, Path]:
    """Create a session directory with timestamp and return its path."""
    
    base_dir = Path(path)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = base_dir / f"session_{session_id}"
    session_dir.mkdir(exist_ok=True)
    
    log_file = session_dir / f"conversation_{session_id}.txt"
    
    print_colored(f"\nCreated session directory: {session_dir}", colors.CYAN, bold=True)
    print_colored(f"Log file: {log_file}", colors.CYAN, bold=True)
    
    return session_dir, log_file

def route_next(state: AgentState) -> Literal["Supervisor", "Orchestrator", "Generator", "Validator", "Assessor", "FINISH"]:
    """Route to the next node based on the state."""
    return state["next_step"]

def build_workflow():
    """Build and return the workflow graph."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("Orchestrator", orchestrator_node)
    workflow.add_node("Generator", generator_node)
    workflow.add_node("Validator", validator_node)
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
            "Validator": "Validator",
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
        "Validator", 
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



if __name__ == "__main__":
    # Prompt user to enter the model source directly
    source = get_model_source_from_input()
    
    print_colored("\n=== C Parser Generator System ===", colors.CYAN, bold=True)
    print_colored("You can chat with the Supervisor about C programming or request a parser", colors.CYAN)

    # Initialize the graph
    graph = build_workflow()
    config = RunnableConfig(recursion_limit=100)

    # Initialize parameters
    messages = []
    system_metrics = SystemMetrics()
    user_action = "GENERATE_PARSER"
    file_format = get_file_format_from_input()
    # TODO: optimize this fixed prompt
    user_input = f"generate a parser function for {file_format} files"

    # Create session
    session_dir, log_file = create_session_directory(f"output/{source}/multi_agent/{file_format.lower()}")
    
    # Main interaction loop
    while True:
        user_message = f"{user_action}: {user_input}"
        
        try:
            # Start a new round
            system_metrics.start_new_round(user_message)
            
            # Initial state
            initial_state = {
                "messages": messages + [ HumanMessage(content=user_message) ],
                "user_action": user_action,
                "user_request": user_input,
                "file_format": file_format,
                "supervisor_specifications": None,
                "generator_code": None,
                "validator_compilation": None,
                "validator_testing": None,
                "assessor_assessment": None,
                "iteration_count": 0,
                "max_iterations": 10,
                "model_source": source,
                "session_dir": session_dir,
                "next_step": "Supervisor",
                "system_metrics": system_metrics
            }

            result = graph.invoke(initial_state, config)

            messages = result["messages"]
            system_metrics = result["system_metrics"]

            # Save metrics
            system_metrics.complete_round()
            system_metrics.save_summary(session_dir)
            
            # Get supervisor last response
            #supervisor_messages = [msg for msg in messages if hasattr(msg, 'name') and msg.name == "Supervisor"]
            #supervisor_response = supervisor_messages[-1].content if supervisor_messages else "no message found"
        except Exception as e:
            print_colored(f"\nAn error occurred: {e}", colors.RED, bold=True)
            print_colored(format_exc(), colors.RED, bold=True)
            print_colored("Please try again with a different query.", colors.RED, bold=True)
        
        # Ask the user again
        user_action = get_action_from_input()
        if user_action == "EXIT":
            break
        else:
            print_colored("\nYou:", colors.GREEN, bold=True)
            user_input = input()
    
    # Log conversation
    with open(log_file, "w", encoding="utf-8") as f:
        for m in messages:
            #m.pretty_print()
            f.write(f"{m.pretty_repr()}\n\n")