from datetime import datetime
from pathlib import Path
from typing import Literal
from traceback import format_exc
from langchain_core.messages import HumanMessage
from langgraph.graph import START, END, StateGraph
from agents.supervisor.supervisor_agent import supervisor_node
from agents.generator.generator_agent import generator_node
from agents.validator.validator_agent import validator_node
from models import AgentState, SystemMetrics
from utils import colors
from utils.general import get_model_source_from_input, get_file_format_from_input, print_colored
from utils.multi_agent import get_action_from_input



def create_session_directory(source: str) -> tuple[Path, Path]:
    """Create a session directory with timestamp and return its path."""
    full_path = f"output/{source}/multi_agent"
    
    base_dir = Path(full_path)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = base_dir / f"session_{session_id}"
    session_dir.mkdir(exist_ok=True)
    
    log_file = session_dir / f"conversation_{session_id}.txt"
    
    print_colored(f"\nCreated session directory: {session_dir}", colors.CYAN, bold=True)
    print_colored(f"Log file: {log_file}", colors.CYAN, bold=True)
    
    return session_dir, log_file

def route_next(state: AgentState) -> Literal["Generator", "Validator", "Supervisor", "FINISH"]:
    """Route to the next node based on the state."""
    return state["next_step"]

def build_workflow():
    """Build and return the workflow graph."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("Generator", generator_node)
    workflow.add_node("Validator", validator_node)
    
    # Set the entry point
    workflow.add_edge(START, "Supervisor")

    # Add conditional edges
    workflow.add_conditional_edges(
        "Supervisor",
        route_next,
        {
            "Generator": "Generator",
            "FINISH": END
        }
    )
    
    workflow.add_conditional_edges(
        "Generator", 
        route_next,
        {
            "Validator": "Validator"
        }
    )
    
    workflow.add_conditional_edges(
        "Validator", 
        route_next,
        {
            "Generator": "Generator",
            "Supervisor": "Supervisor"
        }
    )
    
    return workflow.compile()



if __name__ == "__main__":
    # Prompt user to enter the model source directly
    source = get_model_source_from_input()
    
    print_colored("\n=== C Parser Generator System ===", colors.CYAN, bold=True)
    print_colored("You can chat with the Supervisor about C programming or request a parser", colors.CYAN)
    print_colored("Enter your message or type 'exit' to quit", colors.CYAN)
    print_colored(f"Session files will be saved in the 'output/{source}/multi_agent/' directory", colors.CYAN)

    # Create session
    session_dir, log_file = create_session_directory(source)
    
    # Initialize parameters
    messages = []
    system_metrics = SystemMetrics()
    start = True
    file_format = ""

    # Initialize the graph
    graph = build_workflow()
    
    # Main interaction loop
    while True:
        action = get_action_from_input(start)
        start = False

        if action == "EXIT":
            break
        elif action == "GENERATE_PARSER":
            file_format = get_file_format_from_input()
            # TODO: optimize this fixed prompt
            user_input = f"generate a parser function for {file_format} files"
        else:
            print_colored("\nYou:", colors.GREEN, bold=True)
            user_input = input()

        user_message = f"{action}: {user_input}"
        
        try:
            # Start a new round
            system_metrics.start_new_round(user_message)
            
            # Initial state
            initial_state = {
                "messages": messages + [ HumanMessage(content=user_message) ],
                "user_action": action,
                "user_request": user_input,
                "file_format": file_format,
                "generator_specs": None,
                "generator_code": None,
                "validator_assessment": None,
                "iteration_count": 0,
                "max_iterations": 3,
                "model_source": source,
                "session_dir": session_dir,
                "next_step": "Supervisor",
                "system_metrics": system_metrics
            }

            result = graph.invoke(initial_state)

            messages = result["messages"]
            system_metrics = result["system_metrics"]

            # Save metrics
            system_metrics.complete_round()
            system_metrics.save_summary(session_dir)
            
            # Get supervisor response
            #supervisor_messages = [msg for msg in messages if hasattr(msg, 'name') and msg.name == "Supervisor"]
            #supervisor_response = supervisor_messages[-1].content if supervisor_messages else "no message found"
            supervisor_response = messages[-1].content if messages else "no message found"
            print_colored("\nSupervisor response:", colors.GREEN, bold=True)
            print(supervisor_response)
        except Exception as e:
            print_colored(f"\nAn error occurred: {e}", colors.RED, bold=True)
            print_colored(format_exc(), colors.RED, bold=True)
            print_colored("Please try again with a different query.", colors.RED, bold=True)
    
    # Log conversation
    with open(log_file, "w", encoding="utf-8") as f:
        for m in messages:
            #m.pretty_print()
            f.write(f"{m.pretty_repr()}\n\n")