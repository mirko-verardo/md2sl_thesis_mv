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
from utils.general import get_model_source_from_input, print_colored
from utils.multi_agent import get_satisfaction, get_compilation_status



def create_session_directory(source: str, base_path: str = "output") -> tuple[Path, Path]:
    """Create a session directory with timestamp and return its path."""
    full_path = f"{base_path}/{source}/multi_agent"
    
    base_dir = Path(full_path)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = base_dir / f"session_{session_id}"
    session_dir.mkdir(exist_ok=True)
    
    log_file = session_dir / f"conversation_{session_id}.txt"
    with open(log_file, 'w') as f:
        f.write(f"=== C Parser Generator Chat - {datetime.now()} ===\n")
    
    print_colored(f"\nCreated session directory: {session_dir}", "1;36")
    print_colored(f"Log file: {log_file}", "1;36")
    
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
            "Validator": "Validator",
            "Supervisor": "Supervisor"
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

def run_parser_system(
        question: str, 
        max_iterations: int, 
        model_source: str, 
        session_dir: Path, 
        log_file: Path, 
        supervisor_memory: list[dict], 
        system_metrics: SystemMetrics
    ) -> tuple[str, list[dict], SystemMetrics]:
    """Run the parser system with the given question."""
    # Create a session directory if not provided
    if session_dir is None or log_file is None:
        session_dir, log_file = create_session_directory(model_source)
        
    # Initialize the graph
    graph = build_workflow()
        
    # Start a new round
    system_metrics.start_new_round(question)
    
    print_colored(f"Memory before workflow: {len(supervisor_memory)} entries", "1;36")
    
    # Initial state
    initial_state = {
        "messages": [ HumanMessage(content=question) ],
        "user_request": question,
        "supervisor_memory": supervisor_memory,
        "generator_specs": None,
        "generator_code": None,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "model_source": model_source,
        "session_dir": session_dir,
        "log_file": log_file,
        "next_step": "Supervisor",
        "parser_mode": False,
        "system_metrics": system_metrics
    }
    
    final_response = None
    updated_memory = supervisor_memory.copy()
    
    # Run the graph
    for step in graph.stream(initial_state):
        if "__end__" in step:
            print_colored("\nWorkflow completed successfully!", "1;36")
            
            # Process final state
            final_state = step["__end__"]
            final_messages = final_state["messages"]
            updated_memory = final_state["supervisor_memory"]
            
            print_colored(f"Memory after workflow: {len(updated_memory)} entries", "1;36")
                
            # Save metrics
            if session_dir and final_state.get("system_metrics"):
                final_state["system_metrics"].save_summary(session_dir)
            
            # Get supervisor response
            supervisor_messages = [msg for msg in final_messages if hasattr(msg, 'name') and msg.name == "Supervisor"]
            
            if supervisor_messages:
                final_response = supervisor_messages[-1].content
                print_colored("\nFinal response:", "1;32")
                print(final_response)
            else:
                print_colored("\nNo final supervisor message found", "1;31")
            
            break
    
    return final_response, updated_memory, system_metrics



if __name__ == "__main__":
    # Prompt user to enter the model source directly
    source = get_model_source_from_input()
    
    print_colored("\n=== C Parser Generator System ===", "1;36")
    print_colored("You can chat with the Supervisor about C programming or request a parser", "36")
    print_colored("Enter your message or type 'exit' to quit", "36")
    print_colored(f"Session files will be saved in the 'output/{source}/multi_agent/' directory", "36")

    # Create session
    session_dir, log_file = create_session_directory(source)
    
    # Initialize memory and metrics
    supervisor_memory = []
    system_metrics = SystemMetrics()
    
    # Main interaction loop
    while True:
        print_colored("\nYou:", "1;32")
        user_input = input()
        
        # Log user input
        with open(log_file, 'a') as f:
            f.write(f"\nUser: {user_input}\n\n")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print_colored("\nExiting. Goodbye!", "1;36")
            break
        
        # Handle memory inspection command
        if user_input.lower() in ['show memory', 'memory status', 'list memory', 'memory']:
            print_colored("\n=== Current Memory Status ===", "1;34")
            if not supervisor_memory:
                print_colored("No parsers in memory yet.", "1;33")
            else:
                for i, mem in enumerate(supervisor_memory):
                    print_colored(f"Memory entry {i}:", "1;33")
                    print(f"  Iteration: {mem.get('iteration')}")

                    ass = mem.get('validator_assessment')
                    print(f"  Satisfactory: {get_satisfaction(ass)}")
                    print(f"  Compilation: {get_compilation_status(ass)}")
                    
                    # Show code excerpt
                    code_excerpt = mem.get('code', '')
                    code_excerpt = code_excerpt[:300] + "..." if len(code_excerpt) > 300 else code_excerpt
                    print(f"  Code excerpt: {code_excerpt}")
            continue
        
        try:
            # Run parser system
            response, new_memory, updated_metrics = run_parser_system(
                question=user_input,
                max_iterations=3,
                model_source=source,
                session_dir=session_dir,
                log_file=log_file,
                supervisor_memory=supervisor_memory,
                system_metrics=system_metrics
            )
            
            # Update memory if new content
            if new_memory is not None and len(new_memory) > len(supervisor_memory):
                supervisor_memory = new_memory
                print_colored(f"\nMemory updated: Now contains {len(supervisor_memory)} parsers", "1;36")
                
            # Update metrics
            system_metrics = updated_metrics
                
            # Print response if needed
            if response and not response.startswith("\033"):
                print_colored("\nSystem Response:", "1;34")
                print(response)
                
        except Exception as e:
            print_colored(f"\nAn error occurred: {e}", "1;31")
            print_colored(format_exc(), "1;31")
            print_colored("Please try again with a different query.", "1;31")
            
    # Save final metrics
    system_metrics.save_summary(session_dir)