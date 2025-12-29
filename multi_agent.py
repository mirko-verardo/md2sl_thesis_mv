from traceback import format_exc
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from models import SystemMetrics, BenchmarkMetrics
from utils import colors
from utils.general import create_session, get_model_source_from_input, get_file_format_from_input, print_colored
from utils.graph import build_workflow
from utils.multi_agent import get_action_from_input, get_request_from_action



if __name__ == "__main__":
    print_colored("\n=== C Parser Generator System ===", colors.CYAN, bold=True)

    # Initialize the graph
    graph = build_workflow()
    config = RunnableConfig(recursion_limit=100)

    # Initialize parameters
    user_action = "GENERATE_PARSER"
    type = "multi_agent"
    source = get_model_source_from_input()
    file_format = get_file_format_from_input()
    system_metrics = SystemMetrics()
    benchmark_metrics = BenchmarkMetrics(1, type, file_format)
    session_dir, log_file = create_session(source, type, file_format)
    messages = []
    last_parser = None
    
    # Main interaction loop
    while True:
        user_request = get_request_from_action(user_action, file_format)
        if user_request is None:
            break
        user_message = f"{user_action}: {user_request}"
        
        try:
            # Start a new round
            system_metrics.start_new_round(user_message)
            
            # Initial state
            initial_state = {
                "messages": [ HumanMessage(content=user_message) ],
                "user_action": user_action,
                "user_request": user_request,
                "file_format": file_format,
                "supervisor_specifications": None,
                "generator_code": None,
                "compiler_result": None,
                "tester_result": None,
                "code_assessment": None,
                "iteration_count": 0,
                "max_iterations": 10,
                "model_source": source,
                "session_dir": session_dir,
                "next_step": "Supervisor",
                "system_metrics": system_metrics,
                "benchmark_metrics": benchmark_metrics,
                "last_parser": last_parser
            }

            result = graph.invoke(initial_state, config)

            # Save conversation
            messages += result["messages"]

            # Save metrics
            system_metrics = result["system_metrics"]
            system_metrics.complete_round()
            system_metrics.save_summary(session_dir)

            print(result["benchmark_metrics"].get_benchmark())
            
            # Get last parser
            last_parser = result["last_parser"]
        except Exception as e:
            print_colored(f"\nAn error occurred: {e}", colors.RED, bold=True)
            print_colored(format_exc(), colors.RED, bold=True)
            print_colored("Please try again with a different query.", colors.RED, bold=True)
        
        # Ask the user again
        user_action = get_action_from_input()
    
    # Log conversation
    with open(log_file, "w", encoding="utf-8") as f:
        for m in messages:
            #m.pretty_print()
            f.write(f"{m.pretty_repr()}\n\n")