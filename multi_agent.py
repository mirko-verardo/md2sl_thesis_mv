from csv import DictWriter
from traceback import format_exc
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from models import BenchmarkMetrics
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
    session_dir = create_session(source, type, file_format)
    messages = []
    benchmarks = []
    round = 1
    attempts = 10
    last_parser = None
    
    # Main interaction loop
    while True:
        # Initialize parameters
        user_request = get_request_from_action(user_action, file_format)
        if user_request is None:
            break
        user_message = f"{user_action}: {user_request}"
        benchmark_metrics = BenchmarkMetrics(round, type, file_format)
        
        try:
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
                "round": round,
                "iteration_count": 0,
                "max_iterations": attempts,
                "model_source": source,
                "session_dir": session_dir,
                "next_step": "Supervisor",
                "benchmark_metrics": benchmark_metrics,
                "last_parser": last_parser
            }

            result = graph.invoke(initial_state, config)

            # Save conversation
            messages += result["messages"]

            # Save benchmark
            benchmarks.append(result["benchmark_metrics"].get_benchmark())
            
            # Save last parser
            last_parser = result["last_parser"]
        except Exception as e:
            print_colored(f"\nAn error occurred: {e}", colors.RED, bold=True)
            print_colored(format_exc(), colors.RED, bold=True)
            print_colored("Please try again with a different query.", colors.RED, bold=True)
        
        # Ask the user again
        user_action = get_action_from_input()
        round += 1
    
    # Log conversation
    with open(session_dir / "conversation.txt", "w", encoding="utf-8") as f:
        for m in messages:
            f.write(f"{m.pretty_repr()}\n\n")
    
    # Log benchmark
    with open(session_dir / "benchmark.csv", "w", encoding="utf-8", newline="") as f:
        writer = DictWriter(f, fieldnames=benchmarks[0].keys())
        writer.writeheader()
        writer.writerows(benchmarks)