from csv import DictWriter
from traceback import format_exc
from langchain_core.runnables import RunnableConfig
from models import BenchmarkMetrics
from utils import colors
from utils.general import create_session, get_model_source_from_input, get_file_format_from_input, print_colored
from utils.graph import build_workflow, start_workflow
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
    round = 1
    attempts = 10
    last_parser = None
    messages = []
    benchmarks = []
    
    # Main interaction loop
    while True:
        # Initialize parameters
        user_request = get_request_from_action(user_action, file_format)
        if user_request is None:
            break
        benchmark_metrics = BenchmarkMetrics(round, type, file_format)
        
        try:
            # Get workflow result
            result = start_workflow(graph, config, user_action, user_request, file_format, round, attempts, source, session_dir, benchmark_metrics, last_parser)

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