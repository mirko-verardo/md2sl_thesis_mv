from csv import DictWriter
from pathlib import Path
from traceback import format_exc
from langchain_core.runnables import RunnableConfig
from models import BenchmarkMetrics
from utils.general import create_session
from utils.graph import build_workflow, start_workflow
from utils.multi_agent import get_request_from_action
from utils.single_agent import start_chat



if __name__ == "__main__":
    benchmarks_file = Path("benchmark") / "benchmark.csv"

    # Initialize the graph
    graph = build_workflow()
    config = RunnableConfig(recursion_limit=100)

    # Initialize parameters
    user_action = "GENERATE_PARSER"
    reps = range(10)
    types = [ "multi_agent", "zero_shot" ]
    formats = [ "CSV", "HTML", "HTTP", "JSON", "PDF", "XML" ]
    #sources = [ "google", "openai", "anthropic" ]
    #reps = range(2)
    #formats = [ "CSV", "JSON" ]
    sources = [ "anthropic" ]
    attempts = 15
    benchmarks = []

    for rep in reps:
        for type in types:
            for format in formats:
                for source in sources:
                    if type == "zero_shot":
                        # Log benchmark
                        benchmark_metrics = start_chat(source, format, n=rep, react_loops=attempts, exit_at_first=True)
                    else:
                        # Initialize parameters
                        session_dir = create_session(source, type, format)
                        conversation_file = session_dir / "conversation.txt"
                        user_request = get_request_from_action(user_action, format)
                        benchmark_metrics = BenchmarkMetrics(rep, type, format, source)
                        
                        try:
                            # Get workflow result
                            result = start_workflow(graph, config, user_action, user_request, format, 1, attempts, source, session_dir, benchmark_metrics)

                            # Log conversation
                            with open(conversation_file, "w", encoding="utf-8") as f:
                                for m in result["messages"]:
                                    f.write(f"{m.pretty_repr()}\n\n")

                            # Log benchmark
                            benchmark_metrics = result["benchmark_metrics"]
                        except Exception as e:
                            # Log error
                            with open(conversation_file, "w", encoding="utf-8") as f:
                                f.write(f"An error occurred: {e}\n\n")
                                f.write(format_exc())
                    
                    # Save benchmark
                    benchmarks.append(benchmark_metrics.get_benchmark())
    
    # Log benchmark
    with open(benchmarks_file, "w", encoding="utf-8", newline="") as f:
        writer = DictWriter(f, fieldnames=benchmarks[0].keys())
        writer.writeheader()
        writer.writerows(benchmarks)
    print(benchmarks)