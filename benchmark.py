from csv import DictWriter
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from models import BenchmarkMetrics
from utils.general import create_session
from utils.graph import build_workflow
from utils.multi_agent import get_request_from_action
from utils.single_agent import start_chat



if __name__ == "__main__":
    # Initialize the graph
    graph = build_workflow()
    config = RunnableConfig(recursion_limit=100)

    # Initialize parameters
    user_action = "GENERATE_PARSER"
    #types = [ "multi_agent", "zero_shot" ]
    #sources = [ "google", "openai", "anthropic" ]
    formats = [ "CSV", "HTML", "HTTP", "JSON", "PDF", "XML" ]
    reps = range(10)
    types = [ "multi_agent" ]
    sources = [ "google" ]
    #formats = [ "CSV", "HTML", "JSON" ]
    #reps = range(2)
    benchmarks = []
    attempts = 15

    for rep in reps:
        for type in types:
            for source in sources:
                for format in formats:
                    if type == "zero_shot":
                        # Log benchmark
                        benchmark_metrics = start_chat(source, format, n=rep, react_loops=attempts, exit_at_first=True)
                    else:
                        # Initialize parameters
                        session_dir = create_session(source, type, format)
                        user_request = get_request_from_action(user_action, format)
                        user_message = f"{user_action}: {user_request}"
                        benchmark_metrics = BenchmarkMetrics(rep, type, format)
                        
                        # Initial state
                        initial_state = {
                            "messages": [ HumanMessage(content=user_message) ],
                            "user_action": user_action,
                            "user_request": user_request,
                            "file_format": format,
                            "supervisor_specifications": None,
                            "generator_code": None,
                            "compiler_result": None,
                            "tester_result": None,
                            "code_assessment": None,
                            "round": 1,
                            "iteration_count": 0,
                            "max_iterations": attempts,
                            "model_source": source,
                            "session_dir": session_dir,
                            "next_step": "Supervisor",
                            "benchmark_metrics": benchmark_metrics,
                            "last_parser": None
                        }

                        result = graph.invoke(initial_state, config)

                        # Log conversation
                        with open(session_dir / "conversation.txt", "w", encoding="utf-8") as f:
                            for m in result["messages"]:
                                f.write(f"{m.pretty_repr()}\n\n")

                        # Log benchmark
                        benchmark_metrics = result["benchmark_metrics"]
                    
                    # Save benchmark
                    benchmarks.append(benchmark_metrics.get_benchmark())
    
    # Log benchmark
    with open("benchmark.csv", "w", encoding="utf-8", newline="") as f:
        writer = DictWriter(f, fieldnames=benchmarks[0].keys())
        writer.writeheader()
        writer.writerows(benchmarks)
    print(benchmarks)