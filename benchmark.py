from csv import DictWriter
from traceback import format_exc
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from models import SystemMetrics, BenchmarkMetrics
from utils.general import create_session
from utils.graph import build_workflow
from utils.multi_agent import get_request_from_action



if __name__ == "__main__":
    user_action = "GENERATE_PARSER"

    #types = [ "multi_agent", "zero_shot" ]
    #sources = [ "google", "openai", "anthropic" ]
    #formats = [ "CSV", "HTML", "HTTP", "JSON", "XML" ]
    #reps = range(10)
    types = [ "multi_agent" ]
    sources = [ "google" ]
    formats = [ "CSV", "HTML", "HTTP", "JSON", "XML" ]
    reps = range(2)

    # Initialize the graph
    graph = build_workflow()
    config = RunnableConfig(recursion_limit=100)
    benchmark_data = []

    for n in reps:
        for type in types:
            for source in sources:
                for format in formats:
                    # Initialize parameters
                    system_metrics = SystemMetrics()
                    benchmark_metrics = BenchmarkMetrics(n, type, format)
                    session_dir, log_file = create_session(source, type, format)
                    user_request = get_request_from_action(user_action, format)
                    user_message = f"{user_action}: {user_request}"
                    
                    try:
                        # Start a new round
                        system_metrics.start_new_round(user_message)
                        
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
                            "iteration_count": 0,
                            "max_iterations": 10,
                            "model_source": source,
                            "session_dir": session_dir,
                            "next_step": "Supervisor",
                            "system_metrics": system_metrics,
                            "benchmark_metrics": benchmark_metrics,
                            "last_parser": None
                        }

                        result = graph.invoke(initial_state, config)

                        # Log conversation
                        with open(log_file, "w", encoding="utf-8") as f:
                            for m in result["messages"]:
                                f.write(f"{m.pretty_repr()}\n\n")

                        # Save metrics
                        system_metrics = result["system_metrics"]
                        system_metrics.complete_round()
                        system_metrics.save_summary(session_dir)

                        # Save benchmark
                        bm = result["benchmark_metrics"].get_benchmark()
                        benchmark_data.append(bm)
                        print(bm)
                    except Exception as e:
                        print(f"\nAn error occurred: {e}")
                        print(format_exc())
    
    with open("benchmark.csv", "w", encoding="utf-8") as f:
        writer = DictWriter(f, fieldnames=benchmark_data[0].keys())
        writer.writeheader()
        writer.writerows(benchmark_data)