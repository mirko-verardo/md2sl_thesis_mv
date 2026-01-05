from models import AgentState
from utils import colors
from utils.general import print_colored, get_parser_dir
from utils.multi_agent import is_satisfactory



def orchestrator_node(state: AgentState) -> AgentState:
    """Orchestrator agent that manages the flow."""
    messages = state["messages"]
    generator_code = state["generator_code"]
    compiler_result = state["compiler_result"]
    tester_result = state["tester_result"]
    code_assessment = state["code_assessment"]
    round = state["round"]
    iteration_count = state["iteration_count"]
    max_iterations = state["max_iterations"]
    session_dir = state["session_dir"]
    benchmark_metrics = state["benchmark_metrics"]

    # NB: here they can't be None
    if not messages:
        raise Exception("Something goes wrong :(")
    
    # Get current parser directory
    parser_dir = get_parser_dir(session_dir, round, iteration_count)
    
    prev_node = messages[-1].name
    if prev_node == "Supervisor":
        next_node = "Generator"
    elif prev_node == "Generator":
        next_node = "Compiler"
    elif prev_node == "Compiler":
        # NB: here it can't be None
        if not compiler_result:
            raise Exception("Something goes wrong :(")
        
        # Check compilation results
        if compiler_result["success"]:
            # go on with testing
            next_node = "Tester"
            benchmark_metrics.record_parser_compilation(iteration_count, parser_dir)
        else:
            # go back with error correction
            next_node = "Generator"
            code_assessment = "The parser implementation needs improvements."
            code_assessment += f"\n❌ COMPILATION failed with the following errors:\n{compiler_result["stderr"]}"
    elif prev_node == "Tester":
        # NB: here it can't be None
        if not tester_result:
            raise Exception("Something goes wrong :(")
        
        # Check testing results
        if tester_result["success"]:
            # go on with qualitative assessment
            next_node = "Assessor"
            benchmark_metrics.record_parser_testing(iteration_count, parser_dir)
        else:
            # go back with error correction
            next_node = "Generator"
            code_assessment = "The parser implementation needs improvements."
            code_assessment += f"\n✅ COMPILATION successful"
            code_assessment += f"\n❌ TESTING failed with the following errors:\n{tester_result["stderr"]}"
    elif prev_node == "Assessor":
        # NB: here it can't be None
        if not code_assessment:
            raise Exception("Something goes wrong :(")
        
        # Check if qualitative assessment is positive (bad condition)
        if is_satisfactory(code_assessment):
            next_node = "Supervisor"
            benchmark_metrics.record_parser_validation(iteration_count, parser_dir)
        else:
            next_node = "Generator"
        
        # Add compilation and testing status (NB: if here, both must be successful)
        code_assessment = f"✅ COMPILATION successful\n✅ TESTING successful\nAssessment: {code_assessment}"
    else:
        raise Exception(f"The node {prev_node} doesn't exist!")
    
    if next_node == "Generator":
        # Check if the iteration limit has been reached
        if iteration_count < max_iterations:
            # Increment interaction count
            iteration_count += 1
        else:
            # Go back to the user
            next_node = "Supervisor"
            code_assessment = f"{code_assessment}\n" if code_assessment else ""
            code_assessment += "After iterations limit, this is the best parser implementation available. While it is NOT SATISFACTORY, it could serve as a good starting point."
    
    if next_node == "Supervisor":
        benchmark_metrics.record_parser_end()

    print_colored(f"\nOrchestrator sending flow to {next_node}", colors.YELLOW, bold=True)
    
    return {
        "messages": [],
        "user_action": state["user_action"],
        "user_request": state["user_request"],
        "file_format": state["file_format"],
        "supervisor_specifications": state["supervisor_specifications"],
        "generator_code": generator_code,
        "compiler_result": compiler_result,
        "tester_result": tester_result,
        "code_assessment": code_assessment,
        "round": round,
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
        "model_source": state["model_source"],
        "session_dir": session_dir,
        "next_step": next_node,
        "benchmark_metrics": benchmark_metrics
    }