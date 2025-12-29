from models import AgentState
from utils import colors
from utils.general import print_colored
from utils.multi_agent import get_parser_dir, is_satisfactory



def orchestrator_node(state: AgentState) -> AgentState:
    """Orchestrator agent that manages the flow."""
    messages = state["messages"]
    generator_code = state["generator_code"]
    compiler_result = state["compiler_result"]
    tester_result = state["tester_result"]
    code_assessment = state["code_assessment"]
    iteration_count = state["iteration_count"]
    max_iterations = state["max_iterations"]
    session_dir = state["session_dir"]
    system_metrics = state["system_metrics"]
    benchmark_metrics = state["benchmark_metrics"]
    last_parser = state["last_parser"]

    # NB: here they can't be None
    if not messages:
        raise Exception("Something goes wrong :(")
    
    # Get current parser directory
    parser_dir_str = str(get_parser_dir(session_dir, system_metrics.get_round_number(), iteration_count))
    
    prev_node = messages[-1].name
    if prev_node == "Supervisor":
        next_node = "Generator"
        if last_parser:
            generator_code = last_parser["code"]
            code_assessment = last_parser["assessment"]
    elif prev_node == "Generator":
        next_node = "Compiler"
    elif prev_node == "Compiler":
        # NB: here it can't be None
        if not compiler_result:
            raise Exception("Something goes wrong :(")
        
        # Check compilation results
        compiler_result_success = compiler_result["success"]
        system_metrics.record_parser_compilation(parser_dir_str, compiler_result_success)
        if compiler_result_success:
            # go on with testing
            next_node = "Tester"
            benchmark_metrics.record_parser_compilation(iteration_count, parser_dir_str)
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
        tester_result_success = tester_result["success"]
        system_metrics.record_parser_testing(tester_result_success)
        if tester_result_success:
            # go on with qualitative assessment
            next_node = "Assessor"
            benchmark_metrics.record_parser_testing(iteration_count, parser_dir_str)
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
            system_metrics.satisfy_round()
            benchmark_metrics.record_parser_validation(iteration_count, parser_dir_str)
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
            system_metrics.increment_generator_interaction()
        else:
            # Go back to the user
            next_node = "Supervisor"
            code_assessment = f"{code_assessment}\n" if code_assessment else ""
            code_assessment += "After iterations limit, this is the best parser implementation available. While it is NOT SATISFACTORY, it could serve as a good starting point."
    
    # NB: always updated
    last_parser = {
        "code": generator_code,
        "assessment": code_assessment
    } if next_node == "Supervisor" else None

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
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
        "model_source": state["model_source"],
        "session_dir": session_dir,
        "next_step": next_node,
        "system_metrics": system_metrics,
        "benchmark_metrics": benchmark_metrics,
        "last_parser": last_parser
    }