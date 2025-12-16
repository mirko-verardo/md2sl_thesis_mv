from langchain_core.messages import AIMessage
from models import AgentState
from multi_agent import get_file_name
from utils import colors
from utils.general import print_colored, execute_c_code



def tester_node(state: AgentState) -> AgentState:
    """Tester agent that tests parser code."""
    file_format = state["file_format"]
    generator_code = state["generator_code"]
    iteration_count = state["iteration_count"]
    max_iterations = state["max_iterations"]
    session_dir = state["session_dir"]
    system_metrics = state["system_metrics"]

    # NB: here it can't be None
    if not generator_code:
        raise Exception("Something goes wrong :(")

    # Get files for testing
    file_name = get_file_name(system_metrics.get_round_number(), iteration_count)
    o_file_path_str = str(session_dir / file_name)
    format = file_format.lower()
    test_file_path = f"input/{format}/test.{format}"
        
    # Test the code
    print_colored("\n--- Parser Testing ---", colors.YELLOW, bold=True)
    testing_result = execute_c_code(o_file_path_str, test_file_path)
    
    # Check if code has been tested with success
    is_tested_ok = testing_result["success"]
    testing_status = "✅ Testing successful" if is_tested_ok else "❌ Testing failed with the following errors:\n" + testing_result["stderr"]

    with open(session_dir / f"{file_name}.txt", "w", encoding="utf-8") as f:
        test_output = f"success: {"OK" if is_tested_ok else "ERR"}\n"
        test_output += f"stdout: {testing_result["stdout"]}\n"
        test_output += f"stderr: {testing_result["stderr"]}"
        f.write(test_output)

    # Record parser testing in metrics
    system_metrics.record_parser_testing(is_tested_ok)
    
    # Log the results
    print_colored(f"Tester (Iteration {iteration_count}/{max_iterations}):", colors.BLUE, bold=True)
    print_colored(f"Testing result: {testing_status}", colors.GREEN if is_tested_ok else colors.RED, bold=True)

    # for conversation history only
    tester_response = f"Testing result: {testing_status}"
    
    return {
        "messages": [AIMessage(content=tester_response, name="Tester")],
        "user_action": state["user_action"],
        "user_request": state["user_request"],
        "file_format": file_format,
        "supervisor_specifications": state["supervisor_specifications"],
        "generator_code": generator_code,
        "compiler_result": None,
        "tester_result": testing_result,
        "code_assessment": None,
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
        "model_source": state["model_source"],
        "session_dir": session_dir,
        "next_step": "Orchestrator",
        "system_metrics": system_metrics,
        "last_parser": None
    }