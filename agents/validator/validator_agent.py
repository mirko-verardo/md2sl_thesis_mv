from datetime import datetime
from pathlib import Path
from langchain_core.messages import AIMessage
from models import AgentState
from utils import colors
from utils.general import print_colored, compile_c_code, execute_c_code



def validator_node(state: AgentState) -> AgentState:
    """Validator agent that compiles and test parser code."""
    file_format = state["file_format"]
    generator_code = state["generator_code"]
    iteration_count = state["iteration_count"]
    max_iterations = state["max_iterations"]
    session_dir = state["session_dir"]
    system_metrics = state["system_metrics"]

    # Save c code to temporary file for compilation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    c_file_name = f"parser_{timestamp}.c"
    c_file_path = session_dir / c_file_name
    o_file_path = c_file_path.with_suffix('')
    
    with open(c_file_path, "w", encoding="utf-8") as f:
        f.write(generator_code)
    
    print_colored(f"\nSaved C code to: {c_file_path} for compilation testing", colors.CYAN, bold=True)
    
    # Compile the code
    print_colored("\n--- Parser Compilation ---", colors.YELLOW, bold=True)
    compilation_result = compile_c_code(str(c_file_path), str(o_file_path))
    # Check if code has been compiled with success
    is_compiled = compilation_result["success"]
    compilation_status = "✅ Compilation successful" if is_compiled else "❌ Compilation failed with the following errors:\n" + compilation_result["stderr"]
    
    if is_compiled:
        # Test the code
        print_colored("\n--- Parser Testing ---", colors.YELLOW, bold=True)
        base_dir = Path("input")
        format = file_format.lower()
        test_file_name = "test." + format
        test_file_path = base_dir / format / test_file_name
        testing_result = execute_c_code(str(o_file_path), str(test_file_path))
        # Check if code has been executed with success
        is_tested = testing_result["success"]
        testing_status = "✅ Testing successful" if is_tested else "❌ Testing failed with the following errors:\n" + testing_result["stderr"]

        with open(session_dir / "test.txt", "w", encoding="utf-8") as f:
            test_output = f"success: {"OK" if is_tested else "ERR"}\n"
            test_output += f"stdout: {testing_result["stdout"]}\n"
            test_output += f"stderr: {testing_result["stderr"]}"
            f.write(test_output)
    else:
        testing_result = {
            "success": False,
            "stdout": "",
            "stderr": "Not even compiled"
        }
        is_tested = False
        testing_status = "❌ Test execution failed because the generated code doesn't even compile"

    # Record parser validation in metrics
    system_metrics.record_parser_validation(c_file_name, is_compiled, is_tested)
    
    # Log the results
    print_colored(f"Validator (Iteration {iteration_count}/{max_iterations}):", colors.BLUE, bold=True)
    print_colored(f"Compilation result: {compilation_status}", colors.GREEN if is_compiled else colors.RED, bold=True)
    print_colored(f"Testing result: {testing_status}", colors.GREEN if is_tested else colors.RED, bold=True)
    
    return {
        "messages": [AIMessage(content="", name="Validator")],
        "user_action": state["user_action"],
        "user_request": state["user_request"],
        "file_format": file_format,
        "supervisor_specifications": state["supervisor_specifications"],
        "generator_code": generator_code,
        "validator_compilation": compilation_result,
        "validator_testing": testing_result,
        "assessor_assessment": None,
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
        "model_source": state["model_source"],
        "session_dir": session_dir,
        "next_step": "Orchestrator",
        "system_metrics": system_metrics
    }