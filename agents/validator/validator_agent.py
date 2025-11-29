from datetime import datetime
from pathlib import Path
from langchain_core.messages import AIMessage
from models import AgentState
from utils import colors, multi_agent
from utils.general import print_colored, compile_c_code, execute_c_code



def validator_node(state: AgentState) -> AgentState:
    """Validator agent that evaluates parser code."""
    file_format = state["file_format"]
    generator_code = state["generator_code"]
    iteration_count = state["iteration_count"]
    max_iterations = state["max_iterations"]
    session_dir = state["session_dir"]
    system_metrics = state["system_metrics"]

    # Check if code has been generated
    is_code_generated = generator_code and not multi_agent.had_agent_problems(generator_code)

    if is_code_generated:
        # Save c code to temporary file for compilation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        c_file_name = f"parser_{timestamp}.c"
        c_file_path = session_dir / c_file_name
        o_file_path = c_file_path.with_suffix('')
        
        with open(c_file_path, "w", encoding="utf-8") as f:
            f.write(generator_code)
        
        print_colored(f"\nSaved C code to: {c_file_path} for compilation testing", colors.CYAN, bold=True)
        
        # Compile the code
        print_colored("\n--- Testing Compilation ---", colors.YELLOW, bold=True)
        compilation_result = compile_c_code(str(c_file_path), str(o_file_path))
        # Check if code has been compiled with success
        is_compiled = compilation_result["success"]
        # Get compilation status
        compilation_status = "✅ Compilation successful" if is_compiled else "❌ Compilation failed with the following errors:\n" + compilation_result["stderr"]
        
        # Test the code
        is_tested = False
        execution_status = "❌ Test execution failed because the generated code doesn't even compile"
        if is_compiled:
            print_colored("\n--- Testing Parser ---", colors.YELLOW, bold=True)
            base_dir = Path("input")
            format = file_format.lower()
            test_file_name = "test." + format
            test_file_path = base_dir / format / test_file_name
            test_result = execute_c_code(str(o_file_path), str(test_file_path))
            # Check if code has been executed with success
            is_tested = test_result["success"]
            # Get execution status
            execution_status = "✅ Test execution successful" if is_tested else "❌ Test execution failed with the following errors:\n" + test_result["stderr"]

            with open(session_dir / "test.txt", "w", encoding="utf-8") as f:
                test_output = """success: """ + ("OK" if is_tested else "ERR") + """
stdout: """ + test_result["stdout"] + """
stderr: """ + test_result["stderr"] + """
"""
                f.write(test_output)

        # Record parser validation in metrics
        system_metrics.record_parser_validation(c_file_name, is_compiled, is_tested)
    else:
        is_compiled = False
        is_tested = False
        compilation_status = "❌ Compilation failed"
        execution_status = "❌ Test execution failed"

    # Check if code is satisfactory
    is_satisfactory = is_compiled and is_tested
    # Check if code is the final iteration
    is_final_iteration = iteration_count >= max_iterations

    if is_satisfactory:
        # end
        next_node = "Supervisor"
        feedback_message = "The parser implementation has been validated and it is SATISFACTORY."
    elif is_final_iteration:
        # end
        next_node = "Supervisor"
        feedback_message = "After iterations limit, this is the best parser implementation available. While it is NOT SATISFACTORY, it could serve as a good starting point."
    elif is_code_generated:
        # continue
        next_node = "Generator"
        feedback_message = "The parser implementation needs improvements."
    else:
        # continue
        next_node = "Generator"
        feedback_message = generator_code
    
    feedback_message += f"\nCompilation status: {compilation_status}"
    feedback_message += f"\nExecution status: {execution_status}"
    
    # Log the validator's assessment
    print_colored(f"Validator (Iteration {iteration_count}/{max_iterations}):", colors.BLUE, bold=True)
    print_colored(f"Compilation result: {compilation_status}", colors.GREEN if is_compiled else colors.RED, bold=True)
    print_colored(f"Execution result: {execution_status}", colors.GREEN if is_tested else colors.RED, bold=True)
    print_colored(f"\nValidator sending FEEDBACK to {next_node}", colors.YELLOW, bold=True)
    
    return {
        "messages": [AIMessage(content=feedback_message, name="Validator")],
        "user_action": state["user_action"],
        "user_request": state["user_request"],
        "file_format": file_format,
        "generator_specs": state["generator_specs"],
        "generator_code": generator_code,
        "validator_assessment": feedback_message,
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
        "model_source": state["model_source"],
        "session_dir": session_dir,
        "next_step": next_node,
        "system_metrics": system_metrics
    }