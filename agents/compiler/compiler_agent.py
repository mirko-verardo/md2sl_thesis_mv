from langchain_core.messages import AIMessage
from models import AgentState
from utils import colors
from utils.general import print_colored, compile_c_code



def compiler_node(state: AgentState) -> AgentState:
    """Compiler agent that compiles parser code."""
    file_format = state["file_format"]
    generator_code = state["generator_code"]
    iteration_count = state["iteration_count"]
    max_iterations = state["max_iterations"]
    session_dir = state["session_dir"]
    system_metrics = state["system_metrics"]

    # NB: here it can't be None
    if not generator_code:
        raise Exception("Something goes wrong :(")

    # Save c code to temporary file for compilation
    c_file_name = f"parser_{iteration_count}.c"
    c_file_path = session_dir / c_file_name
    c_file_path_str = str(c_file_path)
    o_file_path_str = str(c_file_path.with_suffix(''))
    
    with open(c_file_path, "w", encoding="utf-8") as f:
        f.write(generator_code)
    
    print_colored(f"\nSaved C code to: {c_file_path_str} for compilation", colors.CYAN, bold=True)
    
    # Compile the code
    print_colored("\n--- Parser Compilation ---", colors.YELLOW, bold=True)
    compilation_result = compile_c_code(c_file_path_str, o_file_path_str)
    
    # Check if code has been compiled with success
    is_compiled = compilation_result["success"]
    compilation_status = "✅ Compilation successful" if is_compiled else f"❌ Compilation failed with the following errors:\n{compilation_result["stderr"]}"

    # Record parser compilation in metrics
    system_metrics.record_parser_compilation(c_file_name, is_compiled)
    
    # Log the results
    print_colored(f"Compiler (Iteration {iteration_count}/{max_iterations}):", colors.BLUE, bold=True)
    print_colored(f"Compilation result: {compilation_status}", colors.GREEN if is_compiled else colors.RED, bold=True)

    # for conversation history only
    compiler_response = f"Compilation result: {compilation_status}"
    
    return {
        "messages": [AIMessage(content=compiler_response, name="Compiler")],
        "user_action": state["user_action"],
        "user_request": state["user_request"],
        "file_format": file_format,
        "supervisor_specifications": state["supervisor_specifications"],
        "generator_code": generator_code,
        "compiler_result": compilation_result,
        "tester_result": None,
        "code_assessment": None,
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
        "model_source": state["model_source"],
        "session_dir": session_dir,
        "next_step": "Orchestrator",
        "system_metrics": system_metrics,
        "last_parser": None
    }