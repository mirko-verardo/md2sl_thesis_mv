from langchain_core.messages import AIMessage
from models import AgentState
from utils import colors
from utils.general import print_colored, compile_c_code, get_parser_dir



def compiler_node(state: AgentState) -> AgentState:
    """Compiler agent that compiles parser code."""
    file_format = state["file_format"]
    generator_code = state["generator_code"]
    round = state["round"]
    iteration_count = state["iteration_count"]
    max_iterations = state["max_iterations"]
    session_dir = state["session_dir"]

    # NB: here it can't be None
    if not generator_code:
        raise Exception("Something goes wrong :(")

    # Create the parser dir
    parser_dir = get_parser_dir(session_dir, round, iteration_count)
    parser_dir.mkdir()
    
    # Compile the code
    print_colored("\n--- Parser Compilation ---", colors.YELLOW, bold=True)
    compilation_result = compile_c_code(parser_dir, generator_code, runtime=False)
    
    # Check if code has been compiled with success
    is_compiled = compilation_result["success"]
    compilation_flags = "buildtime"
    if is_compiled:
        # Recompile the code for the tester (runtime flags)
        compilation_result = compile_c_code(parser_dir, generator_code)
        is_compiled = compilation_result["success"]
        compilation_flags = "runtime"
    
    compilation_status = "✅ Compilation successful" if is_compiled else f"❌ Compilation failed with the following errors:\n{compilation_result["stderr"]}"
    
    # Log the results
    print_colored(f"Compiler (Iteration {iteration_count}/{max_iterations}):", colors.BLUE, bold=True)
    print_colored(f"Compilation flags: {compilation_flags}", colors.BLUE, bold=True)
    print_colored(f"Compilation result: {compilation_status}", colors.GREEN if is_compiled else colors.RED, bold=True)

    # for conversation history only
    compiler_response = f"Compilation result ({compilation_flags}): {compilation_status}"
    
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
        "round": round,
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
        "model_source": state["model_source"],
        "session_dir": session_dir,
        "next_step": "Orchestrator",
        "benchmark_metrics": state["benchmark_metrics"]
    }