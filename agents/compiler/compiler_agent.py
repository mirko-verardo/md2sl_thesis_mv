from lizard import analyze_file
from langchain_core.messages import AIMessage
from models import AgentState
from utils import colors
from utils.general import print_colored, compile_c_code
from utils.multi_agent import get_file_name
from time import sleep



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

    # Save C code to file for compilation
    file_name = get_file_name(system_metrics.get_round_number(), iteration_count)
    c_file_name = f"{file_name}.c"
    parser_dir = session_dir / file_name
    c_file_path = parser_dir / c_file_name
    c_file_path_str = str(c_file_path)
    o_file_path_str = str(parser_dir / file_name)
    
    parser_dir.mkdir()
    with open(c_file_path, "w", encoding="utf-8") as f:
        f.write(generator_code)
    
    print_colored(f"\nSaved C code to: {c_file_path_str} for compilation", colors.CYAN, bold=True)

    # Metric the code
    i = analyze_file(c_file_path_str)
    print(i.__dict__)
    print(i.function_list[0].__dict__)
    sleep(10)
    
    # Compile the code
    print_colored("\n--- Parser Compilation ---", colors.YELLOW, bold=True)
    compilation_result = compile_c_code(c_file_path_str, o_file_path_str)
    
    # Check if code has been compiled with success
    is_compiled = compilation_result["success"]
    compilation_flags = "buildtime"
    if is_compiled:
        # Recompile the code for the tester (runtime flags)
        compilation_result = compile_c_code(c_file_path_str, o_file_path_str, runtime=True)
        is_compiled = compilation_result["success"]
        compilation_flags = "runtime"
    
    compilation_status = "✅ Compilation successful" if is_compiled else f"❌ Compilation failed with the following errors:\n{compilation_result["stderr"]}"

    # Record parser compilation in metrics
    system_metrics.record_parser_compilation(c_file_name, is_compiled)
    
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
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
        "model_source": state["model_source"],
        "session_dir": session_dir,
        "next_step": "Orchestrator",
        "system_metrics": system_metrics,
        "last_parser": None
    }