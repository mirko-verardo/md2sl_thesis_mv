from models import AgentState
from utils import colors, multi_agent
from utils.general import print_colored



def orchestrator_node(state: AgentState) -> AgentState:
    """Orchestrator agent that manages the flow."""
    messages = state["messages"]
    generator_code = state["generator_code"]
    validator_compilation = state["validator_compilation"]
    validator_testing = state["validator_testing"]
    assessor_assessment = state["assessor_assessment"]
    iteration_count = state["iteration_count"]
    max_iterations = state["max_iterations"]
    system_metrics = state["system_metrics"]

    # NB: here they can't be None
    if not messages:
        raise Exception("Something goes wrong :(")
    
    prev_node = messages[-1].name
    if prev_node == "Supervisor":
        next_node = "Generator"
    elif prev_node == "Generator":
        # Check if code has been generated (TODO: not a problem anymore)
        next_node = "Validator" if (generator_code and not multi_agent.had_agent_problems(generator_code)) else "Generator"
    elif prev_node == "Validator":
        # Check compilation and testing results
        is_compilation_ok = validator_compilation["success"]
        is_testing_ok = validator_testing["success"]
        
        if is_compilation_ok and is_testing_ok:
            # go on with a qualitative assessment
            next_node = "Assessor"
        else:
            # go back with error corrction
            next_node = "Generator"

            compilation_status = "✅ Compilation successful" if is_compilation_ok else "❌ Compilation failed with the following errors:\n" + validator_compilation["stderr"]
            testing_status = "✅ Testing successful" if is_testing_ok else "❌ Testing failed with the following errors:\n" + validator_testing["stderr"]
            
            assessor_assessment = "The parser implementation needs improvements."
            assessor_assessment += f"\nCompilation status: {compilation_status}"
            assessor_assessment += f"\nTesting status: {testing_status}"
    elif prev_node == "Assessor":
        # Check if qualitative assessment is positive
        is_satisfactory = multi_agent.is_satisfactory(assessor_assessment)
        next_node = "Supervisor" if is_satisfactory else "Generator"
        # Add compilation and testing status (NB: if here, both must be successful)
        assessor_assessment = f"Assessment: {assessor_assessment}" if assessor_assessment else ""
        assessor_assessment = f"Compilation status: ✅ Compilation successful\nTesting status: ✅ Testing successful\n{assessor_assessment}"
    else:
        raise Exception(f"The node {prev_node} doesn't exist!")
    
    if next_node == "Generator":
        # Increment interaction count
        iteration_count += 1
        system_metrics.increment_generator_validator_interaction()
    
    if iteration_count > max_iterations:
        next_node = "Supervisor"
        assessor_assessment = f"{assessor_assessment}\n\n" if assessor_assessment else ""
        assessor_assessment += "After iterations limit, this is the best parser implementation available. While it is NOT SATISFACTORY, it could serve as a good starting point."
    
    print_colored(f"\nOrchestrator sending flow to {next_node}", colors.YELLOW, bold=True)
    
    return {
        "messages": [],
        "user_action": state["user_action"],
        "user_request": state["user_request"],
        "file_format": state["file_format"],
        "supervisor_specifications": state["supervisor_specifications"],
        "generator_code": generator_code,
        "validator_compilation": validator_compilation,
        "validator_testing": validator_testing,
        "assessor_assessment": assessor_assessment,
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
        "model_source": state["model_source"],
        "session_dir": state["session_dir"],
        "next_step": next_node,
        "system_metrics": system_metrics
    }