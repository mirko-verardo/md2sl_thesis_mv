from datetime import datetime
from pathlib import Path
from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from models import AgentState
from agents.validator import validator_prompts
from utils import colors, multi_agent
from utils.general import print_colored, compile_c_code, execute_c_code, initialize_llm



def validator_node(state: AgentState) -> AgentState:
    """Validator agent that evaluates parser code."""
    generator_specs = state["generator_specs"]
    generator_code = state["generator_code"]
    iteration_count = state["iteration_count"]
    max_iterations = state["max_iterations"]
    model_source = state["model_source"]
    session_dir = state["session_dir"]
    system_metrics = state["system_metrics"]

    # Record generator validator interaction numbers
    system_metrics.increment_generator_validator_interaction()
    
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
        is_compiled = compilation_result['success']
        # Get compilation status
        compilation_status = "✅ Compilation successful!" if is_compiled else "❌ Compilation failed with the following errors:\n" + compilation_result['stderr']
        
        # Test the code
        if is_compiled:
            print_colored("\n--- Testing Parser ---", colors.YELLOW, bold=True)
            base_dir = Path("input")
            format = "json"
            test_file_name = "test." + format
            test_file_path = base_dir / format / test_file_name
            test_result = execute_c_code(str(o_file_path), str(test_file_path))
            test_output = """success: """ + ("OK" if test_result["success"] else "ERR") + """
stdout: """ + test_result["stdout"] + """
stderr: """ + test_result["stderr"] + """
"""
            with open(session_dir / "test.txt", "w", encoding="utf-8") as f:
                f.write(test_output)

        # Record parser validation in metrics
        system_metrics.record_parser_validation(c_file_name, is_compiled)

        # Create the prompt
        validator_template = validator_prompts.get_validator_template()
        validator_template = validator_template.replace("{specifications}", validator_prompts.get_specifications_template() if generator_specs else "")
        validator_input = {
            "requirements": multi_agent.get_parser_requirements(),
            "code": generator_code,
            "compilation_status": compilation_status
        }
        if generator_specs:
            validator_input.update({
                "supervisor_specifications": generator_specs
            })
        validator_prompt = PromptTemplate.from_template(validator_template)

        # Initialize model for validator
        validator_llm = initialize_llm(model_source)
        validator_llm.temperature = 0.4
        validator_tools = []
        
        # Create the ReAct agent instead of OpenAI tools agent
        validator_agent = create_react_agent(validator_llm, validator_tools, validator_prompt)
        
        validator_executor = AgentExecutor(
            agent=validator_agent,
            tools=validator_tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            early_stopping_method="force"
        )
        
        # Invoke the agent
        try:
            validator_result = validator_executor.invoke(validator_input)
            validator_response = str(validator_result["output"])
            validator_response_color = colors.BLUE
        except Exception as e:
            validator_response = f"Error occurred during code validation: {str(e)}\n\nPlease try again."
            validator_response_color = colors.RED
    else:
        is_compiled = False
        compilation_status = "❌ Compilation failed!"
        validator_response = "Retry with a simpler and shorter version"
        validator_response_color = colors.YELLOW

    # Check if code is satisfactory
    is_satisfactory = is_compiled and multi_agent.get_satisfaction(validator_response) == "SATISFACTORY"
    # Check if code is the final iteration
    is_final_iteration = iteration_count >= max_iterations

    if is_satisfactory:
        # end
        next_node = "Supervisor"
        feedback_message = "The parser implementation has been validated and meets all requirements.\n"
        feedback_message += f"Compilation status: {compilation_status}\n"
        feedback_message += f"Final assessment:\n{validator_response}"
    elif is_final_iteration:
        # end
        next_node = "Supervisor"
        feedback_message = "After iterations limit, this is the best parser implementation available. While it may not be perfect, it could serve as a good starting point.\n"
        feedback_message += f"Compilation status: {compilation_status}\n"
        feedback_message += f"Final assessment:\n{validator_response}"
    elif is_compiled or is_code_generated:
        # continue
        next_node = "Generator"
        feedback_message = "The parser implementation needs improvements.\n"
        feedback_message += f"Compilation status: {compilation_status}\n"
        feedback_message += f"Final assessment:\n{validator_response}"
    else:
        # continue
        next_node = "Generator"
        feedback_message = validator_response
    
    # Log compilation results
    compilation_status_color = colors.GREEN if is_compiled else colors.RED
    print_colored(f"Compilation Test Results (Iteration {iteration_count}/{max_iterations}): {compilation_status}", compilation_status_color, bold=True)
    # Log the validator's assessment
    print_colored(f"Validator assessment (Iteration {iteration_count}/{max_iterations}):", validator_response_color, bold=True)
    print(validator_response)
    print_colored(f"\nValidator sending FEEDBACK to {next_node}", colors.YELLOW, bold=True)
    
    return {
        "messages": [AIMessage(content=feedback_message, name="Validator")],
        "user_action": state["user_action"],
        "user_request": state["user_request"],
        "generator_specs": generator_specs,
        "generator_code": generator_code,
        "validator_assessment": feedback_message,
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
        "model_source": state["model_source"],
        "session_dir": session_dir,
        "next_step": next_node,
        "system_metrics": system_metrics
    }