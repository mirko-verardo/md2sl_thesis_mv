from datetime import datetime
from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from models import AgentState
from agents.validator import validator_prompts
from utils import colors, multi_agent
from utils.general import print_colored, log, compile_c_code, initialize_llm



def validator_node(state: AgentState) -> AgentState:
    """Validator agent that evaluates parser code."""
    generator_specs = state["generator_specs"]
    generator_code = state["generator_code"]
    iteration_count = state["iteration_count"]
    max_iterations = state["max_iterations"]
    model_source = state["model_source"]
    session_dir = state["session_dir"]
    log_file = state["log_file"]
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
        
        with open(c_file_path, 'w', encoding="utf-8") as ff:
            ff.write(generator_code)
        
        print_colored(f"\nSaved C code to: {c_file_path} for compilation testing", "1;36")
        
        # Compile the code
        print_colored("\n--- Testing Compilation ---", "1;33")
        compilation_result = compile_c_code(c_file_path)
        # Check if code has been compiled with success
        is_compiled = compilation_result['success']
        # Get compilation status
        compilation_status = "✅ Compilation successful!" if is_compiled else "❌ Compilation failed with the following errors:\n" + compilation_result['stderr']
        
        # Record parser validation in metrics
        system_metrics.record_parser_validation(c_file_name, is_compiled)

        # Manage prompt's input
        validator_input = {
            "requirements": multi_agent.get_parser_requirements(),
            "specifications": "",
            "code": generator_code,
            "compilation_status": compilation_status
        }
        if generator_specs:
            validator_input.update({
                "specifications": validator_prompts.get_specifications_template(),
                "supervisor_specifications": generator_specs
            })
        
        # Create the prompt
        validator_template = validator_prompts.get_validator_template()
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
            handle_parsing_errors=True
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
    
    compilation_status_color = colors.GREEN if is_compiled else colors.RED
    with open(log_file, 'a', encoding="utf-8") as f:
        # Log compilation results
        log(f, f"Compilation Test Results (Iteration {iteration_count}/{max_iterations}): {compilation_status}", compilation_status_color, bold=True)
        # Log the validator's assessment
        log(f, f"Validator assessment (Iteration {iteration_count}/{max_iterations}):", validator_response_color, bold=True)
        log(f, validator_response)

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
    
    print_colored(f"\nValidator sending FEEDBACK to {next_node}", "1;33")
    
    return {
        "messages": [AIMessage(content=feedback_message, name="Validator")],
        "user_request": state["user_request"],
        "generator_specs": generator_specs,
        "generator_code": generator_code,
        "validator_assessment": feedback_message,
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
        "model_source": state["model_source"],
        "session_dir": session_dir,
        "log_file": log_file,
        "next_step": next_node,
        "system_metrics": system_metrics
    }