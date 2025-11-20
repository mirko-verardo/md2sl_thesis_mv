from datetime import datetime
from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from models import AgentState
from agents.validator.validator_prompts import get_validator_template
from utils import colors
from utils.general import print_colored, log, extract_c_code, compile_c_code, initialize_llm
from utils.multi_agent import get_parser_requirements, get_satisfaction



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

    # Logger
    f = open(log_file, 'a', encoding="utf-8")
    
    # Extract clean c code
    clean_c_code = extract_c_code(generator_code)
    if not clean_c_code:
        print_colored("Warning: Could not extract clean C code, using original code", "1;33")
        clean_c_code = generator_code
    
    # Main problems
    problems = [
        "agent stopped due to max iterations.",
        "agent stopped due to iteration limit or time limit."
    ]
    if clean_c_code.lower() in problems:
        validator_response = "Retry with a simpler and shorter version"
        validator_response_color = colors.YELLOW
        is_compiled = False
    else:
        # Save c code to temporary file for compilation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        c_file_name = f"parser_{timestamp}.c"
        c_file_path = session_dir / c_file_name
        
        with open(c_file_path, 'w', encoding="utf-8") as ff:
            ff.write(clean_c_code)
        
        print_colored(f"\nSaved C code to: {c_file_path} for compilation testing", "1;36")
        
        # Compile the code
        print_colored("\n--- Testing Compilation ---", "1;33")
        compilation_result = compile_c_code(c_file_path)
        # Check if code has been compiled with success
        is_compiled = compilation_result['success']
        
        # Record parser validation in metrics
        system_metrics.record_parser_validation(c_file_name, is_compiled)

        # Print compilation status
        if is_compiled:
            compilation_status = "✅ Compilation successful!"
            color = colors.GREEN
        else:
            compilation_status = "❌ Compilation failed with the following errors:\n" + compilation_result['stderr']
            color = colors.RED
        
        # Log compilation results
        log(f, f"--- Compilation Test Results (Iteration {iteration_count}/{max_iterations}) ---")
        log(f, compilation_status, color, bold=True)

        # Manage prompt's input
        validator_input = {
            "requirements": get_parser_requirements(),
            "specifications": "",
            "code": clean_c_code,
            "compilation_status": compilation_status
        }
        if generator_specs:
            validator_input.update({
                "specifications": generator_specs
            })
        
        # Create the prompt
        validator_template = get_validator_template()
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
            validator_response = validator_result["output"]
            validator_response_color = colors.BLUE
        except Exception as e:
            validator_response = f"Error occurred during code validation: {str(e)}\n\nPlease try again."
            validator_response_color = colors.RED
    
    # Log the validator's assessment
    log(f, f"Validator assessment (Iteration {iteration_count}/{max_iterations}):", validator_response_color, bold=True)
    log(f, validator_response)
    f.close()

    # Check if code is satisfactory
    is_satisfactory = is_compiled and get_satisfaction(validator_response) == "SATISFACTORY"
    # Check if code is the final iteration
    is_final_iteration = iteration_count >= max_iterations
    # Check if code has been generated
    is_code_generated = not validator_response.startswith("Retry")

    if is_satisfactory:
        # end
        next_node = "Supervisor"
        feedback_message = f"The parser implementation has been validated and is SATISFACTORY. It compiles successfully and meets all requirements. Here's the final assessment:\n\n{validator_response}"
    elif is_final_iteration:
        # end
        next_node = "Supervisor"
        feedback_message = f"After iterations limit, this is the best parser implementation available. While it may not be perfect, it could serve as a good starting point. Here's the final assessment:\n\n{validator_response}"
    elif is_compiled:
        # continue
        next_node = "Generator"
        feedback_message = f"The parser implementation compiles successfully but needs improvements. Here's the assessment:\n\n{validator_response}"
    elif is_code_generated:
        # continue
        next_node = "Generator"
        feedback_message = f"The parser implementation does not compile and needs improvements. Here's the assessment:\n\n{validator_response}"
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