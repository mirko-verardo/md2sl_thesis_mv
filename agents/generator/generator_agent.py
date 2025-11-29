from langchain_core.messages import AIMessage, get_buffer_string
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from models import AgentState
from agents.generator import generator_prompts
from utils import colors
from utils.general import print_colored, extract_c_code, initialize_llm, get_parser_requirements



def generator_node(state: AgentState) -> AgentState:
    """Generator agent that creates C code."""
    messages = state["messages"]
    file_format = state["file_format"]
    generator_specs = state["generator_specs"]
    generator_code = state["generator_code"]
    validator_assessment = state["validator_assessment"]
    iteration_count = state["iteration_count"]
    max_iterations = state["max_iterations"]
    model_source = state["model_source"]
    system_metrics = state["system_metrics"]

    # Record generator validator interaction numbers
    system_metrics.increment_generator_validator_interaction()

    # Create the prompt
    generator_template = generator_prompts.get_generator_template()
    generator_template = generator_template.replace("{feedback}", generator_prompts.get_feedback_template() if validator_assessment else "")
    generator_input = {
        "requirements": get_parser_requirements(),
        # NB: here it can't be None
        "specifications": generator_specs if generator_specs else "",
        #"conversation_history": get_buffer_string(messages)
    }
    if generator_code and validator_assessment:
        generator_input.update({
            "validator_assessment": validator_assessment,
            "code": generator_code
        })
    generator_prompt = PromptTemplate.from_template(generator_template)

    # Initialize model for generator
    generator_llm = initialize_llm(model_source)
    generator_llm.temperature = 0.5

    # Create a normal LLM call (no ReAct needed)
    generator_executor = LLMChain(
        llm=generator_llm,
        prompt=generator_prompt,
        verbose=True
    )
    
    try:
        generator_result = generator_executor.invoke(generator_input)
        generator_response = str(generator_result["text"])
        generator_response_color = colors.MAGENTA
        # Extract clean c code
        generator_response_c_code = extract_c_code(generator_response)
        if not generator_response_c_code:
            print_colored("Warning: Could not extract clean C code, using original code", colors.YELLOW, bold=True)
            generator_response_c_code = generator_response
    except Exception as e:
        generator_result = {}
        generator_response = f"Error occurred during code generation: {str(e)}\n\nPlease try again."
        generator_response_color = colors.RED
        generator_response_c_code = None
    
    action_attempts = {}
    steps = generator_result.get("intermediate_steps", [])
    
    for step in steps:
        action = step[0]
        action_output = step[1]
        
        action_tool = action.tool
        if action_tool not in [ "compilation_check", "execution_check" ]:
            continue

        attempts = action_attempts.get(action_tool, 0) + 1
        action_attempts.update({ action_tool: attempts })
        action_success = action_output["success"]
        
        system_metrics.record_tool_usage(action_tool, action_success)
        print_colored(f"\n{action_tool} tool used (attempt {attempts})", colors.GREEN, bold=True)
        
        if action_success:
            print_colored(f"{action_tool} successful!", colors.GREEN, bold=True)
        else:
            print_colored(f"{action_tool} failed!", colors.RED, bold=True)
        
    compilation_attempts = action_attempts.get("compilation_check", 0)
    execution_attempts = action_attempts.get("execution_check", 0)

    if compilation_attempts == 0:
        print_colored("\nWarning: Compilation check tool was NOT used!", colors.YELLOW, bold=True)
    if execution_attempts == 0:
        print_colored("\nWarning: Execution check tool was NOT used!", colors.YELLOW, bold=True)
    
    # increment iteration
    iteration_count += 1
    
    print_colored(f"Generator (Iteration {iteration_count}/{max_iterations}):", generator_response_color, bold=True)
    print(generator_response)
    
    return {
        "messages": [AIMessage(content=generator_response, name="Generator")],
        "user_action": state["user_action"],
        "user_request": state["user_request"],
        "file_format": file_format,
        "generator_specs": generator_specs,
        "generator_code": generator_response_c_code,
        "validator_assessment": None,
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
        "model_source": model_source,
        "session_dir": state["session_dir"],
        "next_step": "Validator",
        "system_metrics": system_metrics
    }