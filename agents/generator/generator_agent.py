from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from models import AgentState, CompilationCheck, ExecutionCheck
from agents.generator import generator_prompts
from utils import colors
from utils.general import print_colored, extract_c_code, initialize_llm
from utils.multi_agent import get_parser_requirements



def generator_node(state: AgentState) -> AgentState:
    """Generator agent that creates C code."""
    file_format = state["file_format"]
    generator_specs = state["generator_specs"]
    validator_assessment = state["validator_assessment"]
    iteration_count = state["iteration_count"]
    max_iterations = state["max_iterations"]
    model_source = state["model_source"]
    system_metrics = state["system_metrics"]

    # Record generator validator interaction numbers
    system_metrics.increment_generator_validator_interaction()

    # Create the prompt
    generator_template = generator_prompts.get_generator_template()
    generator_template = generator_template.replace("{specifications}", generator_prompts.get_specifications_template() if generator_specs else "")
    generator_template = generator_template.replace("{feedback}", generator_prompts.get_feedback_template() if validator_assessment else "")
    generator_input = {
        "requirements": get_parser_requirements()
    }
    if generator_specs:
        generator_input.update({
            "supervisor_specifications": generator_specs
        })
    if validator_assessment:
        generator_input.update({
            "validator_assessment": validator_assessment
        })
    generator_prompt = PromptTemplate.from_template(generator_template)

    # Initialize model for generator
    generator_llm = initialize_llm(model_source)
    generator_llm.temperature = 0.5
    generator_tools = [ CompilationCheck, ExecutionCheck(file_format) ]
    
    # Create the ReAct agent instead of OpenAI tools agent
    generator_agent = create_react_agent(generator_llm, generator_tools, generator_prompt)

    #max_i = 10 if validator_assessment.startswith("Retry") else 5
    max_i = 5
    
    generator_executor = AgentExecutor(
        agent=generator_agent,
        tools=generator_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=max_i,
        early_stopping_method="force",
        return_intermediate_steps=True
    )
    
    try:
        generator_result = generator_executor.invoke(generator_input)
        generator_response = str(generator_result["output"])
        generator_response_color = colors.MAGENTA
        # Extract clean c code
        generator_response_c_code = extract_c_code(generator_response)
        if not generator_response_c_code:
            print_colored("Warning: Could not extract clean C code, using original code", colors.YELLOW, bold=True)
            generator_response_c_code = generator_response
    except Exception as e:
        generator_response = f"Error occurred during code generation: {str(e)}\n\nPlease try again."
        generator_response_color = colors.RED
        generator_response_c_code = None
    
    action_attempts = {}
    steps = generator_result.get("intermediate_steps", []) if generator_result else []
    
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