from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from models import AgentState, CompilationCheckTool, mister_wolf
from agents.generator.generator_prompts import get_generator_template, get_feedback_template
from utils import colors
from utils.general import print_colored, log, initialize_llm
from utils.multi_agent import get_parser_requirements



def generator_node(state: AgentState) -> AgentState:
    """Generator agent that creates C code."""
    generator_specs = state["generator_specs"]
    validator_assessment = state["validator_assessment"]
    iteration_count = state["iteration_count"]
    max_iterations = state["max_iterations"]
    model_source = state["model_source"]
    log_file = state["log_file"]
    system_metrics = state["system_metrics"]
    
    # TODO: da rivedere
    system_metrics.increment_generator_validator_interaction()

    # Manage prompt's input
    generator_input = {
        "requirements": get_parser_requirements(),
        "specifications": generator_specs if generator_specs is not None else "",
        "feedback": ""
    }
    if validator_assessment:
        generator_input.update({
            "feedback": get_feedback_template(),
            "validator_assessment": validator_assessment
        })

    # Create the prompt
    generator_template = get_generator_template()
    generator_prompt = PromptTemplate.from_template(generator_template)

    # Initialize model for generator
    generator_llm = initialize_llm(model_source)
    generator_llm.temperature = 0.5
    #generator_tools = [CompilationCheckTool()]
    generator_tools = [
        Tool(name="compilation_check", func=mister_wolf, description="blah blah blah")
    ]
    
    # Create the ReAct agent instead of OpenAI tools agent
    generator_agent = create_react_agent(generator_llm, generator_tools, generator_prompt)

    if validator_assessment == "Retry":
        max_i = 15 if iteration_count > 1 else 10
    else:
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
        generator_response = generator_result["output"]
        generator_response_color = colors.MAGENTA
        
        if "intermediate_steps" in generator_result:
            compilation_attempts = 0
            
            for step in generator_result["intermediate_steps"]:
                action = step[0]
                action_output = step[1]
                
                if action.tool == "compilation_check":
                    compilation_attempts += 1
                    compilation_success = "Compilation successful" in action_output
                    
                    system_metrics.record_tool_usage(compilation_success)
                    print_colored(f"\nCompilation check tool used (Attempt {compilation_attempts})", "1;32")
                    
                    if compilation_success:
                        print_colored("Compilation successful!", "1;32")
                    else:
                        print_colored("Compilation failed!", "1;31")
            
            if compilation_attempts == 0:
                print_colored("\nWarning: Compilation check tool was NOT used!", "1;33")
                
    except Exception as e:
        generator_response = f"Error occurred during code generation: {str(e)}\n\nPlease try again."
        generator_response_color = colors.RED
    
    # increment iteration
    iteration_count += 1
    
    with open(log_file, 'a', encoding="utf-8") as f:
        log(f, f"Generator (Iteration {iteration_count}/{max_iterations}):", generator_response_color, bold=True)
        log(f, generator_response)
    
    return {
        "messages": [AIMessage(content=generator_response, name="Generator")],
        "user_request": state["user_request"],
        "supervisor_memory": state["supervisor_memory"],
        "generator_specs": generator_specs,
        "generator_code": generator_response,
        "validator_assessment": None,
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
        "model_source": model_source,
        "session_dir": state["session_dir"],
        "log_file": log_file,
        "next_step": "Validator",
        "system_metrics": system_metrics
    }