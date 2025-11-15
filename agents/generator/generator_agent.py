from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from models import AgentState, CompilationCheckTool
from agents.generator.generator_prompts import get_generator_template
from utils import colors
from utils.general import print_colored, log, initialize_llm



def generator_node(state: AgentState) -> AgentState:
    """Generator agent that creates C code."""
    messages = state["messages"]
    generator_specs = state["generator_specs"]
    user_request = state["user_request"]
    iteration_count = state["iteration_count"]
    model_source = state["model_source"]
    session_dir = state["session_dir"]
    log_file = state["log_file"]
    
    if state.get("system_metrics") and (iteration_count == 0 or state["next_step"] == "Generator"):
        state["system_metrics"].increment_generator_validator_interaction()
    
    validator_messages = [msg for msg in messages if hasattr(msg, 'name') and msg.name == "Validator"]
    has_feedback = len(validator_messages) > 0 and iteration_count > 0

    generator_specs_coalesced = generator_specs if generator_specs is not None else ""
    feedback = validator_messages[-1].content if has_feedback else ""
    generator_template = get_generator_template(generator_specs_coalesced, iteration_count, feedback)

    # Initialize model for generator
    generator_llm = initialize_llm(model_source)
    generator_llm.temperature = 0.5
    generator_tools = [CompilationCheckTool()]

    # Create a prompt using PromptTemplate.from_template
    generator_prompt = PromptTemplate.from_template(generator_template)
    
    # Create the ReAct agent instead of OpenAI tools agent
    generator_agent = create_react_agent(generator_llm, generator_tools, generator_prompt)
    
    generator_executor = AgentExecutor(
        agent=generator_agent,
        tools=generator_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        early_stopping_method="force",
        return_intermediate_steps=True
    )
    
    try:
        generator_result = generator_executor.invoke({})
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
                    
                    if state.get("system_metrics"):
                        state["system_metrics"].record_tool_usage(compilation_success)
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
        log(f, f"Generator (Iteration {iteration_count}):", generator_response_color, bold=True)
        log(f, generator_response)
    
    next_step = "Supervisor" if iteration_count > state["max_iterations"] else "Validator"
    
    return {
        "messages": [AIMessage(content=generator_response, name="Generator")],
        "user_request": user_request,
        "supervisor_memory": state["supervisor_memory"],
        "generator_specs": generator_specs,
        "generator_code": generator_response,
        "iteration_count": iteration_count,
        "max_iterations": state["max_iterations"],
        "model_source": model_source,
        "session_dir": session_dir,
        "log_file": log_file,
        "next_step": next_step,
        "parser_mode": state["parser_mode"],
        "system_metrics": state.get("system_metrics")
    }