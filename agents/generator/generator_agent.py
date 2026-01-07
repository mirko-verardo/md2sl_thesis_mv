from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from models import AgentState
from agents.generator import generator_prompts
from utils import colors
from utils.general import print_colored, extract_c_code, initialize_llm, get_parser_requirements
from utils.multi_agent import invoke_agent



def generator_node(state: AgentState) -> AgentState:
    """Generator agent that creates C code."""
    supervisor_specifications = state["supervisor_specifications"]
    generator_code = state["generator_code"]
    code_assessment = state["code_assessment"]
    iteration_count = state["iteration_count"]
    max_iterations = state["max_iterations"]
    model_source = state["model_source"]

    # NB: here it can't be None
    if not supervisor_specifications:
        raise Exception("Something goes wrong :(")

    # Create the prompt
    generator_template = generator_prompts.get_generator_template()
    generator_input = {
        "requirements": get_parser_requirements(),
        "specifications": supervisor_specifications
    }
    if generator_code and code_assessment:
        feedback_template = generator_prompts.get_fixing_template()
        generator_input.update({
            "code": generator_code,
            "assessment": code_assessment
        })
    else:
        feedback_template = generator_prompts.get_starting_template()
    generator_template = generator_template.replace("{feedback}", feedback_template)
    generator_prompt = PromptTemplate.from_template(generator_template)

    # Initialize model for generator
    generator_llm = initialize_llm(model_source)

    # Create a normal LLM chain (no ReAct needed)
    generator_executor = generator_prompt | generator_llm

    # Print the prompt
    generator_prompt_rendered = generator_prompt.format(**generator_input)
    print_colored(f"Generator PROMPT (Iteration {iteration_count}/{max_iterations}):", colors.GREEN, bold=True)
    print_colored(generator_prompt_rendered, colors.GREEN)

    # Invoke the agent
    generator_outcome, generator_response = invoke_agent(generator_executor, generator_input)
    if generator_outcome:
        generator_response_color = colors.MAGENTA
        # Extract clean c code
        generator_response_code = extract_c_code(generator_response)
    else:
        generator_response_color = colors.RED
        generator_response_code = None
    
    print_colored(f"Generator RESPONSE (Iteration {iteration_count}/{max_iterations}):", generator_response_color, bold=True)
    print_colored(generator_response, generator_response_color)
    
    return {
        "messages": [AIMessage(content=generator_response, name="Generator")],
        "user_action": state["user_action"],
        "user_request": state["user_request"],
        "file_format": state["file_format"],
        "supervisor_specifications": supervisor_specifications,
        "generator_code": generator_response_code,
        "compiler_result": None,
        "tester_result": None,
        "code_assessment": None,
        "round": state["round"],
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
        "model_source": model_source,
        "session_dir": state["session_dir"],
        "next_step": "Orchestrator",
        "benchmark_metrics": state["benchmark_metrics"]
    }