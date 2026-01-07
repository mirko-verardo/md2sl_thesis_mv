from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from models import AgentState
from agents.assessor import assessor_prompts
from utils import colors
from utils.general import print_colored, initialize_llm, get_parser_requirements
from utils.multi_agent import invoke_agent



def assessor_node(state: AgentState) -> AgentState:
    """Assessor agent that evaluates parser code."""
    supervisor_specifications = state["supervisor_specifications"]
    generator_code = state["generator_code"]
    iteration_count = state["iteration_count"]
    max_iterations = state["max_iterations"]
    model_source = state["model_source"]

    # NB: here they can't be None
    if not supervisor_specifications or not generator_code:
        raise Exception("Something goes wrong :(")

    # Create the prompt
    assessor_template = assessor_prompts.get_assessor_template()
    assessor_input = {
        "requirements": get_parser_requirements(),
        "specifications": supervisor_specifications,
        "code": generator_code
    }
    assessor_prompt = PromptTemplate.from_template(assessor_template)

    # Initialize model for assessor
    assessor_llm = initialize_llm(model_source, 0.4)

    # Create a normal LLM chain (no ReAct needed)
    assessor_executor = assessor_prompt | assessor_llm

    # Print the prompt
    assessor_prompt_rendered = assessor_prompt.format(**assessor_input)
    print_colored(f"Assessor PROMPT (Iteration {iteration_count}/{max_iterations}):", colors.GREEN, bold=True)
    print_colored(assessor_prompt_rendered, colors.GREEN)
    
    # Invoke the agent
    assessor_outcome, assessor_response = invoke_agent(assessor_executor, assessor_input)
    if assessor_outcome:
        assessor_response_color = colors.CYAN
        code_assessment = assessor_response
    else:
        assessor_response_color = colors.RED
        code_assessment = None
    
    # Print the assessment
    print_colored(f"Assessor RESPONSE (Iteration {iteration_count}/{max_iterations}):", assessor_response_color, bold=True)
    print_colored(assessor_response, assessor_response_color)
    
    return {
        "messages": [AIMessage(content=assessor_response, name="Assessor")],
        "user_action": state["user_action"],
        "user_request": state["user_request"],
        "file_format": state["file_format"],
        "supervisor_specifications": supervisor_specifications,
        "generator_code": generator_code,
        "compiler_result": None,
        "tester_result": None,
        "code_assessment": code_assessment,
        "round": state["round"],
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
        "model_source": model_source,
        "session_dir": state["session_dir"],
        "next_step": "Orchestrator",
        "benchmark_metrics": state["benchmark_metrics"]
    }