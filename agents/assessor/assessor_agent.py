from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from models import AgentState
from agents.assessor import assessor_prompts
from utils import colors
from utils.general import print_colored, initialize_llm, get_parser_requirements



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

    # Initialize model for validator
    assessor_llm = initialize_llm(model_source)
    assessor_llm.temperature = 0.4

    # Create a normal LLM call (no ReAct needed)
    validator_executor = LLMChain(
        llm=assessor_llm,
        prompt=assessor_prompt,
        verbose=True
    )
    
    # Invoke the agent
    try:
        assessor_result = validator_executor.invoke(assessor_input)
        assessor_response = str(assessor_result["text"])
        assessor_response_color = colors.BLUE
    except Exception as e:
        assessor_response = f"Error occurred during code assessment: {str(e)}\n\nPlease try again."
        assessor_response_color = colors.RED
    
    # Log the assessment
    print_colored(f"Assessor (Iteration {iteration_count}/{max_iterations}):", assessor_response_color, bold=True)
    print(assessor_response)
    
    return {
        "messages": [AIMessage(content=assessor_response, name="Assessor")],
        "user_action": state["user_action"],
        "user_request": state["user_request"],
        "file_format": state["file_format"],
        "supervisor_specifications": supervisor_specifications,
        "generator_code": generator_code,
        "validator_compilation": None,
        "validator_testing": None,
        "assessor_assessment": assessor_response,
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
        "model_source": model_source,
        "session_dir": state["session_dir"],
        "next_step": "Orchestrator",
        "system_metrics": state["system_metrics"]
    }