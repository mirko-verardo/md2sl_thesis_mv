from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from models import AgentState
from agents.generator import generator_prompts
from utils import colors
from utils.general import print_colored, extract_c_code, initialize_llm, get_parser_requirements



def generator_node(state: AgentState) -> AgentState:
    """Generator agent that creates C code."""
    supervisor_specifications = state["supervisor_specifications"]
    generator_code = state["generator_code"]
    assessor_assessment = state["assessor_assessment"]
    iteration_count = state["iteration_count"]
    max_iterations = state["max_iterations"]
    model_source = state["model_source"]

    # Create the prompt
    generator_template = generator_prompts.get_generator_template()
    generator_input = {
        "requirements": get_parser_requirements(),
        # NB: here it can't be None
        "specifications": supervisor_specifications if supervisor_specifications else ""
    }
    if generator_code and assessor_assessment:
        feedback_template = generator_prompts.get_feedback_template()
        generator_input.update({
            "code": generator_code,
            "assessment": assessor_assessment
        })
    else:
        feedback_template = ""
    generator_template = generator_template.replace("{feedback}", feedback_template)
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
        generator_response = f"Error occurred during code generation: {str(e)}\n\nPlease try again."
        generator_response_color = colors.RED
        generator_response_c_code = None
    
    print_colored(f"Generator (Iteration {iteration_count}/{max_iterations}):", generator_response_color, bold=True)
    print(generator_response)
    
    return {
        "messages": [AIMessage(content=generator_response, name="Generator")],
        "user_action": state["user_action"],
        "user_request": state["user_request"],
        "file_format": state["file_format"],
        "supervisor_specifications": supervisor_specifications,
        "generator_code": generator_response_c_code,
        "validator_compilation": None,
        "validator_testing": None,
        "assessor_assessment": None,
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
        "model_source": model_source,
        "session_dir": state["session_dir"],
        "next_step": "Orchestrator",
        "system_metrics": state["system_metrics"]
    }