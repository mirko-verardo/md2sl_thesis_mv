from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from models import AgentState
from utils import colors
from utils.general import print_colored, initialize_llm, get_parser_requirements
from utils.multi_agent import invoke_agent
from agents.supervisor import supervisor_prompts



def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor agent that converses with the user and manages the parser generation process."""
    user_action = state["user_action"]
    user_request = state["user_request"]
    generator_code = state["generator_code"]
    code_assessment = state["code_assessment"]
    iteration_count = state["iteration_count"]
    model_source = state["model_source"]
    
    # Create the prompt
    supervisor_template = supervisor_prompts.get_supervisor_template()
    supervisor_input = {
        "input": user_request
    }
    if iteration_count == 0:
        if user_action == "GENERATE_PARSER":
            adaptive_instructions = supervisor_prompts.get_supervisor_input_generate_parser()
            supervisor_input.update({
                "requirements": get_parser_requirements()
            })
            purpose = "creating detailed specifications"
            next_step = "Orchestrator"
        elif user_action == "CORRECT_ERROR" and generator_code and code_assessment:
            adaptive_instructions = supervisor_prompts.get_supervisor_input_correct_error()
            supervisor_input.update({
                "code": generator_code,
                "assessment": code_assessment
            })
            purpose = "creating updated specifications"
            next_step = "Orchestrator"
        elif user_action == "ASSESS_CODE" and generator_code and code_assessment:
            adaptive_instructions = supervisor_prompts.get_supervisor_input_assess_code()
            supervisor_input.update({
                "code": generator_code,
                "assessment": code_assessment
            })
            purpose = "providing code assessment"
            next_step = "FINISH"
        elif user_action == "GENERAL_CONVERSATION":
            adaptive_instructions = supervisor_prompts.get_supervisor_input_general_conversation()
            purpose = "conversation"
            next_step = "FINISH"
        else:
            raise Exception("Something goes wrong :(")
    elif generator_code and code_assessment:
        adaptive_instructions = supervisor_prompts.get_supervisor_input_validated()
        supervisor_input.update({
            "code": generator_code,
            "assessment": code_assessment
        })
        purpose = "providing final parser"
        next_step = "FINISH"
    else:
        raise Exception("Something goes wrong :(")
    supervisor_template = supervisor_template.replace("{adaptive_instructions}", adaptive_instructions)
    supervisor_prompt = PromptTemplate.from_template(supervisor_template)

    # Initialize model for supervisor
    supervisor_llm = initialize_llm(model_source, 0.6)
    
    # Create a normal LLM chain (no ReAct needed)
    supervisor_executor = supervisor_prompt | supervisor_llm

    # Print the prompt
    #prompt_input = supervisor_input.copy()
    #prompt_input.update({
    #    "tools": "",
    #    "tool_names": "",
    #    "agent_scratchpad": ""
    #})
    #supervisor_prompt_rendered = supervisor_prompt.format(**prompt_input)
    supervisor_prompt_rendered = supervisor_prompt.format(**supervisor_input)
    print_colored(f"Supervisor PROMPT ({purpose}):", colors.GREEN, bold=True)
    print_colored(supervisor_prompt_rendered, colors.GREEN)

    # Invoke the agent
    supervisor_outcome, supervisor_response = invoke_agent(supervisor_executor, supervisor_input)
    if supervisor_outcome:
        supervisor_response_color = colors.BLUE
        # Set the specifications (NB: only for orchestrator -> generator)
        supervisor_specifications = supervisor_response if next_step == "Orchestrator" else None
    else:
        supervisor_response_color = colors.RED
        supervisor_specifications = None

    print_colored(f"Supervisor RESPONSE ({purpose}):", supervisor_response_color, bold=True)
    print_colored(supervisor_response, supervisor_response_color)
    
    return {
        "messages": [AIMessage(content=supervisor_response, name="Supervisor")],
        "user_action": user_action,
        "user_request": user_request,
        "file_format": state["file_format"],
        "supervisor_specifications": supervisor_specifications,
        "generator_code": generator_code,
        "compiler_result": None,
        "tester_result": None,
        "code_assessment": code_assessment,
        "round": state["round"],
        "iteration_count": iteration_count,
        "max_iterations": state["max_iterations"],
        "model_source": model_source,
        "session_dir": state["session_dir"],
        "next_step": next_step,
        "benchmark_metrics": state["benchmark_metrics"]
    }