from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from models import AgentState
from utils import colors
from utils.general import print_colored, initialize_llm, get_parser_requirements
from agents.supervisor import supervisor_prompts



def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor agent that converses with the user and manages the parser generation process."""
    user_action = state["user_action"]
    user_request = state["user_request"]
    generator_code = state["generator_code"]
    code_assessment = state["code_assessment"]
    model_source = state["model_source"]
    last_parser = state["last_parser"]
    
    # Create the prompt
    supervisor_template = supervisor_prompts.get_supervisor_template()
    supervisor_input = {
        "input": user_request
    }
    if generator_code and code_assessment:
        adaptive_instructions = supervisor_prompts.get_supervisor_input_validated()
        supervisor_input.update({
            "code": generator_code,
            "assessment": code_assessment
        })
        purpose = "providing final parser"
        next_step = "FINISH"
    elif user_action == "GENERATE_PARSER":
        adaptive_instructions = supervisor_prompts.get_supervisor_input_generate_parser()
        supervisor_input.update({
            "requirements": get_parser_requirements()
        })
        purpose = "creating detailed specifications"
        next_step = "Orchestrator"
    elif user_action == "CORRECT_ERROR":
        # NB: here it can't be None
        if not last_parser:
            raise Exception("Something goes wrong :(")
        
        adaptive_instructions = supervisor_prompts.get_supervisor_input_correct_error()
        supervisor_input.update({
            "code": last_parser["code"],
            "assessment": last_parser["assessment"]
        })
        purpose = "creating updated specifications"
        next_step = "Orchestrator"
    elif user_action == "ASSESS_CODE":
        # NB: here it can't be None
        if not last_parser:
            raise Exception("Something goes wrong :(")
        
        adaptive_instructions = supervisor_prompts.get_supervisor_input_assess_code()
        supervisor_input.update({
            "code": last_parser["code"],
            "assessment": last_parser["assessment"]
        })
        purpose = "providing code assessment"
        next_step = "FINISH"
    else:
        # GENERAL_CONVERSATION
        adaptive_instructions = supervisor_prompts.get_supervisor_input_general_conversation()
        purpose = "conversation"
        next_step = "FINISH"
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

    try:
        supervisor_result = supervisor_executor.invoke(supervisor_input)
        supervisor_response = str(supervisor_result.content)
    except Exception as e:
        supervisor_response = f"Error occurred during supervisor response: {str(e)}\n\nPlease try again."

    # NB: set the specifications for generator
    supervisor_specifications = supervisor_response if next_step == "Orchestrator" else None

    print_colored(f"Supervisor RESPONSE ({purpose}):", colors.BLUE, bold=True)
    print_colored(supervisor_response, colors.BLUE)
    
    return {
        "messages": [AIMessage(content=supervisor_response, name="Supervisor")],
        "user_action": user_action,
        "user_request": user_request,
        "file_format": state["file_format"],
        "supervisor_specifications": supervisor_specifications,
        "generator_code": None,
        "compiler_result": None,
        "tester_result": None,
        "code_assessment": None,
        "iteration_count": state["iteration_count"],
        "max_iterations": state["max_iterations"],
        "model_source": model_source,
        "session_dir": state["session_dir"],
        "next_step": next_step,
        "system_metrics": state["system_metrics"],
        "last_parser": last_parser
    }