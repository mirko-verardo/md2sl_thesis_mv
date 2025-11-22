from langchain_core.messages import AIMessage, get_buffer_string
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from models import AgentState
from utils import colors, multi_agent
from utils.general import print_colored, initialize_llm
from agents.supervisor import supervisor_prompts



def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor agent that converses with the user and manages the parser generation process."""
    messages = state["messages"]
    user_action = state["user_action"]
    user_request = state["user_request"]
    generator_specs = state["generator_specs"]
    generator_code = state["generator_code"]
    validator_assessment = state["validator_assessment"]
    model_source = state["model_source"]
    
    # Create the prompt
    supervisor_template = supervisor_prompts.get_supervisor_template()
    # temp
    if generator_code and validator_assessment:
        supervisor_template = supervisor_template.replace("{adaptive_instructions}", supervisor_prompts.get_supervisor_input_validated())
    supervisor_prompt = PromptTemplate.from_template(supervisor_template)

    # Initialize model for supervisor
    supervisor_llm = initialize_llm(model_source)
    supervisor_llm.temperature = 0.6
    supervisor_tools = []
    
    # Create the ReAct agent instead of OpenAI tools agent
    supervisor_agent = create_react_agent(supervisor_llm, supervisor_tools, supervisor_prompt)
    
    supervisor_executor = AgentExecutor(
        agent=supervisor_agent,
        tools=supervisor_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="force"
    )

    supervisor_input = {
        "input": user_request,
        "conversation_history": get_buffer_string(messages)
    }

    if generator_code and validator_assessment:
        supervisor_input.update({
            "c_code": generator_code,
            "validator_assessment": validator_assessment
        })

        #prompt_input = supervisor_input.copy()
        #prompt_input.update({
        #    "tools": "",
        #    "tool_names": "",
        #    "agent_scratchpad": ""
        #})
        #final_prompt = supervisor_prompt.format(**prompt_input)
        #prova_file = state["session_dir"] / "prova.txt"
        #with open(prova_file, 'w', encoding="utf-8") as kk:
        #    kk.write(final_prompt)

        result = supervisor_executor.invoke(supervisor_input)
        supervisor_response = result["output"]
        
        purpose = "providing final parser"
        next_step = "FINISH"
    else:
        #supervisor_input.update({
        #    "adaptive_instructions": supervisor_prompts.get_supervisor_input_actions()
        #})
        #result = supervisor_executor.invoke(supervisor_input)
        #user_action = str(result["output"]).strip()

        if user_action == "GENERATE_PARSER":
            # temp
            adaptive_instructions = supervisor_prompts.get_supervisor_input_generate_parser()
            adaptive_instructions = adaptive_instructions.replace("{requirements}", multi_agent.get_parser_requirements())
            supervisor_input.update({
                "adaptive_instructions": adaptive_instructions
            })
            result = supervisor_executor.invoke(supervisor_input)
            supervisor_response = result["output"]
            # NB: change the specs for generator
            generator_specs = supervisor_response
            
            purpose = "creating detailed specifications"
            next_step = "Generator"
        elif user_action == "CORRECT_ERROR":
            #most_recent_assessment = most_recent_parser["validator_assessment"]
            supervisor_input.update({
                "adaptive_instructions": supervisor_prompts.get_supervisor_input_correct_error(),
                #"c_code": most_recent_parser["code"],
                #"validator_assessment": most_recent_assessment if most_recent_assessment else "No assessment available"
            })
            result = supervisor_executor.invoke(supervisor_input)
            supervisor_response = result["output"]
            # NB: change the specs for generator
            generator_specs = supervisor_response
            
            purpose = "creating updated specifications"
            next_step = "Generator"
        elif user_action == "ASSESS_CODE":
            #most_recent_assessment = most_recent_parser["validator_assessment"]
            supervisor_input.update({
                "adaptive_instructions": supervisor_prompts.get_supervisor_input_assess_code(),
                #"c_code": most_recent_parser["code"],
                #"validator_assessment": most_recent_assessment if most_recent_assessment else "No assessment available"
            })
            result = supervisor_executor.invoke(supervisor_input)
            supervisor_response = result["output"]
            
            purpose = "providing code assessment"
            next_step = "FINISH"
        else:
            # GENERAL_CONVERSATION
            supervisor_input.update({
                "adaptive_instructions": supervisor_prompts.get_supervisor_input_general_conversation()
            })
            result = supervisor_executor.invoke(supervisor_input)
            supervisor_response = result["output"]
            
            purpose = "conversation"
            next_step = "FINISH"

    print_colored(f"Supervisor ({purpose}):", colors.BLUE, bold=True)
    print(supervisor_response)
    
    return {
        "messages": [AIMessage(content=supervisor_response, name="Supervisor")],
        "user_action": user_action,
        "user_request": user_request,
        "generator_specs": generator_specs,
        "generator_code": generator_code,
        "validator_assessment": None,
        "iteration_count": state["iteration_count"],
        "max_iterations": state["max_iterations"],
        "model_source": model_source,
        "session_dir": state["session_dir"],
        "next_step": next_step,
        "system_metrics": state["system_metrics"]
    }