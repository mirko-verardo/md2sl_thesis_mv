from langchain_core.messages import AIMessage, get_buffer_string
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from models import AgentState
from utils import colors, multi_agent
from utils.general import log, extract_c_code, initialize_llm
from agents.supervisor import supervisor_prompts



def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor agent that converses with the user and manages the parser generation process."""
    messages = state["messages"]
    user_request = state["user_request"]
    generator_specs = state["generator_specs"]
    generator_code = state["generator_code"]
    validator_assessment = state["validator_assessment"]
    iteration_count = state["iteration_count"]
    model_source = state["model_source"]
    log_file = state["log_file"]

    # Logger
    f = open(log_file, 'a', encoding="utf-8")
    
    # Create the prompt
    supervisor_template = supervisor_prompts.get_supervisor_template()
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
        handle_parsing_errors=True
    )

    supervisor_input = {
        "input": user_request,
        "conversation_history": get_buffer_string(messages)
    }

    if generator_code and validator_assessment:
        # Extract clean c code
        clean_c_code = extract_c_code(generator_code)
        if clean_c_code is None:
            clean_c_code = generator_code
        
        supervisor_input.update({
            "adaptive_instructions": supervisor_prompts.get_supervisor_input_validated(),
            "c_code": clean_c_code,
            "validator_assessment": validator_assessment,
            "code_satisfaction": multi_agent.get_satisfaction(validator_assessment),
            "compilation_status": multi_agent.get_compilation_status(validator_assessment),
            "code_satisfaction_instructions": multi_agent.get_satisfaction_instructions(validator_assessment)
        })
        result = supervisor_executor.invoke(supervisor_input)
        supervisor_response = result["output"]
        
        purpose = "providing final parser"
        generator_specs_new = generator_specs
        generator_code_new = generator_code
        iteration_count_new = iteration_count
        next_step = "FINISH"
    else:
        supervisor_input.update({
            "adaptive_instructions": supervisor_prompts.get_supervisor_input_actions()
        })
        result = supervisor_executor.invoke(supervisor_input)
        action = result["output"].strip()
        
        log(f, f"Supervisor action choosen: {action}", colors.CYAN, bold=True)
        
        if action == "GENERATE_PARSER":
            supervisor_input.update({
                "adaptive_instructions": supervisor_prompts.get_supervisor_input_generate_parser(),
                "requirements": multi_agent.get_parser_requirements()
            })
            result = supervisor_executor.invoke(supervisor_input)
            supervisor_response = result["output"]
            
            purpose = "creating detailed specifications"
            generator_specs_new = supervisor_response
            generator_code_new = None
            iteration_count_new = 0
            next_step = "Generator"
        elif action == "CORRECT_ERROR":
            #most_recent_assessment = most_recent_parser["validator_assessment"]
            supervisor_input.update({
                "adaptive_instructions": supervisor_prompts.get_supervisor_input_correct_error(),
                #"c_code": most_recent_parser["code"],
                #"validator_assessment": multi_agent.get_assessment(most_recent_assessment),
                #"code_satisfaction": multi_agent.get_satisfaction(most_recent_assessment),
                #"compilation_status": multi_agent.get_compilation_status(most_recent_assessment)
            })
            result = supervisor_executor.invoke(supervisor_input)
            supervisor_response = result["output"]
            
            purpose = "creating updated specifications"
            generator_specs_new = supervisor_response
            generator_code_new = None
            iteration_count_new = 0
            next_step = "Generator"
        elif action == "ASSESS_CODE":
            #most_recent_assessment = most_recent_parser["validator_assessment"]
            supervisor_input.update({
                "adaptive_instructions": supervisor_prompts.get_supervisor_input_assess_code(),
                #"c_code": most_recent_parser["code"],
                #"validator_assessment": multi_agent.get_assessment(most_recent_assessment),
                #"code_satisfaction": multi_agent.get_satisfaction(most_recent_assessment),
                #"compilation_status": multi_agent.get_compilation_status(most_recent_assessment)
            })
            result = supervisor_executor.invoke(supervisor_input)
            supervisor_response = result["output"]
            
            purpose = "providing code assessment"
            generator_specs_new = generator_specs
            generator_code_new = generator_code
            iteration_count_new = iteration_count
            next_step = "FINISH"
        else:
            # GENERAL_CONVERSATION
            supervisor_input.update({
                "adaptive_instructions": supervisor_prompts.get_supervisor_input_general_conversation()
            })
            result = supervisor_executor.invoke(supervisor_input)
            supervisor_response = result["output"]
            
            purpose = "conversation"
            generator_specs_new = generator_specs
            generator_code_new = generator_code
            iteration_count_new = iteration_count
            next_step = "FINISH"

    log(f, f"Supervisor ({purpose}):", colors.BLUE, bold=True)
    log(f, supervisor_response)
    f.close()
    
    return {
        "messages": [AIMessage(content=supervisor_response, name="Supervisor")],
        "user_request": user_request,
        "generator_specs": generator_specs_new,
        "generator_code": generator_code_new,
        "validator_assessment": None,
        "iteration_count": iteration_count_new,
        "max_iterations": state["max_iterations"],
        "model_source": model_source,
        "session_dir": state["session_dir"],
        "log_file": log_file,
        "next_step": next_step,
        "system_metrics": state["system_metrics"]
    }