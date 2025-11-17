import re
from pathlib import Path
from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from models import AgentState, ExceptionTool
from utils import colors, multi_agent
from utils.general import print_colored, log, extract_c_code, initialize_llm
from agents.supervisor import supervisor_prompts



def __read_conversation_log(log_file: Path | None) -> str:
    """Read the entire conversation log to provide context for the supervisor."""
    try:
        with open(log_file, 'r', encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print_colored(f"Warning: Could not read log file: {e}", "1;33")
        return ""

def __extract_validator_assessment(c_log: str, i_count: int) -> str | None:
    """Extract the validator's assessment from the conversation log."""

    # search for a specific pattern
    pattern = rf"Validator assessment \(Iteration {i_count}/{i_count}\):\n(.*?)(?:\n\n|\Z)"
    matches = re.findall(pattern, c_log, re.DOTALL)
    if matches:
        return matches[-1].strip()
    
    # search for a more general one
    pattern = r"Validator assessment.*?:\n(.*?)(?:\n\n|\Z)"
    matches = re.findall(pattern, c_log, re.DOTALL)
    if matches:
        return matches[-1].strip()
    
    return None

def __update_supervisor_memory(s_memory: list[dict], c_log: str, code: str, i_count: int) -> list[dict]:
    """Update the supervisor's memory with information from the conversation log."""
    # get assessment
    v_assessment = __extract_validator_assessment(c_log, i_count)
    
    # update assessment if code exists
    code_exists = False
    for entry in s_memory:
        if entry.get("code") == code:
            code_exists = True
            if v_assessment:
                entry["validator_assessment"] = v_assessment
            break
    
    # else add assessment
    if not code_exists:
        new_entry = {
            'code': code,
            'iteration': i_count,
            'validator_assessment': v_assessment
        }
        s_memory.append(new_entry)
    
    return s_memory

def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor agent that converses with the user and manages the parser generation process."""
    messages = state["messages"]
    user_request = state["user_request"]
    supervisor_memory = state["supervisor_memory"]
    generator_code = state["generator_code"]
    iteration_count = state["iteration_count"]
    model_source = state["model_source"]
    session_dir = state["session_dir"]
    log_file = state["log_file"]
    
    conversation_log = __read_conversation_log(log_file)

    if generator_code is not None and iteration_count > 0:
        # Extract clean c code
        clean_c_code = extract_c_code(generator_code)
        if clean_c_code is None:
            clean_c_code = generator_code
        supervisor_memory = __update_supervisor_memory(supervisor_memory, conversation_log, clean_c_code, iteration_count)
        print_colored(f"\nSupervisor memory status (entries: {len(supervisor_memory)})", "1;33")
    
    # Create the prompt
    supervisor_template = supervisor_prompts.get_supervisor_template()
    supervisor_prompt = PromptTemplate.from_template(supervisor_template)

    # Initialize model for supervisor
    supervisor_llm = initialize_llm(model_source)
    supervisor_llm.temperature = 0.6
    supervisor_tools = [ExceptionTool()]
    
    # Create the ReAct agent instead of OpenAI tools agent
    supervisor_agent = create_react_agent(supervisor_llm, supervisor_tools, supervisor_prompt)
    
    supervisor_executor = AgentExecutor(
        agent=supervisor_agent,
        tools=supervisor_tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    validator_messages = [msg for msg in messages if hasattr(msg, 'name') and msg.name == "Validator"]
    has_validated_parser = len(validator_messages) > 0 and generator_code is not None and iteration_count > 0

    f = open(log_file, 'a', encoding="utf-8")

    if has_validated_parser:
        last_message = validator_messages[-1].content

        memory_entry = {
            'code': clean_c_code,
            'iteration': iteration_count,
            'validator_assessment': last_message
        }
        supervisor_memory.append(memory_entry)
        
        code_satisfaction = multi_agent.get_satisfaction(last_message)
        compilation_status = multi_agent.get_compilation_status(last_message)
        print_colored(f"\nStored new parser in memory (satisfactory: {code_satisfaction}, compilation: {compilation_status})", "1;36")
        
        prompt = supervisor_prompts.get_supervisor_input_validated()
        result = supervisor_executor.invoke({
            "input": user_request,
            "conversation_history": conversation_log,
            "prompt": prompt,
            "c_code": clean_c_code,
            "validator_assessment": last_message,
            "code_satisfaction": code_satisfaction,
            "compilation_status": compilation_status,
            "code_satisfaction_instructions": multi_agent.get_satisfaction_instructions(last_message)
        })
        supervisor_response = result["output"]
        
        if state.get("system_metrics"):
            state["system_metrics"].complete_round()
        
        purpose = "providing final parser"
        generator_specs_new = state.get("generator_specs")
        generator_code_new = generator_code
        iteration_count_new = 0
        next_step = "FINISH"
        parser_mode = False
    else:
        prompt = supervisor_prompts.get_supervisor_input_actions()
        result = supervisor_executor.invoke({
            "input": user_request,
            "conversation_history": conversation_log,
            "prompt": prompt
        })
        action = result["output"].strip()
        
        log(f, f"Supervisor action choosen: {action}", colors.CYAN, bold=True)

        most_recent_parser = supervisor_memory[-1] if supervisor_memory else None
        
        if action == "GENERATE_PARSER":
            prompt = supervisor_prompts.get_supervisor_input_generate_parser()
            result = supervisor_executor.invoke({
                "input": user_request,
                "conversation_history": conversation_log,
                "prompt": prompt,
                "requirements": multi_agent.get_parser_requirements()
            })
            supervisor_response = result["output"]
            
            purpose = "creating detailed specifications"
            generator_specs_new = supervisor_response
            generator_code_new = None
            iteration_count_new = 0
            next_step = "Generator"
            parser_mode = True
        elif action == "CORRECT_ERROR" and most_recent_parser:
            most_recent_assessment = most_recent_parser["validator_assessment"]
            prompt = supervisor_prompts.get_supervisor_input_correct_error()
            result = supervisor_executor.invoke({
                "input": user_request,
                "conversation_history": conversation_log,
                "prompt": prompt,
                "c_code": most_recent_parser["code"],
                "validator_assessment": multi_agent.get_assessment(most_recent_assessment),
                "code_satisfaction": multi_agent.get_satisfaction(most_recent_assessment),
                "compilation_status": multi_agent.get_compilation_status(most_recent_assessment)
            })
            supervisor_response = result["output"]
            
            purpose = "creating updated specifications"
            generator_specs_new = supervisor_response
            generator_code_new = None
            iteration_count_new = 0
            next_step = "Generator"
            parser_mode = True
        elif action == "ASSESS_CODE" and most_recent_parser:
            most_recent_assessment = most_recent_parser["validator_assessment"]
            prompt = supervisor_prompts.get_supervisor_input_assess_code(user_request)
            result = supervisor_executor.invoke({
                "input": user_request,
                "conversation_history": conversation_log,
                "prompt": prompt,
                "c_code": most_recent_parser["code"],
                "validator_assessment": multi_agent.get_assessment(most_recent_assessment),
                "code_satisfaction": multi_agent.get_satisfaction(most_recent_assessment),
                "compilation_status": multi_agent.get_compilation_status(most_recent_assessment)
            })
            supervisor_response = result["output"]
            
            if state.get("system_metrics"):
                state["system_metrics"].complete_round()
            
            purpose = "providing code assessment"
            generator_specs_new = state.get("generator_specs")
            generator_code_new = generator_code
            iteration_count_new = iteration_count
            next_step = "FINISH"
            parser_mode = False
        else:
            # GENERAL_CONVERSATION
            keywords = ["memory", "remember", "previous", "code", "generated", "parser"]
            is_asking_about_memory = most_recent_parser and any(keyword in user_request.lower() for keyword in keywords)
            if is_asking_about_memory:
                most_recent_assessment = most_recent_parser["validator_assessment"]
                prompt = supervisor_prompts.get_supervisor_input_general_conversation_1()
                input = {
                    "input": user_request,
                    "conversation_history": conversation_log,
                    "prompt": prompt,
                    "c_code": most_recent_parser["code"],
                    "code_satisfaction": multi_agent.get_satisfaction(most_recent_assessment),
                    "compilation_status": multi_agent.get_compilation_status(most_recent_assessment)
                }
            else:
                prompt = supervisor_prompts.get_supervisor_input_general_conversation_2()
                input = {
                    "input": user_request,
                    "conversation_history": conversation_log,
                    "prompt": prompt
                }
            
            result = supervisor_executor.invoke(input)
            supervisor_response = result["output"]
            
            if state.get("system_metrics"):
                state["system_metrics"].complete_round()
            
            purpose = "conversation"
            generator_specs_new = state.get("generator_specs")
            generator_code_new = generator_code
            iteration_count_new = iteration_count
            next_step = "FINISH"
            parser_mode = False

    log(f, f"Supervisor ({purpose}):", colors.BLUE, bold=True)
    log(f, supervisor_response)
    f.close()
    
    return {
        "messages": [AIMessage(content=supervisor_response, name="Supervisor")],
        "user_request": user_request,
        "supervisor_memory": supervisor_memory,
        "generator_specs": generator_specs_new,
        "generator_code": generator_code_new,
        "iteration_count": iteration_count_new,
        "max_iterations": state["max_iterations"],
        "model_source": model_source,
        "session_dir": session_dir,
        "log_file": log_file,
        "next_step": next_step,
        "parser_mode": parser_mode,
        "system_metrics": state.get("system_metrics")
    }