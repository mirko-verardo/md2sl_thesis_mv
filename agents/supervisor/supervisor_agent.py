import re
from datetime import datetime
from pathlib import Path
from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from models import AgentState, ExceptionTool
from utils.general import print_colored, extract_c_code, initialize_llm
from agents.supervisor import supervisor_prompts



def read_conversation_log(log_file: Path | None) -> str:
    """Read the entire conversation log to provide context for the supervisor."""
    try:
        with open(log_file, 'r') as f:
            return f.read()
    except Exception as e:
        print_colored(f"Warning: Could not read log file: {e}", "1;33")
        return ""

def extract_validator_assessment(conversation_log: str, iteration_count: int) -> str | None:
    """Extract the validator's assessment from the conversation log."""

    # search for a specific pattern
    pattern = rf"Validator assessment \(Iteration {iteration_count}/{iteration_count}\):\n(.*?)(?:\n\n|\Z)"
    matches = re.findall(pattern, conversation_log, re.DOTALL)
    if matches:
        return matches[-1].strip()
    
    # search for a more general one
    pattern = r"Validator assessment.*?:\n(.*?)(?:\n\n|\Z)"
    matches = re.findall(pattern, conversation_log, re.DOTALL)
    if matches:
        return matches[-1].strip()
        
    return None

def update_supervisor_memory(state: AgentState, conversation_log: str) -> list[dict]:
    """Update the supervisor's memory with information from the conversation log."""
    supervisor_memory = state["supervisor_memory"]
    generator_code = state["generator_code"]
    iteration_count = state["iteration_count"]
    
    if generator_code and iteration_count > 0:
        # get assessment
        validator_assessment = None
        is_satisfactory = False
        if conversation_log:
            validator_assessment = extract_validator_assessment(conversation_log, iteration_count)
            if validator_assessment:
                is_satisfactory = "satisfactory" in validator_assessment.lower() and "not satisfactory" not in validator_assessment.lower()

        # update assessment if code exists
        code_exists = False
        for entry in supervisor_memory:
            if entry.get("code") == generator_code:
                code_exists = True
                if validator_assessment:
                    entry["validator_assessment"] = validator_assessment
                    entry["is_satisfactory"] = is_satisfactory
                break
        
        # else add assessment
        if not code_exists:
            new_entry = {
                'code': generator_code,
                'specs': state["generator_specs"],
                'timestamp': datetime.now().isoformat(),
                'iteration': iteration_count,
                'validator_assessment': validator_assessment,
                'is_satisfactory': is_satisfactory
            }
            supervisor_memory.append(new_entry)
    
    print_colored(f"\nSupervisor memory status (entries: {len(supervisor_memory)})", "1;33")
    
    return supervisor_memory

def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor agent that converses with the user and manages the parser generation process."""
    messages = state["messages"]
    user_request = state["user_request"]
    model_source = state["model_source"]
    session_dir = state["session_dir"]
    log_file = state["log_file"]
    supervisor_memory = state["supervisor_memory"]
    
    conversation_log = read_conversation_log(log_file)
    updated_memory = update_supervisor_memory(state, conversation_log)
    supervisor_memory = updated_memory if updated_memory is not None else supervisor_memory
    most_recent_parser = supervisor_memory[-1] if supervisor_memory else None
    conversation_context = conversation_log[-2000:] if len(conversation_log) > 2000 else conversation_log
    
    supervisor_template = supervisor_prompts.get_supervisor_template()
    
    # Initialize model for supervisor
    supervisor_llm = initialize_llm(model_source)
    supervisor_llm.temperature = 0.6
    supervisor_tools = [ExceptionTool()]
    
    # Create a prompt using PromptTemplate.from_template
    supervisor_prompt = PromptTemplate.from_template(supervisor_template)
    
    # Create the ReAct agent instead of OpenAI tools agent
    supervisor_agent = create_react_agent(supervisor_llm, supervisor_tools, supervisor_prompt)
    
    supervisor_executor = AgentExecutor(
        agent=supervisor_agent,
        tools=supervisor_tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    validator_messages = [msg for msg in messages if hasattr(msg, 'name') and msg.name == "Validator"]
    has_validated_parser = len(validator_messages) > 0 and state["iteration_count"] > 0

    if has_validated_parser:
        last_message = validator_messages[-1].content
        is_satisfactory = "satisfactory" in last_message.lower() and "not satisfactory" not in last_message.lower()
        compilation_status = "Compilation successful" if "compilation successful" in last_message.lower() else "Compilation failed"
        
        c_code = state["generator_code"]
        final_clean_c_code = extract_c_code(c_code)
        memory_entry = {
            'code': c_code,
            'clean_code': final_clean_c_code,
            'specs': state["generator_specs"],
            'timestamp': datetime.now().isoformat(),
            'iteration': state["iteration_count"],
            'validator_assessment': last_message,
            'is_satisfactory': is_satisfactory,
            'compilation_status': compilation_status
        }
        
        supervisor_memory.append(memory_entry)
        
        #most_recent_parser = memory_entry
            
        print_colored(f"\nStored new parser in memory (satisfactory: {is_satisfactory}, compilation: {compilation_status})", "1;36")
        
        response_prompt = supervisor_prompts.get_supervisor_input_validated(user_request, c_code, last_message, is_satisfactory, compilation_status)
        
        supervisor_result = supervisor_executor.invoke({
            "input": response_prompt,
            "conversation_context": conversation_context,
            "tools": supervisor_tools,
            "tool_names": [tool.name for tool in supervisor_tools]
        })
        supervisor_response = supervisor_result["output"]
        
        if state.get("system_metrics"):
            state["system_metrics"].complete_round()
        
        purpose = "providing final parser"
        generator_specs = state.get("generator_specs")
        generator_code = state.get("generator_code")
        iteration_count = 0
        next_step = "FINISH"
        parser_mode = False
    else:
        context_prompt = supervisor_prompts.get_supervisor_input_actions(user_request, conversation_context)

        action_result = supervisor_executor.invoke({
            "input": context_prompt,
            "conversation_context": "",
            "tools": supervisor_tools,
            "tool_names": [tool.name for tool in supervisor_tools]
        })
        action = action_result["output"].strip()
        
        print_colored(f"\nSupervisor determined action: {action}", "1;36")
        with open(log_file, 'a') as f:
            f.write(f"Supervisor action decision: {action}\n\n")
        
        if action == "GENERATE_PARSER":
            parser_prompt = supervisor_prompts.get_supervisor_input_generate_parser(user_request)

            specs_result = supervisor_executor.invoke({
                "input": parser_prompt,
                "conversation_context": conversation_context,
                "tools": supervisor_tools,
                "tool_names": [tool.name for tool in supervisor_tools]
            })
            supervisor_response = specs_result["output"]
            
            purpose = "creating detailed specifications"
            generator_specs = supervisor_response
            generator_code = None
            iteration_count = 0
            next_step = "Generator"
            parser_mode = True            
        elif action == "CORRECT_ERROR" and most_recent_parser:
            correction_prompt = supervisor_prompts.get_supervisor_input_correct_error(user_request, conversation_context, most_recent_parser)

            correction_result = supervisor_executor.invoke({
                "input": correction_prompt,
                "conversation_context": "",
                "tools": supervisor_tools,
                "tool_names": [tool.name for tool in supervisor_tools]
            })
            supervisor_response = correction_result["output"]
            
            purpose = "creating updated specifications"
            generator_specs = supervisor_response
            generator_code = None
            iteration_count = 0
            next_step = "Generator"
            parser_mode = True
        elif action == "ASSESS_CODE" and most_recent_parser:
            assess_prompt = supervisor_prompts.get_supervisor_input_assess_code(user_request, most_recent_parser)
            assess_result = supervisor_executor.invoke({
                "input": assess_prompt,
                "conversation_context": "",
                "tools": supervisor_tools,
                "tool_names": [tool.name for tool in supervisor_tools]
            })
            supervisor_response = assess_result["output"]
            
            if state.get("system_metrics"):
                state["system_metrics"].complete_round()
            
            purpose = "providing code assessment"
            generator_specs = state.get("generator_specs")
            generator_code = state.get("generator_code")
            iteration_count = state["iteration_count"]
            next_step = "FINISH"
            parser_mode = False
        else:
            # GENERAL_CONVERSATION
            prompt = supervisor_prompts.get_supervisor_input_general_conversation(user_request, conversation_context, most_recent_parser)
            result = supervisor_executor.invoke({
                "input": prompt,
                "conversation_context": "",
                "tools": supervisor_tools,
                "tool_names": [tool.name for tool in supervisor_tools]
            })
            supervisor_response = result["output"]
            
            if state.get("system_metrics"):
                state["system_metrics"].complete_round()
            
            purpose = "conversation"
            generator_specs = state.get("generator_specs")
            generator_code = state.get("generator_code")
            iteration_count = state["iteration_count"]
            next_step = "FINISH"
            parser_mode = False
        
    print_colored(f"\nSupervisor ({purpose}):", "1;34")
    print(supervisor_response)
    
    with open(log_file, 'a') as f:
        f.write(f"Supervisor ({purpose}):\n")
        f.write(supervisor_response + "\n\n")
    
    return {
        "messages": [AIMessage(content=supervisor_response, name="Supervisor")],
        "user_request": user_request,
        "supervisor_memory": supervisor_memory,
        "generator_specs": generator_specs,
        "generator_code": generator_code,
        "iteration_count": iteration_count,
        "max_iterations": state["max_iterations"],
        "model_source": model_source,
        "session_dir": session_dir,
        "log_file": log_file,
        "next_step": next_step,
        "parser_mode": parser_mode,
        "system_metrics": state.get("system_metrics")
    }