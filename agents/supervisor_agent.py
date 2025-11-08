import re
from datetime import datetime
from pathlib import Path
from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from models import AgentState, ExceptionTool
from utils.general import requirements, print_colored, extract_c_code, initialize_llm



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
        code_exists = False
        for entry in supervisor_memory:
            if entry.get("code") == generator_code:
                code_exists = True
                if conversation_log:
                    assessment = extract_validator_assessment(conversation_log, iteration_count)
                    if assessment:
                        entry["validator_assessment"] = assessment
                        entry["is_satisfactory"] = "satisfactory" in assessment.lower() and "not satisfactory" not in assessment.lower()
                break
        
        if not code_exists:
            is_satisfactory = False
            validator_assessment = None
            
            if conversation_log:
                validator_assessment = extract_validator_assessment(conversation_log, iteration_count)
                if validator_assessment:
                    is_satisfactory = "satisfactory" in validator_assessment.lower() and "not satisfactory" not in validator_assessment.lower()
            
            new_entry = {
                'code': generator_code,
                'specs': state["generator_specs"],
                'timestamp': datetime.now().isoformat(),
                'iteration': state["iteration_count"],
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
    
    supervisor_llm = initialize_llm(model_source)
    supervisor_llm.temperature = 0.6
    supervisor_tools = [ExceptionTool()]
    
    validator_messages = [msg for msg in messages if hasattr(msg, 'name') and msg.name == "Validator"]
    has_validated_parser = len(validator_messages) > 0 and state["iteration_count"] > 0
    
    supervisor_template = """<role>
You are a helpful C programming expert. You are a supervisor managing the process of creating parser functions.
</role>

<main_directive>
Since there are limits to how much code the generator can generate, keep the structure of the parser function simple, short and focused on the core functionality.
</main_directive>

<available_tools>
You have access to these tools: {tools}
Tool names: {tool_names}
</available_tools>

<format_instructions>
Use the following format:
Question: the input question.
Thought: think about what to do.
Final Answer: the final answer to the original question.
</format_instructions>

User request: {input}
Context: {conversation_context}

{agent_scratchpad}
"""
    
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
    
    if has_validated_parser:
        latest_validator_message = validator_messages[-1].content
        is_satisfactory = "satisfactory" in latest_validator_message.lower() and "not satisfactory" not in latest_validator_message.lower()
        
        compilation_status = "Compilation successful" if "compilation successful" in latest_validator_message.lower() else "Compilation failed"
        
        final_clean_c_code = extract_c_code(state["generator_code"])
        memory_entry = {
            'code': state["generator_code"],
            'clean_code': final_clean_c_code,
            'specs': state["generator_specs"],
            'timestamp': datetime.now().isoformat(),
            'iteration': state["iteration_count"],
            'validator_assessment': latest_validator_message,
            'is_satisfactory': is_satisfactory,
            'compilation_status': compilation_status
        }
        
        supervisor_memory.append(memory_entry)
        
        most_recent_parser = memory_entry
            
        print_colored(f"\nStored new parser in memory (satisfactory: {is_satisfactory}, compilation: {compilation_status})", "1;36")
        
        response_prompt = f"""You are a helpful C programming expert. The user requested a parser function, and we've generated one for them.

The user's original request was: {user_request}

The generated parser code is:
```c
{state["generator_code"]}
```

The validator's assessment: {latest_validator_message}

Is the code satisfactory? {"Yes" if is_satisfactory else "No"}
Compilation status: {compilation_status}

Please provide a friendly response to the user that:
1. Acknowledges their request
2. Presents the generated C parser code
3. {"" if is_satisfactory else "IMPORTANTLY, mentions that the code is NOT FULLY SATISFACTORY according to the validator, and briefly explains why"}
4. Gives a brief explanation of what the parser does and how it works
5. Explicitly mentions the compilation status
6. Mentions they can ask for clarification or report any issues they find

Keep your explanation concise and user-friendly.
"""
        
        supervisor_result = supervisor_executor.invoke({
            "input": response_prompt,
            "conversation_context": conversation_log[-2000:] if len(conversation_log) > 2000 else conversation_log,
            "tools": supervisor_tools,
            "tool_names": [tool.name for tool in supervisor_tools]
        })
        supervisor_response = supervisor_result["output"]
        
        print_colored("\nSupervisor (responding with final parser):", "1;34")
        print(supervisor_response)
        
        with open(log_file, 'a') as f:
            f.write("Supervisor (providing final parser):\n")
            f.write(supervisor_response + "\n\n")
        
        if state.get("system_metrics"):
            state["system_metrics"].complete_round()
        
        return {
            "messages": [AIMessage(content=supervisor_response, name="Supervisor")],
            "user_request": user_request,
            "supervisor_memory": supervisor_memory,
            "generator_specs": state["generator_specs"],
            "generator_code": state["generator_code"],
            "iteration_count": 0,  # reset for next time
            "max_iterations": state["max_iterations"],
            "model_source": model_source,
            "next_step": "FINISH",
            "parser_mode": False,  # exit parser mode after delivering result
            "session_dir": session_dir,
            "log_file": log_file,
            "system_metrics": state.get("system_metrics")
        }
    
    else:
        context_prompt = f"""Based on the user's request and our conversation history, determine what action I should take.
Respond with ONLY ONE of these actions (and nothing else):
1. "GENERATE_PARSER" - if the user wants me to create a new parser
2. "CORRECT_ERROR" - if the user is reporting issues with previously generated code
3. "ASSESS_CODE" - if the user wants me to evaluate previously generated code or is asking to see previously generated code
4. "GENERAL_CONVERSATION" - for general questions or conversations

User's request: "{user_request}"

Recent conversation context: 
{conversation_log[-2000:] if len(conversation_log) > 2000 else conversation_log}
"""

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
        
        parser_mode = action in ["GENERATE_PARSER", "CORRECT_ERROR"]
        
        if action == "GENERATE_PARSER":
            parser_prompt = f"""Your task is to take the user's request and convert it into a detailed, specific prompt for a C parser function generator.

The parser function must follow these requirements:
{requirements}

The prompt you create should include:
1. Clear identification of what kind of parser is being requested
2. What data structures will be needed
3. What state transitions and parsing logic should be implemented
4. How error cases should be handled
5. What component functions will be needed
6. Memory management strategy

User request: {user_request}

Create a detailed prompt for the generator.
"""

            specs_result = supervisor_executor.invoke({
                "input": parser_prompt,
                "conversation_context": conversation_log[-2000:] if len(conversation_log) > 2000 else conversation_log,
                "tools": supervisor_tools,
                "tool_names": [tool.name for tool in supervisor_tools]
            })
            supervisor_response = specs_result["output"]
            
            print_colored("\nSupervisor (creating detailed specifications):", "1;34")
            print(supervisor_response)
            
            with open(log_file, 'a') as f:
                f.write("Supervisor (creating detailed specifications):\n")
                f.write(supervisor_response + "\n\n")
            
            return {
                "messages": [AIMessage(content=supervisor_response, name="Supervisor")],
                "user_request": user_request,
                "supervisor_memory": supervisor_memory,
                "generator_specs": supervisor_response,
                "generator_code": None,
                "iteration_count": 0,
                "max_iterations": state["max_iterations"],
                "model_source": model_source,
                "session_dir": session_dir,
                "log_file": log_file,
                "next_step": "Generator",
                "parser_mode": True,
                "system_metrics": state.get("system_metrics")
            }
            
        elif action == "CORRECT_ERROR" and most_recent_parser:
            correction_prompt = f"""Create detailed specifications for updating the parser to address these issues.
Be specific about what changes need to be made and why.

Original user request: {user_request}

Previous parser code:
```c
{most_recent_parser['code']}
```

Previous validator assessment: {most_recent_parser.get('validator_assessment', 'Not available')}
Compilation status: {most_recent_parser.get('compilation_status', 'Unknown')}
Was code satisfactory: {"Yes" if most_recent_parser.get('is_satisfactory', False) else "No"}

User reported issues or requested changes: {user_request}

Conversation history (for context):
{conversation_log[-2000:] if len(conversation_log) > 2000 else conversation_log}
"""

            correction_result = supervisor_executor.invoke({
                "input": correction_prompt,
                "conversation_context": "",
                "tools": supervisor_tools,
                "tool_names": [tool.name for tool in supervisor_tools]
            })
            supervisor_response = correction_result["output"]
            
            print_colored("\nSupervisor (creating updated specifications):", "1;34")
            print(supervisor_response)
            
            with open(log_file, 'a') as f:
                f.write("Supervisor (creating updated specifications):\n")
                f.write(supervisor_response + "\n\n")
            
            return {
                "messages": [AIMessage(content=supervisor_response, name="Supervisor")],
                "user_request": user_request,
                "supervisor_memory": supervisor_memory,
                "generator_specs": supervisor_response,
                "generator_code": None,
                "iteration_count": 0,
                "max_iterations": state["max_iterations"],
                "model_source": model_source,
                "session_dir": session_dir,
                "log_file": log_file,
                "next_step": "Generator",
                "parser_mode": True,
                "system_metrics": state.get("system_metrics")
            }
            
        elif action == "ASSESS_CODE" and most_recent_parser:
            is_asking_for_code = any(keyword in user_request.lower() for keyword in ["show", "code", "generated", "memory", "what was the", "see the", "provide the"])
            
            if is_asking_for_code:
                show_code_prompt = f"""The user is asking about previously generated code.

User request: {user_request}

The complete most recently generated code is:
```c
{most_recent_parser['code']}
```

Validator's assessment: {most_recent_parser.get('validator_assessment', 'No formal assessment available')}
Compilation status: {most_recent_parser.get('compilation_status', 'Unknown')}
Was code satisfactory: {"Yes" if most_recent_parser.get('is_satisfactory', False) else "No"}

Create a response that highlights:
1. Acknowledges their request
2. Summarizes what the parser does and its key components (3-5 sentences)
3. Explicitly mentions the validator's assessment
4. Clearly states whether the code compiles successfully
5. States whether the code was deemed satisfactory or not

Focus on explaining the code's functionality rather than showing the entire codebase. Only provide specific code snippets if the user explicitly asks for them.
"""

                code_result = supervisor_executor.invoke({
                    "input": show_code_prompt,
                    "conversation_context": "",
                    "tools": supervisor_tools,
                    "tool_names": [tool.name for tool in supervisor_tools]
                })
                supervisor_response = code_result["output"]

            else:
                assess_prompt = f"""The user is asking about the quality or validation status of previously generated code.

User request: {user_request}

The most recently generated code was:
```c
{most_recent_parser['code']}
```

Validator's assessment: {most_recent_parser.get('validator_assessment', 'No formal assessment available')}
Compilation status: {most_recent_parser.get('compilation_status', 'Unknown')}
Was code satisfactory: {"Yes" if most_recent_parser.get('is_satisfactory', False) else "No"}

Create a comprehensive response that:
1. Acknowledges their question
2. Provides a brief summary of what the parser does (2-3 sentences)
3. CLEARLY explains the validator's assessment
4. EXPLICITLY states whether the code compiles successfully
5. CLEARLY states whether the code was deemed satisfactory or not
6. Mentions key strengths or limitations

Keep your explanation concise and conversational, focusing on the overall assessment rather than providing the entire code.
"""

                assess_result = supervisor_executor.invoke({
                    "input": assess_prompt,
                    "conversation_context": "",
                    "tools": supervisor_tools,
                    "tool_names": [tool.name for tool in supervisor_tools]
                })
                supervisor_response = assess_result["output"]
            
            print_colored("\nSupervisor (providing code assessment):", "1;34")
            print(supervisor_response)
            
            with open(log_file, 'a') as f:
                f.write("Supervisor (providing code assessment):\n")
                f.write(supervisor_response + "\n\n")
            
            if state.get("system_metrics"):
                state["system_metrics"].complete_round()
            
            return {
                "messages": [AIMessage(content=supervisor_response, name="Supervisor")],
                "user_request": user_request,
                "supervisor_memory": supervisor_memory,
                "generator_specs": state.get("generator_specs"),
                "generator_code": state.get("generator_code"),
                "iteration_count": state["iteration_count"],
                "max_iterations": state["max_iterations"],
                "model_source": model_source,
                "session_dir": session_dir,
                "log_file": log_file,
                "next_step": "FINISH",
                "parser_mode": False,
                "system_metrics": state.get("system_metrics")
            }
            
        else:  # GENERAL_CONVERSATION or fallback
            is_asking_about_memory = (any(keyword in user_request.lower() for keyword in ["memory", "remember", "previous", "code", "generated", "parser"]) 
                                     and most_recent_parser is not None)
            
            if is_asking_about_memory:
                memory_info = (
                    f"I have a recently generated parser from {most_recent_parser.get('timestamp')}\n"
                    f"- Satisfactory: {most_recent_parser.get('is_satisfactory', 'Unknown')}\n"
                    f"- Compilation: {most_recent_parser.get('compilation_status', 'Unknown')}\n"
                )
                
                memory_prompt = f"""Respond to the user mentioning that you have a recently generated parser in memory. Focus ONLY on the most recent parser. Remember to explicitly mention:
1. Whether the code was satisfactory according to the validator
2. Whether the code compiled successfully
3. A brief summary of what the parser does (if you know this information)

Let them know they can ask to see a summary of the code by saying something like "summarize the code" or "what did the parser do?"

User: {user_request}

Recent conversation history (for context):
{conversation_log[-2000:] if len(conversation_log) > 2000 else conversation_log}

Information about your most recent parser:
{memory_info}
"""

                memory_result = supervisor_executor.invoke({
                    "input": memory_prompt,
                    "conversation_context": "",
                    "tools": supervisor_tools,
                    "tool_names": [tool.name for tool in supervisor_tools]
                })
                supervisor_response = memory_result["output"]
            else:
                conversation_prompt = f"""If they're asking about parsers, you can offer to generate a C parser for them by responding to their specific needs.
If they're asking about previous code you've generated, refer ONLY to your most recent parser.
Otherwise, provide a helpful, concise response that addresses their question.

Respond in a conversational, friendly tone.

User: {user_request}

Recent conversation history (for context):
{conversation_log[-2000:] if len(conversation_log) > 2000 else conversation_log}
"""

                conversation_result = supervisor_executor.invoke({
                    "input": conversation_prompt,
                    "conversation_context": "",
                    "tools": supervisor_tools,
                    "tool_names": [tool.name for tool in supervisor_tools]
                })
                supervisor_response = conversation_result["output"]
            
            print_colored("\nSupervisor (conversation):", "1;34")
            print(supervisor_response)
            
            with open(log_file, 'a') as f:
                f.write("Supervisor (conversation):\n")
                f.write(supervisor_response + "\n\n")
            
            if state.get("system_metrics"):
                state["system_metrics"].complete_round()
            
            return {
                "messages": [AIMessage(content=supervisor_response, name="Supervisor")],
                "user_request": user_request,
                "supervisor_memory": supervisor_memory,
                "generator_specs": state.get("generator_specs"),
                "generator_code": state.get("generator_code"),
                "iteration_count": state["iteration_count"],
                "max_iterations": state["max_iterations"],
                "model_source": model_source,
                "session_dir": session_dir,
                "log_file": log_file,
                "next_step": "FINISH",
                "parser_mode": False,
                "system_metrics": state.get("system_metrics")
            }