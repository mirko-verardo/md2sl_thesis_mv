from datetime import datetime
from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from models import AgentState, ExceptionTool
from utils.general import requirements, print_colored, extract_c_code, compile_c_code, initialize_llm



def validator_node(state: AgentState) -> AgentState:
    """Validator agent that evaluates parser code."""
    #messages = state["messages"]
    iteration_count = state["iteration_count"]
    max_iterations = state["max_iterations"]
    generator_code = state["generator_code"]
    user_request = state["user_request"]
    session_dir = state["session_dir"]
    log_file = state["log_file"]
    model_source = state["model_source"]
    
    # Extract clean c code
    clean_c_code = extract_c_code(generator_code)
    if not clean_c_code:
        print_colored("Warning: Could not extract clean C code, using original code", "1;33")
        clean_c_code = generator_code
    
    # Save c code to temporary file for compilation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    c_file_name = f"parser_{timestamp}.c"
    c_file_path = session_dir / c_file_name
    
    with open(c_file_path, 'w') as f:
        f.write(clean_c_code)
    
    print_colored(f"\nSaved C code to: {c_file_path} for compilation testing", "1;36")
    
    # Compile the code
    print_colored("\n--- Testing Compilation ---", "1;33")
    compilation_result = compile_c_code(c_file_path)
    
    # Record parser validation in metrics
    if state.get("system_metrics"):
        state["system_metrics"].record_parser_validation(c_file_name, compilation_result['success'])

    # Log compilation results
    with open(log_file, 'a') as f:
        f.write(f"--- Compilation Test Results (Iteration {iteration_count}) ---\n")
        f.write(f"Compilation {'successful' if compilation_result['success'] else 'failed'}\n")
        if compilation_result['stderr']:
            f.write(f"Errors:\n{compilation_result['stderr']}\n\n")
    
    # Print compilation status
    if compilation_result['success']:
        print_colored("Compilation successful!", "1;32")
        compilation_status = "✅ Compilation successful!"
    else:
        print_colored("Compilation failed. Errors:", "1;31")
        print(compilation_result['stderr'])
        compilation_status = "❌ Compilation failed with the following errors:\n" + compilation_result['stderr']
    
    # Initialize model for validator
    validator_llm = initialize_llm(model_source)
    validator_llm.temperature = 0.4
    validator_tools = [ExceptionTool()]
    
    # Escape curly braces in variable strings
    escaped_generator_code = generator_code.replace("{", "{{").replace("}", "}}")
    escaped_compilation_status = compilation_status.replace("{", "{{").replace("}", "}}")
    escaped_generator_specs = state["generator_specs"].replace("{", "{{").replace("}", "}}") if state["generator_specs"] else ""
    
    # Validator's template with ReAct format
    validator_template = f"""<role>
You are a specialized C programming validator that evaluates parser code against strict requirements.
</role>

<available_tools>
You have access to these tools: {{tools}}
Tool names: {{tool_names}}
</available_tools>

<parser_requirements>
Review this C parser code to determine if it meets the following requirements:
{requirements}
</parser_requirements>

<specifications>
Additionally, it must meet these specific specifications:
{escaped_generator_specs}
</specifications>

<code_to_review>
```c
{escaped_generator_code}
```
</code_to_review>

<compilation_result>
Compilation result:
{escaped_compilation_status}
</compilation_result>

<validation_process>
Validate the code following these steps:
1. Evaluate whether the code compiles correctly. If it doesn't compile, this is a critical issue that must be addressed.
2. Evaluate each general requirement and whether the code satisfies it. Remember that requirement #6 (Composition) is only necessary if and only if one of requirements 1-5 is not satisfied. If all the requirements 1-5 are met, requirement #6 becomes optional.
3. Check if the code implements all the specifications from the supervisor.
4. Verify that the code has COMPLETE IMPLEMENTATIONS with no placeholders, todos or ellipses (...). Every single function must be fully implemented. There should be no comments like "// ... (Implementation details)" OR "// ... (Full implementation below)" or references to previous code developed. No part of the code should be skipped.
5. Ignore the readability, the efficiency and the maintainability of the generated code.
</validation_process>

<task>
Then provide your final verdict: Is the code SATISFACTORY or NOT SATISFACTORY?

A code is NOT SATISFACTORY if:
1. It fails to compile
2. It doesn't meet all the required specifications
3. It contains placeholders or incomplete implementations
If the code is NOT SATISFACTORY, provide specific feedback with details on what needs to be improved and briefly explain how.

This is iteration {iteration_count} of maximum {max_iterations}.
</task>

<format_instructions>
Use the following format:
Question: the input question.
Thought: think about what to do.
Final Answer: the final answer to the original question.
</format_instructions>

Evaluate the code based on the requirements and provide your assessment.

{{agent_scratchpad}}
"""

    # Create a prompt using PromptTemplate.from_template
    validator_prompt = PromptTemplate.from_template(validator_template)
    
    # Create the ReAct agent instead of OpenAI tools agent
    validator_agent = create_react_agent(validator_llm, validator_tools, validator_prompt)
    
    validator_executor = AgentExecutor(
        agent=validator_agent,
        tools=validator_tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    # Invoke the agent
    try:
        validator_result = validator_executor.invoke({
            "tools": validator_tools,
            "tool_names": [tool.name for tool in validator_tools],
            "agent_scratchpad": []
        })
        validator_response = validator_result["output"]
    except Exception as e:
        print_colored(f"Error during validator execution: {str(e)}", "1;31")
        validator_response = f"Error occurred during code validation: {str(e)}\n\nPlease try again."
    
    print_colored(f"\nValidator assessment (Iteration {iteration_count}/{max_iterations}):", "1;34")
    print(validator_response)
    
    # Log the validator's assessment
    with open(log_file, 'a') as f:
        f.write(f"Validator assessment (Iteration {iteration_count}/{max_iterations}):\n")
        f.write(validator_response + "\n\n")

    # Check if code has been compiled with success
    is_compiled = compilation_result['success']
    # Check if code is satisfactory
    is_satisfactory = is_compiled and (
        "satisfactory" in validator_response.lower()) and (
        "not satisfactory" not in validator_response.lower()
    )
    # Check if code is the final iteration
    is_final_iteration = iteration_count >= max_iterations

    if is_satisfactory:
        # end
        next_node = "Supervisor"
        feedback_message = f"The parser implementation has been validated and is SATISFACTORY. It compiles successfully and meets all requirements. Here's the final assessment:\n\n{validator_response}"
    elif is_final_iteration:
        # end
        next_node = "Supervisor"
        feedback_message = f"After {iteration_count} iterations, this is the best parser implementation available. While it may not be perfect, it should serve as a good starting point. Here's the assessment:\n\n{validator_response}\n\n{compilation_status}"
    elif is_compiled:
        # continue
        next_node = "Generator"
        feedback_message = f"The parser implementation needs improvements:\n\n{validator_response}"
    else:
        # continue
        next_node = "Generator"
        feedback_message = f"The parser implementation needs improvements. It failed to compile with the following errors:\n\n{compilation_result['stderr']}\n\nAdditional feedback:\n{validator_response}"
    
    print_colored(f"\nValidator sending FEEDBACK to {next_node}", "1;33")
        
    return {
        #"messages": messages + [AIMessage(content=feedback_message, name="Validator")],
        "messages": [AIMessage(content=feedback_message, name="Validator")],
        "user_request": user_request,
        "supervisor_memory": state["supervisor_memory"],
        "generator_specs": state["generator_specs"],
        "generator_code": generator_code,
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
        "model_source": state["model_source"],
        "session_dir": session_dir,
        "log_file": log_file,
        "next_step": next_node,
        "parser_mode": state["parser_mode"],
        "system_metrics": state.get("system_metrics")
    }