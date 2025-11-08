from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from models import AgentState, CompilationCheckTool
from utils.general import requirements, print_colored, initialize_llm



def generator_node(state: AgentState) -> AgentState:
    """Generator agent that creates C code."""
    messages = state["messages"]
    generator_specs = state["generator_specs"]
    user_request = state["user_request"]
    iteration_count = state["iteration_count"]
    model_source = state["model_source"]
    session_dir = state["session_dir"]
    log_file = state["log_file"]
    
    if state.get("system_metrics"):
        if state["iteration_count"] == 0 or state["next_step"] == "Generator":
            state["system_metrics"].increment_generator_validator_interaction()
    
    validator_messages = [msg for msg in messages if hasattr(msg, 'name') and msg.name == "Validator"]
    has_feedback = len(validator_messages) > 0 and iteration_count > 0
    
    generator_llm = initialize_llm(model_source)
    generator_llm.temperature = 0.5
    generator_tools = [CompilationCheckTool()]
    
    escaped_generator_specs = generator_specs.replace("{", "{{").replace("}", "}}") if generator_specs else ""
    
    feedback_section = ""
    if has_feedback:
        feedback = validator_messages[-1].content
        escaped_feedback = feedback.replace("{", "{{").replace("}", "}}")
        feedback_section = f"""
<feedback>
IMPORTANT: This is iteration {iteration_count} and you received the following feedback from the validator:
{escaped_feedback}
Please, correct your the code you have generated, addressing all these issues while ensuring your implementation remains complete with no placeholders.
</feedback>"""

    generator_template = f"""<role>
You are a specialized C programming agent that creates complete parser functions following strict requirements.
</role>

<main_directive>
IMPORTANT: 
- You have access to a compilation_check tool that verifies the correctness of your C code.
- You must always use the compilation_check tool for all C code you create. This is mandatory and not optional.
- Follow the verification process strictly any time you write C code. You should strive for high-quality, clean C code that follows best practices and compiles without errors.
- Since there are limits to how much code you can generate, keep your code simple, short and focused on the core functionality.
</main_directive>

<available_tools>
You have access to these tools:
{{tools}}
Tool names: {{tool_names}}
</available_tools>

<example_tool_usage>
Here's how you should use the compilation_check tool:
1. Write your C parser code
2. Call the compilation_check tool with your code:
   compilation_check(compilation_check(your_code_here).
3. Review the results
4. Fix any compilation warnings or errors
5. Verify again using the tool until the code compiles successfully
</example_tool_usage>

<parser_requirements>
Each parser you create must implement the following characteristics:
{requirements}
</parser_requirements>

<specifications>
The code you create must follow also the following requirements from the supervisor:
{escaped_generator_specs}
</specifications>

<critical_rules>
CRITICAL: 
- When writing code, you must provide complete implementations with NO placeholders, ellipses (...) or todos. Every function must be fully implemented.
- If the code is not complete, it will not compile and the compilation_check tool will fail.
- Before generating any code, always reason through your approach step by step.
- You only provide code in C. Not in Python. Not in C++. Not in any other language.
</critical_rules>

<verification_process>
CODE VERIFICATION PROCESS (ALWAYS MANDATORY):
- Write your complete C code implementation.
- Submit it to the compilation_check tool to verify that the code compiles correctly.
- If there are any errors or warings, fix them and verify the compilation again. This process may take several iterations.
- Let the structure of the code be simple, so that it is easier to generate code that compiles correctly.
- Continue this process until compilation succeeds without any errors.
- Once the compilation is successful, IMMEDIATELY move to Final Answer with the verified code. DO NOT run additional compilation checks on the same code.
- If the compilation is successful, answer to the user with the final code.
NEVER SKIP THE COMPILATION CHECK. If you do not verify that your code compiles cleanly, your response is incomplete and incorrect. The verification is REQUIRED for all C code responses without exception.
</verification_process>
{feedback_section}

<format_instructions>
Use the following format:
Question: the input question.
Thought: think about what to do.
Action: the tool to use: {{tool_names}}.
Action Input: the input to the tool.
Observation:
- if the compilation is successful, proceed to Final Answer without additional compilation checks.
- if the compilation is not successful, repeat Thought/Action/Action Input/Observation as needed.
Final Answer: the final code you have generated.
</format_instructions>

Generate a complete C parser implementation following all the requirements and specifications. Use the compilation_check tool to verify your code.

{{agent_scratchpad}}
"""

    # Create a prompt using PromptTemplate.from_template
    generator_prompt = PromptTemplate.from_template(generator_template)
    
    # Create the ReAct agent instead of OpenAI tools agent
    generator_agent = create_react_agent(generator_llm, generator_tools, generator_prompt)
    
    generator_executor = AgentExecutor(
        agent=generator_agent,
        tools=generator_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        early_stopping_method="force",
        return_intermediate_steps=True
    )
    
    try:
        generator_result = generator_executor.invoke({
            "tools": generator_tools,
            "tool_names": [tool.name for tool in generator_tools],
            "agent_scratchpad": []
        })
        generator_response = generator_result["output"]
        
        if "intermediate_steps" in generator_result:
            tool_used = False
            compilation_attempts = 0
            
            for step in generator_result["intermediate_steps"]:
                action = step[0]
                action_output = step[1]
                
                if action.tool == "compilation_check":
                    compilation_attempts += 1
                    tool_used = True

                    compilation_success = "Compilation successful" in action_output
                    
                    if state.get("system_metrics"):
                        state["system_metrics"].record_tool_usage(compilation_success)
                    print_colored(f"\nCompilation check tool used (Attempt {compilation_attempts})", "1;32")
                    
                    if compilation_success:
                        print_colored("Compilation successful!", "1;32")
                    else:
                        print_colored("Compilation failed!", "1;31")
            
            if not tool_used:
                print_colored("\nWarning: Compilation check tool was NOT used!", "1;33")
                
    except Exception as e:
        print_colored(f"Error during generator execution: {str(e)}", "1;31")
        generator_response = f"Error occurred during code generation: {str(e)}\n\nPlease try again."
    
    # increment iteration
    iteration_count = iteration_count + 1
    print_colored(f"\nGenerator (Iteration {iteration_count}):", "1;35")
    print(generator_response)
    
    with open(log_file, 'a') as f:
        f.write(f"Generator (Iteration {iteration_count}):\n")
        f.write(generator_response + "\n\n")
    
    next_step = "Supervisor" if iteration_count > state["max_iterations"] else "Validator"
    
    return {
        #"messages": messages + [AIMessage(content=generator_response, name="Generator")],
        "messages": [AIMessage(content=generator_response, name="Generator")],
        "user_request": user_request,
        "supervisor_memory": state["supervisor_memory"],
        "generator_specs": generator_specs,
        "generator_code": generator_response,
        "iteration_count": iteration_count,
        "max_iterations": state["max_iterations"],
        "model_source": model_source,
        "session_dir": session_dir,
        "log_file": log_file,
        "next_step": next_step,
        "parser_mode": state["parser_mode"],
        "system_metrics": state.get("system_metrics")
    }