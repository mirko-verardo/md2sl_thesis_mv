from datetime import datetime
from pathlib import Path
from traceback import format_exc
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from models import CompilationCheckTool
from utils import colors
from utils.general import set_if_undefined, initialize_llm, extract_c_code, compile_c_code, print_colored, log



def __get_template(few_shot: bool) -> str:
    return """<role>
You are a C programming assistant specialized in creating parser functions.
</role>

<main_directive>
IMPORTANT: 
- You must always use the compilation_check tool for all the C code you generate. This is not optional.
- Follow the VERIFICATION PROCESS strictly any time you write C code.
- The user will ask you to create very general parsers, defining only the data format and without any inputs, outputs or structure details.
- You cannot ask the user for details or clarifications about the parser. 
- Be creative and think about the function input, output and structure by yourself. Then, write the code to realize the function you have imagined.
</main_directive>

<conversation_guidelines>
You should engage naturally in conversation with users. When users greet you or make casual conversation, respond appropriately without generating any C code. 
You should only generate code when users explicitly request a parser funtion implementation.
</conversation_guidelines>

<parser_requirements>
When the user does request a parser function, each parser you create must implement all of the following characteristics:
1. Input Handling: The code deals with a pointer to a buffer of bytes or a file descriptor for reading unstructured data.
2. Internal State: The code maintains an internal state in memory to represent the parsing state that is not returned as output.
3. Decision-Making: The code takes decisions based on the input and the internal state.
4. Data Structure Creation: The code builds a data structure representing the accepted input or executes specific actions on it.
5. Outcome: The code returns either a boolean value or a data structure built from the parsed data indicating the outcome of the recognition.
6. Composition: The code behavior is defined as a composition of other parsers. (Note: This requirement is only necessary if one of the previous requirements are not met. If ALL the previous 5 requirements are satisfied, this requirement becomes optional.)
</parser_requirements>

""" + ("" if not few_shot else """<examples>
Here are some examples to clarify what is and isn't acceptable:

UNACCEPTABLE EXAMPLE - This does not satisfy our requirements for a parser function:
```c
void FUN_00401050(void) {{
code **ppcVar1;
for (ppcVar1 = (code **)&DAT_00411198; *ppcVar1 != (code *)0xffffffff; ppcVar1 = ppcVar1 + -1) {{
(**ppcVar1)();
}}
return;
}}
```

ACCEPTABLE EXAMPLE - This satisfies our requirements for a parser function:
```c
void parseCSV(const char *input) {{
char copy[100];
strcpy(copy, input);
char* token = strtok(copy, ",");
while (token != NULL) {{
if (isInteger(token)) {{
printf("Parsed Integer: %s\\n", token);
}} else {{
printf("Invalid Input: %s\\n", token);
}}
token = strtok(NULL, ",");
}}
}}
```

ACCEPTABLE EXAMPLE - This satisfies our requirements for a parser function component:
```c
char *skip_whitespace(char *s) {{
while (*s && isspace((unsigned char)*s)) s++;
return s;
}}
```
</examples>

""") + """<critical_rules>
CRITICAL: 
- When writing code, you must provide complete implementations with NO placeholders, ellipses (...) or todos. Every function must be fully implemented.
- If the code is not complete, it will not compile and the compilation_check tool will fail.
- Before generating any code, always reason through your approach step by step.
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

<available_tools>
You have access to these tools: {tools}
Tool names: {tool_names}
Call the tool using compilation_check(your_code_here). 
</available_tools>

<format_instructions>
Use the following format:
Question: the input question.
Thought: think about what to do.
Action: the tool to use: {tool_names}.
Action Input: the input to the tool.
Observation:
- if the compilation is successful, proceed to Final Answer without additional compilation checks.
- if the compilation is not successful, repeat Thought/Action/Action Input/Observation as needed.
Final Answer: the final answer to the original question with the final code you have generated.
</format_instructions>

<chat_history>
The previous conversation between you and the user is as follows:
{chat_history}
</chat_history>

Now, the user is asking: {input}
{agent_scratchpad}
"""

def setup_agent(source: str, few_shot=False) -> AgentExecutor:
    """Set up and return the agent and memory objects specifically for Claude using ReAct pattern."""
    # initialize model
    llm = initialize_llm(source)
    
    # use the CompilationCheckTool
    tools = [CompilationCheckTool()]
    
    # create the system message 
    template = __get_template(few_shot)

    # create a prompt
    prompt = PromptTemplate.from_template(template)
    # create memory
    memory = ConversationBufferMemory(memory_key="chat_history", output_key='output', return_messages=True)
    # create the ReAct agent
    agent = create_react_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=5,
        early_stopping_method="force",
    )

def start_chat(folder_name: str, agent_executor: AgentExecutor) -> None:
    """Start chat with the agent. Save output files to the specified folder.
    
    Args:
        folder_name: Name of the subfolder within output directory
        agent_executor: The initialized agent executor
    """
    # folder_path: Path where the 'output' folder will be created and files saved
    folder_path = set_if_undefined("FOLDER_PATH")
    # folder_path to a Path object if it's a string
    base_dir = Path(folder_path)
    
    # create 'output' directory in the specified folder
    output_dir = base_dir / 'output' / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # chat history for the current session
    session_history = []
    
    conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    # create 'session_id' directory in output folder
    session_dir = output_dir / f"session_{conversation_id}"
    session_dir.mkdir(exist_ok=True)

    log_file = session_dir / f"conversation_{conversation_id}.txt"
    f = open(log_file, 'a', encoding="utf-8")

    # print welcome message
    log(f, "=== C Parser Generator Chat ===", colors.CYAN, bold=True)
    print_colored("Ask for any parser or C function. Type 'exit' to quit.\n", colors.CYAN)
    print(f"Output directory: {session_dir}")
    
    # main chat loop
    while True:
        # get and log user input
        log(f, "You:", colors.GREEN, bold=True)
        user_input = input()
        f.write(f"\n{user_input}\n")
        
        # check for exit command
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print_colored("\nExiting chat. Goodbye!", "1;36") # Bold Cyan
            break
        
        # response from agent
        print_colored("\nAgent is thinking...", colors.YELLOW)
        
        try:
            # set a timeout for the agent response (in seconds)
            response = agent_executor.invoke({"input": user_input})
            agent_output = response.get('output', '')

            # log the intermediate steps
            if 'intermediate_steps' in response:
                log(f, "--- Agent's Reasoning Process ---", colors.MAGENTA, bold=True)

                compilation_attempts = 0
                last_compilation_result = None

                for step_counter, step in enumerate(response['intermediate_steps']):
                    action = step[0]
                    action_output = step[1]
                    
                    if action.tool == "compilation_check":
                        compilation_attempts += 1
                        last_compilation_result = action_output

                        log(f, f"Step {step_counter + 1}: Using Compilation Check Tool (Attempt {compilation_attempts})", 
                            colors.BLUE, bold=True)
                        
                        # get the code being checked
                        if isinstance(action.tool_input, dict) and 'query' in action.tool_input:
                            code_to_check = action.tool_input['query']
                        else:
                            #code_to_check = action.tool_input
                            code_to_check = str(action.tool_input)
                        
                        # split each code lines
                        code_preview = code_to_check.split("\n")
                        # extract only first 5 lines for preview
                        code_preview = "\n".join(code_preview[:5]) + "\n..." if len(code_preview) > 5 else code_to_check

                        log(f, "Checking code (preview):", colors.BLUE)
                        log(f, code_preview)

                        # log compilation results
                        if "Compilation successful" in action_output:
                            log(f, "Result: Compilation successful without warnings! ✓", colors.GREEN, bold=True)
                        else:
                            log(f, "Result: Compilation failed! ✗", colors.RED, bold=True)

                            # extract errors (more robust approach) and truncate if too long
                            errors = action_output
                            if "Original code:" in action_output:
                                errors = action_output.split("Original code:")[0]
                                # find the last occurrence of ":" before errors and take everything after it
                                #last_colon_pos = errors.rfind(":")
                                #if last_colon_pos != -1:
                                    #errors = errors[last_colon_pos + 1:]
                            errors = errors.strip()
                            errors_preview = errors.split("\n")
                            errors_preview = "\n".join(errors_preview[:10]) + "\n..." if len(errors_preview) > 10 else errors
                            #print(errors_preview)
                            log(f, errors_preview)
                        
                        # anyway
                        #f.write(f"\n{action_output}\n")

                # if the last compilation attempt failed, print a warning
                if last_compilation_result and "Compilation failed" in last_compilation_result:
                    log(f, "WARNING: Last compilation attempt failed! The agent may provide code that doesn't compile.", 
                        colors.RED, bold=True)

                log(f, "--- End of Reasoning Process ---", colors.MAGENTA, bold=True)
            
            # log agent response
            log(f, "Agent:", colors.YELLOW, bold=True)
            log(f, agent_output)

            # store in session history
            session_history.append({"user": user_input, "agent": agent_output})
            
            # extract C code if present
            c_code = extract_c_code(agent_output)
            
            # if C code was found, compile and test it
            if c_code:
                print_colored("\n--- Compiling Final C code ---", "1;33")  # Bold Yellow
                
                # create unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                c_file_name = f"parser_{timestamp}.c"
                c_file_path = session_dir / c_file_name
                
                # save the C code to a file
                with open(c_file_path, 'w') as ff:
                    ff.write(c_code)
                
                print(f"\nC code saved to: {c_file_path}")
                
                # compile the C code
                print("Compiling with gcc...")
                compilation_result = compile_c_code(c_file_path)
                
                # save compilation results
                result_file_name = f"compilation_result_{timestamp}.txt"
                result_file_path = session_dir / result_file_name
                
                with open(result_file_path, 'w') as ff:
                    # with -Werror, successful compilation means clean code
                    if compilation_result['success']:
                        log(ff, "Compilation successful (no errors or warnings)", colors.GREEN, bold=True)
                    else:
                        log(ff, "Compilation failed", colors.RED, bold=True)
                    
                    # update command to show we're using -Werror flag
                    ff.write(f"\nCommand: gcc -Wall -Wextra -Werror {c_file_path} -o {c_file_path.with_suffix('')}\n")
                    
                    if compilation_result['stdout']:
                        ff.write(f"\nStandard output:\n{compilation_result['stdout']}\n")
                    if compilation_result['stderr']:
                        log(ff, f"Standard error:\n{compilation_result['stderr']}")
                    if compilation_result['executable']:
                        log(ff, f"Executable created at: {compilation_result['executable']}")
                    
                # log compilation results
                #f.write(f"\n--- Compilation Results ---\n")
                #f.write(f"\nCompilation {'successful' if compilation_result['success'] else 'failed'}\n")
                #if compilation_result['stderr']:
                #    f.write(f"\nStandard error:\n{compilation_result['stderr']}\n")
                
                print(f"\nCompilation details saved to: {result_file_path}")
            
        except Exception as e:
            log(f, f"Error: {str(e)}", colors.RED, bold=True)
            log(f, "Detailed error:", colors.RED, bold=True)
            log(f, format_exc())

    # manually close the stream
    f.close()

    # save the entire conversation history
    history_file = session_dir / f"full_history_{conversation_id}.txt"
    with open(history_file, 'w') as f:
        f.write(f"=== C Parser Generator Chat History - {datetime.now()} ===\n\n")
        for exchange in session_history:
            f.write(f"You: {exchange['user']}\n\n")
            f.write(f"Agent: {exchange['agent']}\n\n")
            f.write("---\n\n")
    
    print(f"\nConversation history saved to: {history_file}")
    print(f"Conversation log saved to: {log_file}")