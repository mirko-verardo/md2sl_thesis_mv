from datetime import datetime
from pathlib import Path
from traceback import format_exc
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from models import CompilationCheck, ExecutionCheck
from utils import colors
from utils.general import set_if_undefined, initialize_llm, extract_c_code, compile_c_code, print_colored, log, get_parser_requirements



def __get_template() -> str:
    return """<role>
You are a C programming assistant specialized in creating parser functions.
</role>

<main_directive>
- You must always use the "compilation_check" and "execution_check" tools for verifying the correctness of the C code you generate. This is mandatory and not optional.
- Follow the verification process strictly any time you write C code.
- Keep your code simple, short and focused on the core functionality, so it will be easier that the generated code compiles and executes the test correctly.
- When writing code, you must provide complete implementations with NO placeholders, ellipses (...) or todos. Every function must be fully implemented.
- You only provide code in C. Not in Python. Not in C++. Not in any other language.
- You cannot ask the user for details or clarifications about the parser. 
- Be creative and think about the function input, output and structure by yourself. Then, write the code to realize the function you have imagined.
</main_directive>

<available_tools>
You have access to these tools: {tools}
Tool names: {tool_names}
</available_tools>

<conversation_guidelines>
You should engage naturally in conversation with user. 
When user greets you or make casual conversation, respond appropriately without generating any C code. 
You should only generate code when user explicitly requests a parser funtion implementation.
</conversation_guidelines>

<parser_requirements>
The parser must implement the following requirements:
{requirements}
</parser_requirements>

{examples}

<verification_process>
CODE VERIFICATION PROCESS (IMPORTANT AND MANDATORY):
- Write your complete C code implementation.
- Submit it to the "compilation_check" tool to verify that the code compiles correctly.
- If there are any errors or warnings, fix them and verify the compilation again. This process may take several iterations.
- Submit it to the "execution_check" tool to verify that the code executes the test correctly.
- If there are any errors or warnings, fix them and go back to the compilation again. This process may take several iterations.
- Once the test execution is successful, IMMEDIATELY move to Final Answer with the final code without runnning additional loops.
</verification_process>

<input_handling>
CODE INPUT (IMPORTANT AND MANDATORY):
The final C code you generate must read the entire input from standard input (stdin) as raw bytes.
Use a binary-safe approach such as reading in chunks with fread() into a dynamically resized buffer.
Do not use scanf, fgets, or any text-only input functions for the primary input. The parser's input must always come from stdin as bytes.
This is really important because the final C code you generated will be tested giving raw bytes as stdin.
</input_handling>

<output_handling>
CODE OUTPUT (IMPORTANT AND MANDATORY):
Always follow this output rule:
- If parsing succeeds, print a normalized summary of the parsed structure to stdout — NEVER raw input bytes.
- If parsing fails, do NOT print anything to stdout; instead, write a descriptive error message to stderr.
- The summary should be concise and consistent in format, independent of the file type has been parsed.
</output_handling>

<format_instructions>
Use the following format:

Question: the input question.

Thought 1: I need to check if the code compiles.
Action 1: compilation_check
Action Input 1: <code_string>
Observation 1:
- If the compilation is not successful, fix the code and repeat Thought 1/Action 1/Action Input 1/Observation 1.
- If the compilation is successful, proceed to:

Thought 2: The code compiles; now I need to check if it executes the test correctly.
Action 2: execution_check
Action Input 2: <code_string>
Observation 2:
- If execution is not successful, fix the code and return to Thought 1 (restart the entire process).
- If execution is successful, proceed to Final Answer.

Final Answer: the final code you have generated.
</format_instructions>

<chat_history>
The previous conversation between you and the user is as follows:
{chat_history}
</chat_history>

Now, the user is asking: {input}
{agent_scratchpad}
"""

def __get_examples() -> str:
    return """<examples>
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
</examples>"""

def start_chat(source: str, file_format: str, few_shot: bool = False) -> None:
    """Start chat with the agent. Save output files to the specified folder."""

    # initialize model
    llm = initialize_llm(source)
    
    # use the CompilationCheck tool
    tools = [ CompilationCheck, ExecutionCheck(file_format) ]
    
    # create the system message 
    template = __get_template()

    # create a prompt
    prompt = PromptTemplate.from_template(template)
    # create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        input_key="input",
        output_key="output", 
        return_messages=True
    )
    # create the ReAct agent
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=10,
        early_stopping_method="force"
    )
    
    # folder_path: Path where the output folder will be created and files saved
    folder_path = set_if_undefined("FOLDER_PATH")
    # folder_path to a Path object if it's a string
    base_dir = Path(folder_path)
    
    # create output directory in the specified folder
    folder_name = source + "/" + ("few_shot" if few_shot else "zero_shot")
    output_dir = base_dir / "output" / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # chat history for the current session
    session_history = []
    
    conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    # create 'session_id' directory in output folder
    session_dir = output_dir / f"session_{conversation_id}"
    session_dir.mkdir(exist_ok=True)

    log_file = session_dir / f"conversation_{conversation_id}.txt"
    f = open(log_file, "a", encoding="utf-8")

    # print welcome message
    log(f, "=== C Parser Generator Chat ===", colors.CYAN, bold=True)
    print_colored("Ask for any parser or C function. Type 'exit' to quit.\n", colors.CYAN)
    print(f"Output directory: {session_dir}")
    
    # main chat loop
    start = True
    while True:
        # get and log user input
        if start:
            user_input = f"generate a parser function for {file_format} files"
            start = False
        else:
            log(f, "You:", colors.GREEN, bold=True)
            user_input = input()        
            # check for exit command
            if user_input.lower() in ["exit", "quit", "bye"]:
                break
        f.write(f"\n{user_input}\n")
        
        # response from agent
        print_colored("\nAgent is thinking...", colors.YELLOW)
        
        try:
            # set a timeout for the agent response (in seconds)
            agent_input = {
                "input": user_input,
                "requirements": get_parser_requirements(),
                "examples": __get_examples() if few_shot else ""
            }
            agent_response = agent_executor.invoke(agent_input)
            agent_output = str(agent_response["output"])

            # log the intermediate steps
            steps = agent_response.get("intermediate_steps", [])
            if steps:
                log(f, "--- Agent's Reasoning Process ---", colors.MAGENTA, bold=True)

                #compilation_attempts = 0
                for step_counter, step in enumerate(steps):
                    action = step[0]
                    action_output = step[1]
                    
                    action_tool = action.tool
                    if action_tool not in ["compilation_check", "execution_check"]:
                        continue
                        
                    #compilation_attempts += 1
                    #log(f, f"Step {step_counter + 1}: Using Compilation Check Tool (Attempt {compilation_attempts})", colors.BLUE, bold=True)
                    log(f, f"Step {step_counter + 1}: Using {action_tool} tool", colors.BLUE, bold=True)

                    # get the code being checked
                    code_to_check = str(action.tool_input)
                    code_to_check = code_to_check.strip()
                    
                    # extract only top lines for preview
                    code_preview = code_to_check.split("\n")
                    code_preview = "\n".join(code_preview[:10]) + "\n..." if len(code_preview) > 10 else code_to_check

                    log(f, "Checking code (preview):", colors.BLUE)
                    log(f, code_preview)

                    # log tool results
                    if action_output["success"]:
                        log(f, "Result: Tool successful without warnings! ✓", colors.GREEN, bold=True)
                    else:
                        log(f, "Result: Tool failed! ✗", colors.RED, bold=True)
                    
                    # get the errors
                    errors = action_output["stderr"]
                    if errors:
                        errors = errors.strip()
                        # extract only top lines for preview
                        errors_preview = errors.split("\n")
                        errors_preview = "\n".join(errors_preview[:10]) + "\n..." if len(errors_preview) > 10 else errors
                        log(f, errors_preview)

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
                print_colored("\n--- Compiling Final C code ---", colors.YELLOW, bold=True)
                
                # create unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                c_file_name = f"parser_{timestamp}.c"
                c_file_path = session_dir / c_file_name
                o_file_path = c_file_path.with_suffix('')
                
                # save the C code to a file
                with open(c_file_path, "w") as ff:
                    ff.write(c_code)
                
                print(f"\nC code saved to: {c_file_path}")
                
                # compile the C code
                print("Compiling with gcc...")
                compilation_result = compile_c_code(str(c_file_path), str(o_file_path))
                
                # save compilation results
                result_file_name = f"compilation_result_{timestamp}.txt"
                result_file_path = session_dir / result_file_name
                
                with open(result_file_path, "w") as ff:
                    # with -Werror, successful compilation means clean code
                    if compilation_result['success']:
                        log(ff, "Compilation successful (no errors or warnings)", colors.GREEN, bold=True)
                    else:
                        log(ff, "Compilation failed", colors.RED, bold=True)
                    
                    # update command to show we're using -Werror flag
                    log(ff, f"Command: gcc -Wall -Wextra -Werror {c_file_path} -o {o_file_path}\n")
                    
                    if compilation_result['stdout']:
                        log(ff, f"Standard output:\n{compilation_result['stdout']}\n")
                    if compilation_result['stderr']:
                        log(ff, f"Standard error:\n{compilation_result['stderr']}")
                    if compilation_result['executable']:
                        log(ff, f"Executable created at: {compilation_result['executable']}")
                
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