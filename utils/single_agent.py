from datetime import datetime
from traceback import format_exc
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from models import CompilationCheck, ExecutionCheck
from utils import colors
from utils.general import create_session, initialize_llm, extract_c_code, compile_c_code, execute_c_code, print_colored, log, get_parser_requirements



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
When user greets you or makes casual conversation, respond appropriately without generating any C code. 
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

SUCCESS CASE (PARSING SUCCEEDS):
- The program MUST NOT print anything to stderr.
- The program MUST print ONLY a normalized summary of the parsed structure to stdout — NEVER raw input bytes.
- The summary must be concise and consistent in format, independent of the input type.
- After printing the summary to stdout, the program MUST terminate IMMEDIATELY with exit code 0.

FAILURE CASE (PARSING FAILS):
- The program MUST NOT print anything to stdout.
- The program MUST print a descriptive error message to stderr.
- After printing the error to stderr, the program MUST terminate IMMEDIATELY with a NON-ZERO exit code.

GLOBAL RULES (APPLIES TO ALL CODE PATHS):
- ANY condition that produces output on stderr MUST be treated as a fatal parsing error.
- The program MUST NOT print warnings, informational messages, debug output, or non-fatal notices to stderr.
- If the program prints anything to stderr, it MUST exit with a non-zero code.
- Under no circumstances may the program exit with code 0 if ANY output was written to stderr.

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
- If test execution is not successful, fix the code and return to Thought 1 (restart the entire process).
- If test execution is successful, proceed to Final Answer.

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

    # Create session
    session_dir, log_file = create_session(source, "few_shot" if few_shot else "zero_shot", file_format)
    session_history = []
    f = open(log_file, "a", encoding="utf-8")

    # print welcome message
    log(f, "=== C Parser Generator Chat ===", colors.CYAN, bold=True)
    print_colored("Ask for any parser or C function. Type 'exit' to quit.\n", colors.CYAN)

    # Initialize parameters
    user_input = f"Generate a parser function for {file_format} files."
    
    # main chat loop
    while True:
        # log user input
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
            tool_attempts = {}
            steps = agent_response.get("intermediate_steps", [])
            if steps:
                log(f, "--- Agent's Reasoning Process ---", colors.MAGENTA, bold=True)

                for step_counter, step in enumerate(steps):
                    action = step[0]
                    action_output = step[1]
                    
                    action_tool = action.tool
                    if action_tool not in ["compilation_check", "execution_check"]:
                        continue
                        
                    attempts = tool_attempts.get(action_tool, 0) + 1
                    tool_attempts.update({ action_tool: attempts })
                    log(f, f"Step {step_counter + 1}: Using {action_tool} tool (attempt {attempts})", colors.BLUE, bold=True)

                    # code: extract only top lines for preview
                    code = str(action.tool_input).strip()
                    code_preview = code.split("\n")
                    code_preview = "\n".join(code_preview[:10]) + "\n..." if len(code_preview) > 10 else code
                    log(f, "Checking code (preview):", colors.BLUE)
                    log(f, code_preview)

                    # log tool results
                    if action_output["success"]:
                        log(f, f"Result: {action_tool} tool successful without warnings! ✓", colors.GREEN, bold=True)
                    else:
                        log(f, f"Result: {action_tool} tool failed! ✗", colors.RED, bold=True)
                    
                    errors = action_output["stderr"]
                    if errors:
                        # errors: extract only top lines for preview
                        errors = str(errors).strip()
                        errors_preview = errors.split("\n")
                        errors_preview = "\n".join(errors_preview[:10]) + "\n..." if len(errors_preview) > 10 else errors
                        log(f, "Checking errors (preview):", colors.RED)
                        log(f, errors_preview)

                log(f, "--- End of Reasoning Process ---", colors.MAGENTA, bold=True)
            
            if tool_attempts.get("compilation_check", 0) == 0:
                print_colored("\nWarning: Compilation check tool was NOT used!", colors.YELLOW, bold=True)
            if tool_attempts.get("execution_check", 0) == 0:
                print_colored("\nWarning: Execution check tool was NOT used!", colors.YELLOW, bold=True)
            
            # log agent response
            log(f, "Agent:", colors.YELLOW, bold=True)
            log(f, agent_output)

            # store in session history
            session_history.append({"user": user_input, "agent": agent_output})
            
            # extract C code
            code = extract_c_code(agent_output)
            
            # compile and test it
            print_colored("\n--- Compiling and Testing Final C code ---", colors.YELLOW, bold=True)

            # get the parser dir
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            parser_dir = session_dir / f"parser_{timestamp}"
            parser_dir.mkdir()

            # compile the C code
            print("Compiling...")
            compilation_result = compile_c_code(parser_dir, code)
            
            is_compilation_ok = compilation_result["success"]
            if is_compilation_ok:
                # testing the C code
                print("Testing...")
                testing_result = execute_c_code(parser_dir, file_format)
            else:
                testing_result = {
                    "success": False,
                    "stdout": "",
                    "stderr": "Not even compiled"
                }
            
            # save results
            result_file_path = parser_dir / "results.txt"
            
            with open(result_file_path, "w", encoding="utf-8") as ff:
                # Compilation
                if is_compilation_ok:
                    log(ff, "Compilation successful (no errors or warnings)", colors.GREEN, bold=True)
                else:
                    log(ff, "Compilation failed", colors.RED, bold=True)
                stdout = compilation_result["stdout"]
                stderr = compilation_result["stderr"]
                if stdout:
                    log(ff, f"Standard output:\n{stdout}\n")
                if stderr:
                    log(ff, f"Standard error:\n{stderr}")
                
                # Testing
                if testing_result["success"]:
                    log(ff, "Testing successful (no errors or warnings)", colors.GREEN, bold=True)
                else:
                    log(ff, "Testing failed", colors.RED, bold=True)
                stdout = testing_result["stdout"]
                stderr = testing_result["stderr"]
                if stdout:
                    log(ff, f"Standard output:\n{stdout}\n")
                if stderr:
                    log(ff, f"Standard error:\n{stderr}")
            
            print(f"\nCompilation details saved to: {result_file_path}")
            
        except Exception as e:
            log(f, f"Error: {str(e)}", colors.RED, bold=True)
            log(f, "Detailed error:", colors.RED, bold=True)
            log(f, format_exc())
        
        # Ask the user
        user_input = input("\nYou: ")
        # check for exit command
        if user_input.lower() in ["exit", "quit", "bye"]:
            break

    # manually close the stream
    f.close()

    # save the entire conversation history
    history_file = session_dir / f"full_history.txt"
    with open(history_file, "w", encoding="utf-8") as f:
        f.write(f"=== C Parser Generator Chat History - {datetime.now()} ===\n\n")
        for exchange in session_history:
            f.write(f"You: {exchange['user']}\n\n")
            f.write(f"Agent: {exchange['agent']}\n\n")
            f.write("---\n\n")
    
    print(f"\nConversation history saved to: {history_file}")
    print(f"Conversation log saved to: {log_file}")