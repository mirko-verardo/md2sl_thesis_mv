import os
import subprocess
import tempfile
import re
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from getpass import getpass
from pydantic import SecretStr
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain.agents import AgentExecutor
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun



### environment setup
def set_if_undefined(var: str) -> str:
    """Set environment variable (API KEY) from .env file."""
    ### load variables from .env file
    load_dotenv()
    ### check if variable exists in environment
    if not os.environ.get(var):
        os.environ[var] = getpass(f"Please provide your {var}")
    
    ### get the API key from environment
    api_key = os.environ.get(var) or ""
    return api_key

def initialize_llm(source: str):
    """Initialize a hosted model with appropriate parameters."""
    if source == "huggingface":
        # model_id = 'codellama/CodeLlama-34b-Instruct-hf'
        # https://evalplus.github.io/leaderboard.html
        # model_id = 'microsoft/Phi-3-mini-4k-instruct' # 56esimo
        # model_id = 'meta-llama/Meta-Llama-3-8B-Instruct' # 60eismo
        model_id = 'bigcode/starcoder2-15b' # 79eismo
        api_key = set_if_undefined("HUGGING_FACE_API_KEY")
        llm = HuggingFaceEndpoint(
            model=model_id,
            huggingfacehub_api_token = api_key,
            repo_id=model_id,
            task="text-generation",
            temperature=0.5, 
            repetition_penalty=1.3,
            do_sample=True,
            top_p=0.9,
            max_new_tokens=2048  # increase for longer parser implementations
        )
    
    elif source == "google":
        # model_id = 'gemini-1.5-pro'
        model_id = 'gemini-2.0-flash'
        # model_id = 'gemini-2.0-flash-lite'
        api_key = set_if_undefined("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0.5, # 0.2 # Slight temperature increase to improve reasoning
            max_tokens=None,
            google_api_key = api_key,
            timeout=None,
            max_retries=5,
            #top_p=0.95,  # Higher top_p for more creative reasoning
            #top_k=40,  # Reasonable top_k value
            # other params...
        )
    elif source == "openai":
        model_id = 'gpt-4o-mini'
        api_key = set_if_undefined("OPENAI_API_KEY")
        llm = ChatOpenAI(
            model=model_id,
            temperature=0.5,
            timeout=None,
            max_retries=5,
            api_key=SecretStr(api_key)
        )
        
    elif source == "anthropic":
        # model_id = 'claude-3-5-sonnet-20240620'
        model_id = 'claude-3-7-sonnet-20250219'
        # api_key = set_if_undefined("ANTHROPIC_API_KEY_PROF")
        api_key = set_if_undefined("ANTHROPIC_API_KEY_SAM")
        llm = ChatAnthropic(
            model_name=model_id,
            #model=model_id,
            stop=None,
            temperature=0.5,
            max_tokens_to_sample=6144,
            #max_tokens=6144,
            timeout=None,
            max_retries=5,
            api_key=SecretStr(api_key)
        )
    else:
        raise ValueError("Invalid source")

    return llm

class ExceptionTool(BaseTool):
    """Tool that just returns the query."""
    name: str = "_Exception"
    description: str = "Exception tool"

    def _run(self, query: str, run_manager: CallbackManagerForToolRun | None = None) -> str:
        return query

    async def _arun(self, query: str, run_manager: AsyncCallbackManagerForToolRun | None = None) -> str:
        return query

### compilationchecktool
class CompilationCheckTool(BaseTool):
    """Tool that checks if C code compiles correctly without warnings."""
    name: str = "compilation_check"
    description: str = """
    This tool checks if the provided C code compiles correctly without any warnings.
    Input should be valid C code.
    The tool will return compilation results, including any errors or warnings.
    Use this tool to verify that your C parser implementation is syntactically correct
    and free of warnings before providing it to the user.
    """

    #def _run(self, query: str, run_manager: CallbackManagerForToolRun | None = None) -> str:
    def _run(self, query, run_manager: CallbackManagerForToolRun | None = None) -> str:
        """Run the compilation check."""
        ### handle the case where query might be a dictionary
        if isinstance(query, dict):
            if 'query' in query:
                query = query['query']
            elif 'code' in query:  # Added to handle ReAct agent format
                query = query['code']
            else:
                ### try to get the first value if it's a different kind of dict
                try:
                    query = next(iter(query.values()))
                except (StopIteration, AttributeError):
                    return "Error: Invalid input format for compilation check"
        
        ### make sure we have a string at this point
        if not isinstance(query, str):
            return f"Error: Expected string input, got {type(query).__name__}"
            
        ### Clean the code by removing markdown delimiters
        # Remove ```c from the beginning of lines
        query = query.replace("```c", "")
        # Remove ``` from anywhere
        query = query.replace("```", "")
        # Trim whitespace
        query = query.strip()
        
        ### create a temporary file with the C code
        with tempfile.NamedTemporaryFile(suffix='.c', delete=False) as temp_file:
            temp_file.write(query.encode('utf-8'))
            temp_file_path = temp_file.name
        
        try:
            ### compile the C code with all warnings enabled and treating warnings as errors
            ### using -Wall and -Wextra flags to enable all warnings and -Werror to treat warnings as errors
            result = subprocess.run(
                ['gcc', '-Wall', '-Wextra', '-Werror', temp_file_path, '-o', temp_file_path + '.out'],
                capture_output=True,
                text=True
            )
            
            ### prepare the response
            if result.returncode == 0:
                response = "Compilation successful! The code compiles without any errors or warnings."
            else:
                response = f"Compilation failed with the following errors or warnings:\n{result.stderr}"
                ### add the original code after the error message for easy reference
                response += f"\n\nOriginal code:\n```c\n{query}\n```"
            
            return response
        finally:
            ### clean up temporary files
            try:
                os.unlink(temp_file_path)
                if os.path.exists(temp_file_path + '.out'):
                    os.unlink(temp_file_path + '.out')
            except Exception as e:
                pass  ### ignore cleanup errors
    
    async def _arun(self, query: str, run_manager: AsyncCallbackManagerForToolRun | None = None) -> str:
        """Run the compilation check asynchronously."""
        ### call the synchronous version
        #return self._run(query, run_manager)
        return self._run(query)
    
def extract_c_code(text: str) -> str | None:
    """Extract C code from the LLM response."""
    ### find code between ```c and ``` markers
    code_blocks = re.findall(r'```c(.*?)```', text, re.DOTALL)
    
    ### if not found, try without language specifier
    if not code_blocks:
        code_blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
    
    ### if still no code blocks found, check if the entire text is C code
    ### by looking for common C headers or patterns
    if not code_blocks:
        if text.strip().startswith('#include') or re.search(r'int\s+main\s*\(', text):
            return text.strip()
    
    ### return the first code block if any were found
    return code_blocks[0].strip() if code_blocks else None

def compile_c_code(c_file_path: Path) -> dict[str, Any]:
    """Compile the C code using gcc and return compilation result, considering warnings as issues."""
    output_file = c_file_path.with_suffix('')
    
    ### run gcc to compile the code with all warnings enabled and treated as errors
    result = subprocess.run(
        ['gcc', '-Wall', '-Wextra', '-Werror', str(c_file_path), '-o', str(output_file)],
        capture_output=True,
        text=True)
    
    ### with -Werror, compilation success means no warnings
    return {
        'success': result.returncode == 0,  ### compilation success (executable created, no warnings)
        'fully_clean': result.returncode == 0,    ### same as success since warnings cause failure with -Werror
        'stdout': result.stdout,
        'stderr': result.stderr,
        'executable': output_file if result.returncode == 0 else None,
        'has_warnings': False  ### with -Werror, warnings cause compilation failure, so this will always be False if successful
    }

def print_colored(text: str, color_code: str) -> None:
    """Print text with color."""
    print(f"\033[{color_code}m{text}\033[0m")

def start_chat(folder_path: str, folder_name: str, agent_executor: AgentExecutor) -> None:
    """Start chat with the agent. Save output files to the specified folder.
    
    Args:
        folder_path: Path where the 'output' folder will be created and files saved
        folder_name: Name of the subfolder within output directory
        agent_executor: The initialized agent executor
    """
    ### folder_path to a Path object if it's a string
    base_dir = Path(folder_path)
    
    ### create 'output' directory in the specified folder
    output_dir = base_dir / 'output' / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ### chat history for the current session
    session_history = []
    
    ### print welcome message
    print_colored("\n=== C Parser Generator Chat ===", "1;36")  # Bold Cyan
    print_colored("Ask for any parser or C function. Type 'exit' to quit.\n", "36")  # Cyan
    
    conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ### create 'session_id' directory in output folder
    session_dir = output_dir / f"session_{conversation_id}"
    session_dir.mkdir(exist_ok=True)

    print(f"Output directory: {session_dir}")

    log_file = session_dir / f"conversation_{conversation_id}.txt"
    
    with open(log_file, 'w') as f:
        f.write(f"=== C Parser Generator Chat - {datetime.now()} ===\n\n")
    
    ### main chat loop
    while True:
        ### get user input
        print_colored("\nYou: ", "1;32")  # Bold Green
        user_input = input()
        
        ### log user input
        with open(log_file, 'a') as f:
            f.write(f"You: {user_input}\n\n")
        
        ### check for exit command
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print_colored("\nExiting chat. Goodbye!", "1;36")
            break
        
        ### response from agent
        print_colored("\nAgent is thinking...", "33")  # Yellow
        
        try:
            ### set a timeout for the agent response (in seconds)
            response = agent_executor.invoke({"input": user_input})
            agent_output = response.get('output', '')
            
            ### log the agent's intermediate steps
            if 'intermediate_steps' in response:
                print_colored("\n--- Agent's Reasoning Process ---", "1;35")  # Bold Magenta
                
                step_counter = 1
                compilation_attempts = 0
                last_compilation_result = None
                
                for step in response['intermediate_steps']:
                    action = step[0]
                    action_output = step[1]
                    
                    if action.tool == "compilation_check":
                        compilation_attempts += 1
                        last_compilation_result = action_output
                        
                        print_colored(f"\nStep {step_counter}: Using Compilation Check Tool (Attempt {compilation_attempts})", "1;34")  # Bold Blue
                        
                        ### first few lines of code being checked (truncated)
                        if isinstance(action.tool_input, dict) and 'query' in action.tool_input:
                            code_to_check = action.tool_input['query']
                        else:
                            #code_to_check = action.tool_input
                            code_to_check = str(action.tool_input)
                            
                        code_preview = code_to_check.split("\n")
                        code_preview = "\n".join(code_preview[:5]) + "\n..." if len(code_preview) > 5 else code_to_check
                        print_colored("Checking code (preview):", "34")  # Blue
                        print(code_preview)
                        
                        ### print compilation results
                        if "Compilation successful" in action_output:
                            print_colored("Result: Compilation successful without warnings! ✓", "1;32")  # Bold Green
                        else:
                            print_colored("Result: Compilation failed! ✗", "1;31")  # Bold Red
                            ### extract errors, more robust approach
                            try:
                                if "\n\nOriginal code:" in action_output:
                                    errors = action_output.split("\n\nOriginal code:")[0]
                                    ### find the last occurrence of ":" before errors and take everything after it
                                    last_colon_pos = errors.rfind(":")
                                    if last_colon_pos != -1:
                                        errors = errors[last_colon_pos + 1:].strip()
                                    else:
                                        errors = errors.strip()
                                else:
                                    errors = action_output.strip()
                                    
                                ### truncate errors if too long
                                if len(errors.split("\n")) > 10:
                                    errors_lines = errors.split("\n")
                                    errors = "\n".join(errors_lines[:10]) + "\n...(more errors)..."
                                print(errors)
                            except Exception:
                                ### ultimate fallback - just print the whole output if error parsing fails
                                print(action_output)
                    
                    step_counter += 1
                
                ### if compilation was attempted but the last attempt failed, print a warning
                if compilation_attempts > 0 and last_compilation_result:
                    if "Compilation failed" in last_compilation_result:
                        print_colored("\nWARNING: Last compilation attempt failed! The agent may provide code that doesn't compile.", "1;31")  # Bold Red
                
                print_colored("\n--- End of Reasoning Process ---", "1;35")  # Bold Magenta
            
            ### print the final response
            print_colored("\nAgent: ", "1;33")  # Bold Yellow
            print(agent_output)
            
            ### log agent response
            with open(log_file, 'a') as f:
                ### log the intermediate steps
                if 'intermediate_steps' in response:
                    f.write("--- Agent's Reasoning Process ---\n")
                    compilation_attempts = 0
                    for step in response['intermediate_steps']:
                        action = step[0]
                        action_output = step[1]
                        
                        if action.tool == "compilation_check":
                            compilation_attempts += 1
                            f.write(f"Using Compilation Check Tool (Attempt {compilation_attempts}):\n")
                            
                            ### get the code being checked
                            if isinstance(action.tool_input, dict) and 'query' in action.tool_input:
                                code_to_check = action.tool_input['query']
                            else:
                                #code_to_check = action.tool_input
                                code_to_check = str(action.tool_input)
                                
                            f.write(f"Code being checked (truncated):\n")
                            code_preview = code_to_check.split("\n")
                            code_preview = "\n".join(code_preview[:10]) + "\n..." if len(code_preview) > 10 else code_to_check
                            f.write(f"{code_preview}\n\n")
                            f.write(f"Compilation Result:\n{action_output}\n\n")
                    f.write("--- End of Reasoning Process ---\n\n")
                
                f.write(f"Agent: {agent_output}\n\n")
            
            ### store in session history
            session_history.append({"user": user_input, "agent": agent_output})
            
            ### extract C code if present
            c_code = extract_c_code(agent_output)
            
            ### if C code was found, compile and test it
            if c_code:
                print_colored("\n--- Compiling Final C code ---", "1;33")  # Bold Yellow
                
                ### create unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                c_file_name = f"parser_{timestamp}.c"
                c_file_path = session_dir / c_file_name
                
                ### save the C code to a file
                with open(c_file_path, 'w') as f:
                    f.write(c_code)
                
                print(f"C code saved to: {c_file_path}")
                
                ### compile the C code
                print("Compiling with gcc...")
                compilation_result = compile_c_code(c_file_path)
                
                ### save compilation results
                result_file_name = f"compilation_result_{timestamp}.txt"
                result_file_path = session_dir / result_file_name
                
                with open(result_file_path, 'w') as f:
                    f.write(f"Compilation {'successful' if compilation_result['success'] else 'failed'}\n\n")
                    ### with -Werror, successful compilation means clean code
                    if compilation_result['success']:
                        f.write("Status: Clean compilation (no errors or warnings)\n\n")
                    
                    ### update command to show we're using -Werror flag
                    f.write(f"Command: gcc -Wall -Wextra -Werror {c_file_path} -o {c_file_path.with_suffix('')}\n\n")
                    
                    if compilation_result['stdout']:
                        f.write(f"Standard output:\n{compilation_result['stdout']}\n\n")
                    if compilation_result['stderr']:
                        f.write(f"Standard error:\n{compilation_result['stderr']}\n\n")
                    if compilation_result['executable']:
                        f.write(f"Executable created at: {compilation_result['executable']}\n")
                
                ### print compilation status
                if compilation_result['success']:
                    print_colored("Compilation successful without warnings!", "1;32")  # Bold Green
                    print(f"Executable created at: {compilation_result['executable']}")
                else:
                    print_colored("Compilation failed. Errors:", "1;31")  # Bold Red
                    print(compilation_result['stderr'])
                    
                    ### log compilation results
                    with open(log_file, 'a') as f:
                        f.write(f"--- Compilation Results ---\n")
                        f.write(f"Compilation {'successful' if compilation_result['success'] else 'failed'}\n")
                        if compilation_result['stderr']:
                            f.write(f"Errors/Warnings:\n{compilation_result['stderr']}\n\n")
                
                print(f"Compilation details saved to: {result_file_path}")
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print_colored(f"\n{error_msg}", "1;31")  # Bold Red
            
            ### try to provide more detailed error information
            import traceback
            trace = traceback.format_exc()
            print_colored("\nDetailed error information:", "1;31")
            print(trace)
            
            with open(log_file, 'a') as f:
                f.write(f"{error_msg}\n\n")
                f.write(f"Detailed error:\n{trace}\n\n")

    ### save the entire conversation history
    history_file = session_dir / f"full_history_{conversation_id}.txt"
    with open(history_file, 'w') as f:
        f.write(f"=== C Parser Generator Chat History - {datetime.now()} ===\n\n")
        for exchange in session_history:
            f.write(f"You: {exchange['user']}\n\n")
            f.write(f"Agent: {exchange['agent']}\n\n")
            f.write("---\n\n")
    
    print(f"\nConversation history saved to: {history_file}")
    print(f"Conversation log saved to: {log_file}")