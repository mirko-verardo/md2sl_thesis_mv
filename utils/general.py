import os, re
from dotenv import load_dotenv
from getpass import getpass
from pydantic import SecretStr
from subprocess import run
from tempfile import TemporaryDirectory
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from utils import colors



def set_if_undefined(var: str) -> str:
    """Set environment variable from .env file."""
    # load variables from .env file
    load_dotenv()

    # check if variable exists in the environment
    value = os.environ.get(var)
    if not value:
        # set the variable if it doesn't exist
        value = getpass(f"Please provide your {var}")
        os.environ[var] = value
    
    return value

def get_model_source_from_input(speed_up: bool = True) -> str:
    """Get the model source name from the user input"""

    # speed up
    if speed_up:
        return "google"
    
    sources = ["google", "openai", "anthropic"]
    sources_str = f"'{"', '".join(sources)}'"
    print_colored("\n=== C Parser Generator Setup ===", colors.CYAN, bold=True)
    print_colored(f"Available model sources: {sources_str}", colors.YELLOW, bold=True)
    
    # get model source
    while True:
        source = input("\nEnter the model source: ").strip().lower()
        
        if source in sources:
            print_colored(f"\nSelected model source: {source}", colors.GREEN, bold=True)
            return source
        
        print_colored(f"Invalid source. Please enter one of these: {sources_str}.", colors.RED, bold=True)

def map_input_to_file_format(input: int) -> str:
    if input == 1:
        return "CSV"
    elif input == 2:
        return "HTML"
    elif input == 3:
        return "HTTP"
    elif input == 4:
        return "JSON"
    elif input == 5:
        return "GEOJSON"
    elif input == 6:
        return "PDF"
    elif input == 7:
        return "XML"
    
    raise Exception("Cannot map input to file format")

def get_file_format_from_input() -> str:
    """Get the file format from the user input"""    
    print("Available file formats:\n")
    actions = range(1, 8)
    for i in actions:
        print(f"- {i}: {map_input_to_file_format(i)}")

    # get the action
    while True:
        try:
            action = int(input("\nEnter the action: "))
        except Exception as e:
            action = 0
        
        if action in actions:
            return map_input_to_file_format(action)
        
        print("Invalid file format. Please enter one of these: " + (", ".join(actions)))

def initialize_llm(source: str):
    """Initialize a hosted model with appropriate parameters."""
    if source == "google":
        #model_id = 'gemini-3.0-pro'
        #model_id = 'gemini-2.5-pro'
        model_id = 'gemini-2.5-flash'
        #model_id = 'gemini-2.0-flash'
        api_key = set_if_undefined("GOOGLE_API_KEY")

        return ChatGoogleGenerativeAI(
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

        return ChatOpenAI(
            model=model_id,
            temperature=0.5,
            timeout=None,
            max_retries=5,
            api_key=SecretStr(api_key)
        )
    elif source == "anthropic":
        #model_id = 'claude-3-5-sonnet-20240620'
        model_id = 'claude-3-7-sonnet-20250219'
        api_key = set_if_undefined("ANTHROPIC_API_KEY")

        return ChatAnthropic(
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
    
    raise ValueError("Invalid source")
    
def extract_c_code(text: str) -> str:
    """Extract C code from the LLM response."""
    text = text.strip()

    # find code between ```c and ``` markers
    code_blocks = re.findall(r'```c\s*(.*?)```', text, re.DOTALL)
    
    # if not found, try without language specifier
    if not code_blocks:
        code_blocks = re.findall(r'```\s*(.*?)```', text, re.DOTALL)

    # return all blocks joined if any were found
    if code_blocks:
        code_blocks = [ str(code_block).strip() for code_block in code_blocks ]
        code = "\n\n".join(code_blocks)
        return code
    
    # if no code block is found, then return as is
    return text

def compile_c_code(c_file_path: str, out_file_path: str) -> dict[str, bool | str]:
    """Compile the C code using gcc and return compilation result, considering warnings as issues."""
    
    # run gcc to compile the code with all warnings enabled and treated as errors
    # using -Wall and -Wextra flags to enable all warnings and -Werror to treat warnings as errors
    result = run(
        ['gcc', '-Wall', '-Wextra', '-Werror', c_file_path, '-o', out_file_path],
        capture_output=True,
        text=True
    )
    
    # with -Werror, compilation success means executable created and no warnings
    return {
        'success': (result.returncode == 0),  
        'stdout': result.stdout,
        'stderr': result.stderr
    }

def execute_c_code(exe_file_path: str, in_file_path: str) -> dict[str, bool | str]:
    """Execute the compiled C program, feeding it the contents of the input file."""

    try:
        # NB: read bytes, not text (for a general approach that supports all input files)
        # NB: with rb, no encoding must be specified
        with open(in_file_path, "rb") as f:
            input_bytes = f.read()
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': f'Failed to read input file: {e}'
        }

    # Run the executable with the raw-bytes file contents as stdin
    result = run(
        [exe_file_path],
        input=input_bytes,
        capture_output=True,
        text=False
    )

    # Decode stdout/stderr only for human-readable messages
    def safe_decode(b: bytes) -> str:
        try:
            return b.decode("utf-8")
        except:
            return repr(b)

    return {
        'success': (result.returncode == 0),
        'stdout': safe_decode(result.stdout),
        'stderr': safe_decode(result.stderr)
    }

def compilation_check(text: str) -> dict[str, bool | str]:
    """Function that checks if C code compiles correctly without warnings."""
    # clean the code by removing markdown delimiters: remove ```c from the top, remove ``` from anywhere and trim whitespaces
    #code = text.replace("```c", "").replace("```", "").strip()
    code = extract_c_code(text)
    
    # create a temporary directory
    with TemporaryDirectory() as temp_dir:
        temp_name_file = "tool_test"
        temp_c_file = os.path.join(temp_dir, f"{temp_name_file}.c")
        temp_out_file = os.path.join(temp_dir, f"{temp_name_file}.out")

        # write code to file
        with open(temp_c_file, "w", encoding="utf-8") as f:
            f.write(code)

        # compile the C code
        result = compile_c_code(temp_c_file, temp_out_file)

    return result

def execution_check(text: str, format: str) -> dict[str, bool | str]:
    """Function that checks if C code executes correctly without warnings."""
    # clean the code by removing markdown delimiters: remove ```c from the top, remove ``` from anywhere and trim whitespaces
    #code = text.replace("```c", "").replace("```", "").strip()
    code = extract_c_code(text)

    # create a temporary directory
    with TemporaryDirectory() as temp_dir:
        temp_name_file = "tool_test"
        temp_c_file = os.path.join(temp_dir, f"{temp_name_file}.c")
        temp_out_file = os.path.join(temp_dir, temp_name_file)

        # write code to file
        with open(temp_c_file, "w", encoding="utf-8") as f:
            f.write(code)

        # compile the C code
        result = compile_c_code(temp_c_file, temp_out_file)
        
        # prepare the result
        if result["success"]:
            # execute the C code
            test_file_path = f"input/{format}/test.{format}"
            result = execute_c_code(temp_out_file, test_file_path)
    
    return result

def print_colored(text: str, color_code: str, bold: bool = False) -> None:
    """Print text with color."""
    if bold:
        color_code = f"1;{color_code}"
    print(f"\033[{color_code}m{text}\033[0m")

def log(file, text: str, color_code: str | None = None, bold: bool = False) -> None:
    text = f"\n{text}"
    print_colored(text, color_code, bold) if color_code else print(text)
    file.write(f"{text}\n")

def get_parser_requirements() -> str:
    return """1. Input Handling: The code deals with a pointer to a buffer of bytes or a file descriptor for reading unstructured data.
2. Internal State: The code maintains an internal state in memory to represent the parsing state that is not returned as output.
3. Decision-Making: The code takes decisions based on the input and the internal state.
4. Data Structure Creation: The code builds a data structure representing the accepted input or executes specific actions on it.
5. Outcome: The code returns either a boolean value or a data structure built from the parsed data indicating the outcome of the recognition.
6. Composition: The code behavior is defined as a composition of other parsers. (Note: This requirement is only necessary if one of the previous requirements are not met. If ALL the previous 5 requirements are satisfied, this requirement becomes optional.)"""
