import os
import re
from dotenv import load_dotenv
from pathlib import Path
from getpass import getpass
from pydantic import SecretStr
from subprocess import run
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint



# parser requirements
requirements = """1. Input Handling: The code deals with a pointer to a buffer of bytes or a file descriptor for reading unstructured data.
2. Internal State: The code maintains an internal state in memory to represent the parsing state that is not returned as output.
3. Decision-Making: The code takes decisions based on the input and the internal state.
4. Data Structure Creation: The code builds a data structure representing the accepted input or executes specific actions on it.
5. Outcome: The code returns either a boolean value or a data structure built from the parsed data indicating the outcome of the recognition.
6. Composition: The code behavior is defined as a composition of other parsers. (Note: This requirement is only necessary if one of the previous requirements are not met. If ALL the previous 5 requirements are satisfied, this requirement becomes optional.)"""

def set_if_undefined(var: str) -> str:
    """Set environment variable (API KEY) from .env file."""
    # load variables from .env file
    load_dotenv()
    # check if variable exists in environment
    if not os.environ.get(var):
        os.environ[var] = getpass(f"Please provide your {var}")
    
    # get the API key from environment
    api_key = os.environ.get(var) or ""
    return api_key

def get_model_source_from_input() -> str:
    """Get the model source name from the user input"""
    print_colored("\n=== C Parser Generator Setup ===", "1;36")
    print_colored("Available model sources: 'google', 'openai', 'anthropic'", "1;33")

    # speed up
    source = "google"
    print_colored(f"\nSelected model source: {source}", "1;32")
    return source
    
    # get model source
    while True:
        source = input("\nEnter the model source: ").strip().lower()
        
        if source in ['google', 'openai', 'anthropic']:
            print_colored(f"\nSelected model source: {source}", "1;32")
            return source
        else:
            print_colored("Invalid source. Please enter 'google', 'openai', or 'anthropic'.", "1;31")

def initialize_llm(source: str):
    """Initialize a hosted model with appropriate parameters."""
    if source == "huggingface":
        # model_id = 'codellama/CodeLlama-34b-Instruct-hf'
        # https://evalplus.github.io/leaderboard.html
        # model_id = 'microsoft/Phi-3-mini-4k-instruct' # 56esimo
        # model_id = 'meta-llama/Meta-Llama-3-8B-Instruct' # 60eismo
        model_id = 'bigcode/starcoder2-15b' # 79eismo
        api_key = set_if_undefined("HUGGING_FACE_API_KEY")

        return HuggingFaceEndpoint(
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
        #model_id = 'gemini-2.5-pro'
        model_id = 'gemini-2.0-flash'
        #model_id = 'gemini-2.5-flash-lite'
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
    
def extract_c_code(text: str) -> str | None:
    """Extract C code from the LLM response."""
    # find code between ```c and ``` markers
    code_blocks = re.findall(r'```c(.*?)```', text, re.DOTALL)
    
    # if not found, try without language specifier
    if not code_blocks:
        code_blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
    
    # if still no code blocks found, check if the entire text is C code
    # by looking for common C headers or patterns
    if not code_blocks:
        if text.strip().startswith('#include') or re.search(r'int\s+main\s*\(', text):
            return text.strip()
    
    # return the first code block if any were found
    return code_blocks[0].strip() if code_blocks else None

def compile_c_code(c_file_path: Path) -> dict[str, Any]:
    """Compile the C code using gcc and return compilation result, considering warnings as issues."""
    output_file = c_file_path.with_suffix('')
    
    # run gcc to compile the code with all warnings enabled and treated as errors
    result = run(
        ['gcc', '-Wall', '-Wextra', '-Werror', str(c_file_path), '-o', str(output_file)],
        capture_output=True,
        text=True)
    
    # with -Werror, compilation success means no warnings
    return {
        'success': result.returncode == 0,  # compilation success (executable created, no warnings)
        'fully_clean': result.returncode == 0,    # same as success since warnings cause failure with -Werror
        'stdout': result.stdout,
        'stderr': result.stderr,
        'executable': output_file if result.returncode == 0 else None,
        'has_warnings': False  # with -Werror, warnings cause compilation failure, so this will always be False if successful
    }

def print_colored(text: str, color_code: str) -> None:
    """Print text with color."""
    print(f"\033[{color_code}m{text}\033[0m")

def log(file, text: str, color_code: str | None = None, bold: bool = False) -> None:
    if color_code and bold:
        color_code = f"1;{color_code}"
    text = f"\n{text}"
    print_colored(text, color_code) if color_code else print(text)
    file.write(f"{text}\n")
