import os, re
from datetime import datetime
from dotenv import load_dotenv
from getpass import getpass
from pathlib import Path
from pydantic import SecretStr
from subprocess import run, TimeoutExpired
from tempfile import TemporaryDirectory
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from utils import colors



def get_c_parser_path(parser_path: Path) -> Path:
    return parser_path / "source.c"

def get_o_parser_path(parser_path: Path, runtime: bool) -> Path:
    return parser_path / f"{"runtime" if runtime else "buildtime"}"

def __get_in_parser_path(format: str) -> Path:
    format = format.lower()
    return Path("input") / format / f"test.{format}"

def __check_if_wsl(wsl: str) -> bool:
    return (wsl.lower() != "none")

def __to_wsl_path(wsl: str, path: str) -> str:
    # Use wslpath to convert automatically
    result = run(["wsl", "-d", wsl, "wslpath", re.escape(path)], capture_output=True, text=True, encoding="utf-8")
    return result.stdout.strip()

def __get_wsl_cmd(wsl: str) -> list[str]:
    return ["wsl", "-d", wsl]

def __get_stderr_beautified(stderr: str, c_path: str, o_path: str | None = None) -> str:
    """Avoiding confusion for LLM on file name referred in stderr with better line and column number specification."""
    if not stderr:
        return stderr
    
    c_path_esc = re.escape(c_path)

    # NB: order matters!

    pattern = rf"{c_path_esc}:(\d+):(\d+)"
    stderr = re.sub(pattern, r"Line \1 Column \2", stderr)

    pattern = rf"{c_path_esc}:(\d+)"
    stderr = re.sub(pattern, r"Line \1", stderr)

    pattern = f"{c_path_esc}: "
    stderr = re.sub(pattern, "", stderr)

    if o_path:
        pattern = re.escape(o_path)
        stderr = re.sub(pattern, "exe", stderr)

    return stderr

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

def get_parser_dir(session_dir: Path, round_number: int, iteration_number: int) -> Path:
    iteration_number_str = str(iteration_number).zfill(2)
    return session_dir / f"parser_{round_number}_{iteration_number_str}"

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

def create_session(source: str, type: str, format: str) -> Path:
    """Create a session directory with timestamp and return its path."""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path("output") / source / type / format.lower() / f"session_{session_id}"
    session_dir.mkdir(parents=True)
    
    print_colored(f"\nCreated session directory: {session_dir}", colors.CYAN, bold=True)
    
    return session_dir

def initialize_llm(source: str, temp: float = 0.5, timeout: int = 60 * 20, max_retries: int = 3):
    """Initialize a hosted model with appropriate parameters."""
    if source == "google":
        #model_id = "gemini-2.0-flash"
        model_id = "gemini-2.5-flash"
        api_key = set_if_undefined("GOOGLE_API_KEY")

        return ChatGoogleGenerativeAI(
            model = model_id,
            temperature = temp,
            max_tokens = None,
            timeout = timeout,
            max_retries = max_retries,
            api_key = SecretStr(api_key)
        )
    elif source == "openai":
        #model_id = "gpt-4o-mini"
        model_id = "gpt-5-nano"
        api_key = set_if_undefined("OPENAI_API_KEY")

        return ChatOpenAI(
            model = model_id,
            temperature = temp,
            max_tokens = None,
            timeout = timeout,
            max_retries = max_retries,
            api_key = SecretStr(api_key)
        )
    elif source == "anthropic":
        #model_id = "claude-3-7-sonnet-20250219"
        model_id = "claude-sonnet-4-20250514"
        api_key = set_if_undefined("ANTHROPIC_API_KEY")

        return ChatAnthropic(
            model = model_id,
            temperature = temp,
            # can't be None with Anthropic :(
            max_tokens = 8192,
            timeout = timeout,
            max_retries = max_retries,
            api_key = SecretStr(api_key)
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

def compile_c_code(parser_path: Path, parser_code: str, runtime: bool = True) -> dict[str, bool | str]:
    """Compile the C code using gcc with strict optimization, warnings and hardening."""

    runtime_flags = [
        "-O1",
        "-g",
        # Sanitizers (heavy): address -> ASan, undefined (behaviour) -> UBSan, leak -> LSan, thread -> TSan
        "-fsanitize=address,undefined",
        # leak can be appliead on top of ASan with detect_leaks=1 in ASAN_OPTIONS
        #"-fsanitize=address,undefined,leak",
        # DO NOT combine thread with address/leak
        #"-fsanitize=address,undefined,leak,thread",
        "-fno-omit-frame-pointer",
        "-fno-optimize-sibling-calls",
        "-fno-common"
    ]

    buildtime_flags = [
        # OPTIMIZATION
        "-O2",

        # WARNINGS
        # all warnings as compilation errors
        "-Werror",

        # FORTIFY
        # enables glibc bounds checking at runtime
        "-U_FORTIFY_SOURCE", "-D_FORTIFY_SOURCE=3",

        # HARDENING
        # enables run-time checks for stack-based buffer overflows for all functions
        #"-fstack-protector-all",
        # enables run-time checks for stack-based buffer overflows using strong heuristic (performance balanced)
        "-fstack-protector-strong",
        # enables Control-Flow Enforcement Technology (CET) (problematic if not x86_64 and not Kernel Linux)
        "-fcf-protection=none", # must be added before desired value (not required since GCC 14.0.0)
        "-fcf-protection=full",
        # prevents data leakage with zero-initializing padding bits (since GCC 15.0.0)
        #"-fzero-init-padding-bits=all",
        # initializes automatic variables that lack explicit initializers
        "-ftrivial-auto-var-init=zero",
        # static analysis flags
        "-fanalyzer",
        "-fanalyzer-transitivity"
    ]

    compiler_flags = [
        # WARNINGS
        # common warnings, additional warnings
        "-Wall", "-Wextra", 
        # detects unsafe or incorrect printf-style formatting (both must be specified for GCC vs Clang reason)
        "-Wformat", "-Wformat=2",
        # detects implicit type conversions that may change value, including signed/unsigned
        "-Wconversion", "-Wsign-conversion", 
        # detects when trampolines are generated (security risk) (privilege escalation?)
        "-Wtrampolines",
        # detects missing break in switch
        "-Wimplicit-fallthrough",
        # detects Unicode bidirectional characters (Trojan Source attacks)
        "-Wbidi-chars=any,ucn",

        # HARDENING
        # enforces correct use of flexible array members
        "-fstrict-flex-arrays=3",
        # prevents stack clash attacks
        "-fstack-clash-protection",
        # builds a position-independent executable
        "-fPIE",
        # forces retention of null pointer checks
        "-fno-delete-null-pointer-checks",
        # defines behavior for signed integer and pointer arithmetic overflows
        "-fno-strict-overflow",
        # assumes not strict aliasing
        "-fno-strict-aliasing"
    ] + (runtime_flags if runtime else buildtime_flags)

    linker_flags = [
        # restricts dlopen(3) calls to shared objects + marks stack memory as non-executable (-z not supported on Windows)
        #"-Wl,-z,nodlopen,-z,noexecstack",
        # marks relocation table entries resolved at load-time as read-only (-z not supported on Windows)
        #"-Wl,-z,relro,-z,now",
        # allows linker to omit libraries specified on the command line if they are not used
        "-Wl,--as-needed",
        # stop linker from resolving symbols in produced binary to transitive dependencies
        "-Wl,--no-copy-dt-needed-entries",
        # build a position-independent binary
        "-pie"
    ]

    c_parser_path = get_c_parser_path(parser_path)
    o_parser_path = get_o_parser_path(parser_path, runtime)

    with open(c_parser_path, "w", encoding="utf-8") as f:
        f.write(parser_code)

    c_parser_path_str = str(c_parser_path)
    o_parser_path_str = str(o_parser_path)
    command = []
    wsl = set_if_undefined("WSL")
    if __check_if_wsl(wsl):
        c_parser_path_str = __to_wsl_path(wsl, c_parser_path_str)
        o_parser_path_str = __to_wsl_path(wsl, o_parser_path_str)
        command = __get_wsl_cmd(wsl)

    try:
        result = run(
            [*command, "gcc", *compiler_flags, c_parser_path_str, *linker_flags, "-o", o_parser_path_str],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            timeout = 60 * 5
        )
    except TimeoutExpired as te:
        return {
            'success': False,  
            'stdout': '',
            'stderr': f'Failed to compile the code: {te}'
        }

    compilation_stdout = result.stdout
    compilation_stderr = __get_stderr_beautified(result.stderr, c_parser_path_str)
    
    return {
        'success': (result.returncode == 0),  
        'stdout': compilation_stdout,
        'stderr': compilation_stderr
    }

def execute_c_code(parser_path: Path, parser_format: str, runtime: bool = True) -> dict[str, bool | str]:
    """Execute the compiled C program, feeding it the contents of the input file."""

    try:
        # NB: read bytes, not text (for a general approach that supports all input files)
        # NB: with rb, no encoding must be specified
        in_parser_path = __get_in_parser_path(parser_format)
        with open(in_parser_path, "rb") as f:
            input_bytes = f.read()
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': f'Failed to read input file: {e}'
        }

    # Execution command building
    c_parser_path_str = str(get_c_parser_path(parser_path))
    o_parser_path_str = str(get_o_parser_path(parser_path, runtime))
    command = []
    wsl = set_if_undefined("WSL")
    if __check_if_wsl(wsl):
        c_parser_path_str = __to_wsl_path(wsl, c_parser_path_str)
        o_parser_path_str = __to_wsl_path(wsl, o_parser_path_str)
        command = __get_wsl_cmd(wsl)
    if runtime:
        asan_options = [
            "detect_leaks=1",
            "strict_string_checks=1",
            "detect_stack_use_after_return=1",
            "check_initialization_order=1",
            "strict_init_order=1"
        ]
        command += ["env", f"ASAN_OPTIONS={":".join(asan_options)}"]
    
    try:
        # Run the executable with the raw-bytes file contents as stdin and with asan options (maybe)
        # NB: no need for encoding because text is equal to False
        result = run(
            [*command, o_parser_path_str],
            input = input_bytes,
            capture_output = True,
            text = False,
            timeout = 60 * 5
        )
    except TimeoutExpired as te:
        return {
            'success': False,  
            'stdout': '',
            'stderr': f'Failed to execute the code, probably due to an infinite loop: {te}'
        }

    # Decode stdout/stderr only for human-readable messages
    def safe_decode(b: bytes) -> str:
        try:
            return b.decode("utf-8")
        except:
            return repr(b)
    
    execution_stdout = safe_decode(result.stdout)
    execution_stderr = safe_decode(result.stderr)
    execution_stderr = __get_stderr_beautified(execution_stderr, c_parser_path_str, o_parser_path_str)

    return {
        'success': (result.returncode == 0),
        'stdout': execution_stdout,
        'stderr': execution_stderr
    }

def analyze_c_code(parser_path: Path, parser_format: str) -> str:
    """Analyze the C code coverage."""

    try:
        # NB: read bytes, not text (for a general approach that supports all input files)
        # NB: with rb, no encoding must be specified
        in_parser_path = __get_in_parser_path(parser_format)
        with open(in_parser_path, "rb") as f:
            input_bytes = f.read()
    except Exception as e:
        return f"Failed to read input file: {e}"

    coverage_flags = [
        # strongly recommended for accurate coverage
        "-O0",
        # code coverage flags
        "-fprofile-arcs",
        "-ftest-coverage"
    ]

    c_parser_path = get_c_parser_path(parser_path)
    o_parser_path = parser_path / "analyze"

    c_parser_path_str = str(c_parser_path)
    o_parser_path_str = str(o_parser_path)
    
    command = []
    wsl = set_if_undefined("WSL")
    if __check_if_wsl(wsl):
        c_parser_path_str = __to_wsl_path(wsl, c_parser_path_str)
        o_parser_path_str = __to_wsl_path(wsl, o_parser_path_str)
        command = __get_wsl_cmd(wsl)

    try:
        timeout = 60 * 5

        # Compile
        result = run(
            [*command, "gcc", *coverage_flags, c_parser_path_str, "-o", o_parser_path_str],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            timeout = timeout
        )
        if result.stderr:
            raise Exception(result.stderr)
        
        # Execute
        result = run(
            [*command, o_parser_path_str],
            input = input_bytes,
            capture_output = True,
            text = False,
            timeout = timeout
        )
        if result.stderr:
            raise Exception(result.stderr)

        # Analyze
        result = run(
            [*command, "gcov", "--no-output", f"{o_parser_path_str}-{c_parser_path.name.replace(".c", "")}"],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            timeout = timeout
        )
        if result.stderr:
            raise Exception(result.stderr)
    except Exception as e:
        return f"Failed to analyze the code: {e}"
    
    return result.stdout

def compilation_check(text: str) -> dict[str, bool | str]:
    """Function that checks if C code compiles correctly without warnings."""
    # extract the code
    code = extract_c_code(text)
    
    # create a temporary directory
    with TemporaryDirectory() as temp_dir:
        # compile the code
        result = compile_c_code(Path(temp_dir), code, runtime=False)
        if result["success"]:
            result = compile_c_code(Path(temp_dir), code)

    return result

def execution_check(text: str, format: str) -> dict[str, bool | str]:
    """Function that checks if C code executes correctly without warnings."""
    # extract the code
    code = extract_c_code(text)

    # create a temporary directory
    with TemporaryDirectory() as temp_dir:
        # compile the code
        result = compile_c_code(Path(temp_dir), code)
        
        # prepare the result
        if result["success"]:
            # execute the code
            result = execute_c_code(Path(temp_dir), format)
    
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
