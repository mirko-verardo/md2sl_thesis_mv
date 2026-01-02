from datetime import datetime
from operator import add
from pathlib import Path
from typing_extensions import TypedDict
from typing import Annotated, Sequence, Any, Literal, TypeAlias
from langchain_core.messages import BaseMessage
from langchain.tools import Tool
from utils.general import compilation_check, execution_check



class BenchmarkMetrics:
    """Class for benchmark metrics recording."""
    def __init__(self, n: int, type: str, file_format: str, llm: str):
        self.checkpoints = []
        self.data = {
            "n": n,
            "type": type,
            "file_format": file_format,
            "llm": llm,
            "start_time": datetime.now().isoformat(),
            "compilation_time": None,
            "compilation_iteration": None,
            "testing_time": None,
            "testing_iteration": None,
            "validation_time": None,
            "validation_iteration": None,
            "end_time": None,
            "best_parser_folder": None,
            "testing_rate": None,
            "cyclomatic_complexity": None,
            "code_coverage": None
        }
    
    def __record_parser_checkpoint(self, checkpoint: str, iteration: int, parser_folder: Path) -> bool:
        """Record checkpoint about the parser."""
        if checkpoint in self.checkpoints:
            return False
        self.checkpoints.append(checkpoint)
        self.data[f"{checkpoint}_time"] = datetime.now().isoformat()
        self.data[f"{checkpoint}_iteration"] = iteration
        self.data["best_parser_folder"] = str(parser_folder)
        return True

    def record_parser_compilation(self, iteration: int, parser_folder: Path) -> bool:
        return self.__record_parser_checkpoint("compilation", iteration, parser_folder)
    
    def record_parser_testing(self, iteration: int, parser_folder: Path) -> bool:
        return self.__record_parser_checkpoint("testing", iteration, parser_folder)
    
    def record_parser_validation(self, iteration: int, parser_folder: Path) -> bool:
        return self.__record_parser_checkpoint("validation", iteration, parser_folder)
    
    def record_parser_end(self) -> None:
        self.data["end_time"] = datetime.now().isoformat()
    
    def get_benchmark(self) -> dict[str, Any]:
        return self.data

AgentType: TypeAlias = Literal["Supervisor", "Orchestrator", "Generator", "Compiler", "Tester", "Assessor", "FINISH"]

class AgentState(TypedDict):
    """State schema for the agent graph."""
    messages: Annotated[Sequence[BaseMessage], add]
    user_action: Literal["GENERATE_PARSER", "CORRECT_ERROR", "ASSESS_CODE", "GENERAL_CONVERSATION"]
    user_request: str
    file_format: Literal["CSV", "HTML", "HTTP", "JSON", "GEOJSON", "PDF", "XML"]
    supervisor_specifications: str | None
    generator_code: str | None
    compiler_result: dict[str, bool | str] | None
    tester_result: dict[str, bool | str] | None
    code_assessment: str | None
    round: int
    iteration_count: int
    max_iterations: int
    model_source: str
    next_step: AgentType
    session_dir: Path
    benchmark_metrics: BenchmarkMetrics
    last_parser: dict[str, str] | None

# Define tools

CompilationCheck = Tool(
    name="compilation_check", 
    func=compilation_check, 
    description="""This tool checks if the provided C code compiles correctly without any warnings.
    Input should be valid C code.
    The tool will return compilation results, including any errors or warnings.
    Use this tool to verify that your C parser implementation is syntactically correct and free of warnings before providing it to the user."""
)

def ExecutionCheck(format: str) -> Tool:
    def execution_check_format(code: str) -> str:
        return execution_check(code, format)

    return Tool(
        name="execution_check",
        func=execution_check_format,
        description="""This tool compiles and executes C code using predefined test cases.
        Input should be valid C code."""
    )
    