from datetime import datetime
from json import dump
from operator import add
from pathlib import Path
from typing_extensions import TypedDict
from typing import Annotated, Sequence, Any, Literal, TypeAlias
from langchain_core.messages import BaseMessage
from langchain.tools import Tool
from utils.general import compilation_check, execution_check



class SystemMetrics:
    """Class for tracking system interactions and metrics."""
    def __init__(self):
        self.rounds = 0  # Complete rounds
        self.current_round = None  # Track the current round ID
        self.rounds_data = {}  # Detailed data for each round
    
    def get_round_number(self) -> int:
        return self.rounds

    def record_parser_compilation(self, parser_file: str, compilation_success: bool) -> None:
        """Record information about the last compilated parser file."""
        if self.current_round:
            # Name of the last parser file
            self.rounds_data[self.current_round]["last_parser_file"] = parser_file
            # Compilation status of the last parser
            self.rounds_data[self.current_round]["compilation_success"] = compilation_success
    
    def record_parser_testing(self, testing_success: bool) -> None:
        """Record information about the last tested parser file."""
        if self.current_round:
            # Testing status of the last parser
            self.rounds_data[self.current_round]["testing_success"] = testing_success

    def start_new_round(self, user_request: str) -> str:
        """Start tracking a new round."""
        self.rounds += 1
        round_id = f"round_{self.rounds}"
        self.current_round = round_id
        self.rounds_data[round_id] = {
            "user_request": user_request,
            "generator_interactions": 0,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "completed": False,
            "satisfied": False,
            "last_parser_file": None,
            "compilation_success": None,
            "testing_success": None
        }
        return round_id
        
    def increment_generator_interaction(self) -> None:
        """Increment the count of generator interactions for current round."""
        if self.current_round:
            self.rounds_data[self.current_round]["generator_interactions"] += 1
    
    def complete_round(self) -> None:
        """Mark the current round as completed."""
        if self.current_round:
            self.rounds_data[self.current_round]["end_time"] = datetime.now().isoformat()
            self.rounds_data[self.current_round]["completed"] = True
    
    def satisfy_round(self) -> None:
        """Mark the current round as completed."""
        if self.current_round:
            self.rounds_data[self.current_round]["satisfied"] = True
            
    def generate_summary(self) -> dict[str, Any]:
        """Generate only JSON summary data without text output."""
        return {
            "total_rounds": self.rounds,
            "rounds_data": self.rounds_data
        }

    def save_summary(self, session_dir: Path) -> None:
        """Save only the JSON metrics summary to a file."""
        json_data = self.generate_summary()
        
        json_file = session_dir / "system_metrics.json"
        with open(json_file, "w", encoding="utf-8") as f:
            dump(json_data, f, indent=2)

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
    iteration_count: int
    max_iterations: int
    model_source: str
    next_step: AgentType
    session_dir: Path
    system_metrics: SystemMetrics
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

def ExecutionCheck(file_format: str) -> Tool:
    def execution_check_format(code: str) -> str:
        return execution_check(code, file_format.lower())

    return Tool(
        name="execution_check",
        func=execution_check_format,
        description="""This tool compiles and executes C code using predefined test cases.
        Input should be valid C code."""
    )
    