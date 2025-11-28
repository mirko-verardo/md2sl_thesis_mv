from datetime import datetime
from json import dump
from operator import add
from pathlib import Path
from typing_extensions import TypedDict
from typing import Annotated, Sequence, Any, Literal
from langchain_core.messages import BaseMessage
from langchain.tools import Tool
from utils import colors
from utils.general import print_colored, compilation_check, execution_check



class SystemMetrics:
    """Class for tracking system interactions and metrics."""
    def __init__(self):
        self.rounds = 0  # Complete rounds (user-supervisor-generator-validator)
        self.current_round = None  # Track the current round ID
        self.rounds_data = {}  # Detailed data for each round
    
    def record_parser_validation(self, parser_file: str, compilation_success: bool, execution_success: bool) -> None:
        """Record information about the last validated parser file."""
        if self.current_round:
            # Name of the last parser file
            self.rounds_data[self.current_round]["last_parser_file"] = parser_file
            # Compilation status of the last parser
            self.rounds_data[self.current_round]["compilation_success"] = compilation_success
            # Execution status of the last parser
            self.rounds_data[self.current_round]["execution_success"] = execution_success

    def start_new_round(self, user_request: str) -> str:
        """Start tracking a new round."""
        self.rounds += 1
        round_id = f"round_{self.rounds}"
        self.current_round = round_id
        self.rounds_data[round_id] = {
            "user_request": user_request,
            "generator_validator_interactions": 0,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "completed": False,
            "last_parser_file": None,
            "compilation_success": None,
            "execution_success": None
        }
        return round_id
        
    def increment_generator_validator_interaction(self) -> None:
        """Increment the count of generator-validator interactions for current round."""
        if self.current_round:
            self.rounds_data[self.current_round]["generator_validator_interactions"] += 1
        
    def record_tool_usage(self, tool: str, success: bool) -> None:
        """Record tool usage for the current interaction with compilation result."""
        if self.current_round:
            interaction_number = self.rounds_data[self.current_round]["generator_validator_interactions"]
            
            interaction_key = f"{tool}_interactions_{interaction_number}"
            print_colored(f"Recording tool usage for {self.current_round}, {interaction_key}", colors.YELLOW, bold=True)
            
            num = self.rounds_data[self.current_round].get(interaction_key, 0)
            self.rounds_data[self.current_round].update({ interaction_key: num + 1 })
            print_colored(f"New tool usage count: {self.rounds_data[self.current_round][interaction_key]}", colors.YELLOW, bold=True)
            
            if success:
                success_key = f"{tool}_successes_{interaction_number}"
                num = self.rounds_data[self.current_round].get(success_key, 0)
                self.rounds_data[self.current_round].update({ success_key: num + 1 })
                print_colored(f"Successful compilations: {self.rounds_data[self.current_round][success_key]}", colors.YELLOW, bold=True)
    
    def complete_round(self) -> None:
        """Mark the current round as completed."""
        if self.current_round:
            self.rounds_data[self.current_round]["end_time"] = datetime.now().isoformat()
            self.rounds_data[self.current_round]["completed"] = True
            
    def generate_summary(self) -> dict[str, Any]:
        """Generate only JSON summary data without text output."""
        json_data = {
            "total_rounds": self.rounds,
            "rounds_data": {}
        }
        
        for round_id, data in self.rounds_data.items():
            compilation_rates = {}
            max = data["generator_validator_interactions"]
            for i in range(1, max + 1):
                for tool in ["compilation_check", "execution_check"]:
                    interaction_key = f"{tool}_interactions_{i}"
                    interaction_num = data.get(interaction_key, 0)
                    if interaction_num > 0:
                        success_key = f"{tool}_successes_{i}"
                        success_num = data.get(success_key, 0)
                        rate_str = f"{success_num}/{interaction_num}"
                        compilation_rates[f"{tool}_rate_{i}"] = rate_str
            
            round_data = {
                "user_request": data["user_request"],
                "generator_validator_interactions": data["generator_validator_interactions"],
                **compilation_rates,
                "last_parser_file": data["last_parser_file"],
                "compilation_success": data["compilation_success"],
                "execution_success": data["execution_success"],
                "start_time": data["start_time"],
                "end_time": data["end_time"],
                "completed": data["completed"]
            }
            
            json_data["rounds_data"][round_id] = round_data
        
        return json_data

    def save_summary(self, session_dir: Path) -> None:
        """Save only the JSON metrics summary to a file."""
        json_data = self.generate_summary()
        
        json_file = session_dir / "system_metrics.json"
        with open(json_file, 'w') as f:
            dump(json_data, f, indent=2)
        
        print_colored(f"\nMetrics saved to: {json_file}", colors.CYAN, bold=True)

class AgentState(TypedDict):
    """State schema for the agent graph."""
    messages: Annotated[Sequence[BaseMessage], add]
    user_action: Literal["GENERATE_PARSER", "CORRECT_ERROR", "ASSESS_CODE", "GENERAL_CONVERSATION"]
    user_request: str
    file_format: Literal["CSV", "HTML", "HTTP", "JSON", "GEOJSON", "PDF", "XML"]
    generator_specs: str | None
    generator_code: str | None
    validator_assessment: str | None
    iteration_count: int
    max_iterations: int
    model_source: str
    next_step: str  # "Supervisor", "Generator", "Validator", or "FINISH"
    session_dir: Path  # path to the session directory
    system_metrics: SystemMetrics  # system interaction metrics

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
    def execution_check_format(c_code: str) -> str:
        return execution_check(c_code, file_format.lower())

    return Tool(
        name="execution_check",
        func=execution_check_format,
        description="""This tool compiles and executes C code using predefined test cases.
        Input should be valid C code."""
    )
    