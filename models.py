from datetime import datetime
from json import dump
from operator import add
from pathlib import Path
from typing_extensions import TypedDict
from typing import Annotated, Sequence, Any
from langchain_core.messages import BaseMessage
from langchain.tools import Tool
from utils import colors
from utils.general import print_colored, compilation_check



class SystemMetrics:
    """Class for tracking system interactions and metrics."""
    def __init__(self):
        self.rounds = 0  # Complete rounds (user-supervisor-generator-validator)
        self.current_round = None  # Track the current round ID
        self.rounds_data = {}  # Detailed data for each round
        self.last_parser_file = None  # Name of the last parser file
        self.last_compilation_status = None  # Compilation status of the last parser
    
    def record_parser_validation(self, parser_file: str, compilation_success: bool) -> None:
        """Record information about the last validated parser file."""
        if self.current_round:
            self.last_parser_file = parser_file
            self.last_compilation_status = compilation_success
            self.rounds_data[self.current_round]["last_parser_file"] = parser_file
            self.rounds_data[self.current_round]["compilation_success"] = compilation_success

    def start_new_round(self, user_request: str) -> str:
        """Start tracking a new round."""
        self.rounds += 1
        round_id = f"round_{self.rounds}"
        self.current_round = round_id
        self.rounds_data[round_id] = {
            "user_request": user_request,
            "generator_validator_interactions": 0,
            "tool_interaction1": 0,
            "tool_interaction2": 0, 
            "tool_interaction3": 0,
            "successful_compilations1": 0,
            "successful_compilations2": 0,
            "successful_compilations3": 0,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "completed": False,
            "last_parser_file": None,
            "compilation_success": None
        }
        return round_id
        
    def increment_generator_validator_interaction(self) -> None:
        """Increment the count of generator-validator interactions for current round."""
        if self.current_round:
            self.rounds_data[self.current_round]["generator_validator_interactions"] += 1
        
    def record_tool_usage(self, compilation_success: bool) -> None:
        """Record tool usage for the current interaction with compilation result."""
        if self.current_round:
            interaction_number = self.rounds_data[self.current_round]["generator_validator_interactions"]
            
            if 1 <= interaction_number <= 3:
                interaction_key = f"tool_interaction{interaction_number}"
                success_key = f"successful_compilations{interaction_number}"
                
                print_colored(f"Recording tool usage for {self.current_round}, {interaction_key}", colors.YELLOW, bold=True)
                
                self.rounds_data[self.current_round][interaction_key] += 1
                
                if compilation_success:
                    self.rounds_data[self.current_round][success_key] += 1
                
                print_colored(f"New tool usage count: {self.rounds_data[self.current_round][interaction_key]}", colors.YELLOW, bold=True)
                print_colored(f"Successful compilations: {self.rounds_data[self.current_round][success_key]}", colors.YELLOW, bold=True)
            else:
                print_colored(f"Warning: Invalid interaction number {interaction_number}", colors.YELLOW, bold=True)
    
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
            for i in range(1, 4):
                interaction_key = f"tool_interaction{i}"
                success_key = f"successful_compilations{i}"
                
                if interaction_key in data and data[interaction_key] > 0:
                    rate_str = f"{data.get(success_key, 0)}/{data[interaction_key]}"
                    compilation_rates[f"compilation_rate_interaction{i}"] = rate_str
            
            round_data = {
                "user_request": data["user_request"],
                "generator_validator_interactions": data["generator_validator_interactions"],
                **{f"tool_interaction{i}": data.get(f"tool_interaction{i}", 0) for i in range(1, 4)},
                **compilation_rates,
                "last_parser_file": data.get("last_parser_file"),
                "compilation_success": data.get("compilation_success"),
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
    user_action: str
    user_request: str
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
    Use this tool to verify that your C parser implementation is syntactically correct
    and free of warnings before providing it to the user."""
)
    