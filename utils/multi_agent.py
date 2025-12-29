from pathlib import Path

def get_parser_dir(session_dir: Path, round_number: int, iteration_number: int) -> Path:
    iteration_number_str = str(iteration_number).zfill(2)
    return session_dir / f"parser_{round_number}_{iteration_number_str}"

def is_satisfactory(assessment: str) -> bool:
    assessment = assessment.lower()
    # NB: this condition imply some constraints on agent's prompt to manage its output (bad)
    return ("satisfactory" in assessment) and ("not satisfactory" not in assessment)

def map_input_to_action(input: int) -> str:
    if input == 1:
        # for creating a new parser
        return "GENERATE_PARSER"
    elif input == 2:
        # for reporting issues with previously generated code
        return "CORRECT_ERROR"
    elif input == 3:
        # for asking to evaluate or see previously generated code
        return "ASSESS_CODE"
    elif input == 4:
        # for general questions or conversations
        return "GENERAL_CONVERSATION"
    elif input == 5:
        return "EXIT"
    
    raise Exception("Cannot map input to action")

def get_action_from_input() -> str:
    """Get the action from the user input"""
    
    print("Available actions:\n")
    actions = range(2, 6)
    for i in actions:
        print(f"- {i}: {map_input_to_action(i)}")

    # get the action
    while True:
        try:
            action = int(input("\nEnter the action: "))
        except Exception as e:
            action = 0
        
        if action in actions:
            return map_input_to_action(action)
        
        print("Invalid action. Please enter one of these: " + (", ".join(actions)))

def get_request_from_action(action: str, file_format: str) -> str | None:
    """Get the request from the user action"""
    # TODO: optimize the fixed prompts

    if action == "EXIT":
        return None
    if action == "GENERATE_PARSER":
        return f"Generate a parser function for {file_format} files."
    elif action == "CORRECT_ERROR":
        return "Correct the problems on the generated parser."
    
    return input("\nYou: ")