def get_parser_requirements() -> str:
    return """1. Input Handling: The code deals with a pointer to a buffer of bytes or a file descriptor for reading unstructured data.
2. Internal State: The code maintains an internal state in memory to represent the parsing state that is not returned as output.
3. Decision-Making: The code takes decisions based on the input and the internal state.
4. Data Structure Creation: The code builds a data structure representing the accepted input or executes specific actions on it.
5. Outcome: The code returns either a boolean value or a data structure built from the parsed data indicating the outcome of the recognition.
6. Composition: The code behavior is defined as a composition of other parsers. (Note: This requirement is only necessary if one of the previous requirements are not met. If ALL the previous 5 requirements are satisfied, this requirement becomes optional.)"""

def had_agent_problems(output: str) -> bool:
    # Main problems
    problems = [
        "agent stopped due to max iterations.",
        "agent stopped due to iteration limit or time limit."
    ]

    return output.lower() in problems

def get_satisfaction(assessment: str) -> str:
    assessment = assessment.lower()
    # NB: this condition imply some constraints on validator's prompt to manage its output (bad)
    condition = "satisfactory" in assessment and "not satisfactory" not in assessment
    return "SATISFACTORY" if condition else "NOT SATISFACTORY"

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

def get_action_from_input(start: bool) -> str:
    """Get the action from the user input"""
    # speed up
    if start:
        return map_input_to_action(1)
    
    print("Available actions:\n")
    actions = range(1, 6)
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
