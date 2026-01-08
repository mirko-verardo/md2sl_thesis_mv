from time import sleep



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

def invoke_agent(agent, agent_input: dict[str, str]) -> tuple[bool, str]:
    for i in range(3):
        if i > 0:
            print("Let's wait before restarting...")
            sleep(60)
        try:
            agent_result = agent.invoke(agent_input)
            agent_response = str(agent_result.content)
            return True, agent_response
        except Exception as e:
            agent_response = str(e)
            print(agent_response)
    
    return False, f"Error occurred during agent response: {agent_response}\n\nPlease try again."