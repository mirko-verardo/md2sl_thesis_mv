from utils.general import get_model_source_from_input
from utils.single_agent import setup_agent, start_chat



if __name__ == "__main__":
    # Prompt user to enter the model source directly
    source = get_model_source_from_input()

    # Define folders and agent
    folder_name = source + '/few_shot'
    agent_executor = setup_agent(source, few_shot=True)

    # Let's start
    start_chat(folder_name, agent_executor)
