from utils.general import get_model_source_from_input, get_file_format_from_input
from utils.single_agent import start_chat



if __name__ == "__main__":
    # Prompt user to enter the model source directly
    source = get_model_source_from_input()

    # Prompt user to enter the file format directly
    file_format = get_file_format_from_input()

    # Let's start
    start_chat(source, file_format, few_shot=True)
