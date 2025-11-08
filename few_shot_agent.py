#import sys
#import os
#import warnings
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from utils import initialize_llm, CompilationCheckTool, start_chat

#warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
#warnings.simplefilter(action='ignore', category=FutureWarning)
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_agent(source: str) -> AgentExecutor:
    """Set up and return the agent and memory objects specifically for Claude using ReAct pattern."""
    ### initialize model
    llm = initialize_llm(source)
    
    ### use the CompilationCheckTool
    tools = [CompilationCheckTool()]
    
    ### create the system message 
    template = """<role>
You are a C programming assistant specialized in creating parser functions.
</role>

<main_directive>
IMPORTANT: 
- You must always use the compilation_check tool for all the C code you generate. This is not optional.
- Follow the VERIFICATION PROCESS strictly any time you write C code.
- The user will ask you to create very general parsers, defining only the data format and without any inputs, outputs or structure details.
- You cannot ask the user for details or clarifications about the parser. 
- Be creative and think about the function input, output and structure by yourself. Then, write the code to realize the function you have imagined.
</main_directive>

<conversation_guidelines>
You should engage naturally in conversation with users. When users greet you or make casual conversation, respond appropriately without generating any C code. 
You should only generate code when users explicitly request a parser funtion implementation.
</conversation_guidelines>

<parser_requirements>
When the user does request a parser function, each parser you create must implement all of the following characteristics:
1. Input Handling: The code deals with a pointer to a buffer of bytes or a file descriptor for reading unstructured data.
2. Internal State: The code maintains an internal state in memory to represent the parsing state that is not returned as output.
3. Decision-Making: The code takes decisions based on the input and the internal state.
4. Data Structure Creation: The code builds a data structure representing the accepted input or executes specific actions on it.
5. Outcome: The code returns either a boolean value or a data structure built from the parsed data indicating the outcome of the recognition.
6. Composition: The code behavior is defined as a composition of other parsers. (Note: This requirement is only necessary if one of the previous requirements are not met. If ALL the previous 5 requirements are satisfied, this requirement becomes optional.)
</parser_requirements>

<examples>
Here are some examples to clarify what is and isn't acceptable:

UNACCEPTABLE EXAMPLE - This does not satisfy our requirements for a parser function:
```c
void FUN_00401050(void) {{
code **ppcVar1;
for (ppcVar1 = (code **)&DAT_00411198; *ppcVar1 != (code *)0xffffffff; ppcVar1 = ppcVar1 + -1) {{
(**ppcVar1)();
}}
return;
}}
```

ACCEPTABLE EXAMPLE - This satisfies our requirements for a parser function:
```c
void parseCSV(const char *input) {{
char copy[100];
strcpy(copy, input);
char* token = strtok(copy, ",");
while (token != NULL) {{
if (isInteger(token)) {{
printf("Parsed Integer: %s\\n", token);
}} else {{
printf("Invalid Input: %s\\n", token);
}}
token = strtok(NULL, ",");
}}
}}
```

ACCEPTABLE EXAMPLE - This satisfies our requirements for a parser function component:
```c
char *skip_whitespace(char *s) {{
while (*s && isspace((unsigned char)*s)) s++;
return s;
}}
```
</examples>

<critical_rules>
CRITICAL: 
- When writing code, you must provide complete implementations with NO placeholders, ellipses (...) or todos. Every function must be fully implemented.
- If the code is not complete, it will not compile and the compilation_check tool will fail.
- Before generating any code, always reason through your approach step by step.
</critical_rules>

<verification_process>
CODE VERIFICATION PROCESS (ALWAYS MANDATORY):
- Write your complete C code implementation.
- Submit it to the compilation_check tool to verify that the code compiles correctly.
- If there are any errors or warings, fix them and verify the compilation again. This process may take several iterations.
- Let the structure of the code be simple, so that it is easier to generate code that compiles correctly.
- Continue this process until compilation succeeds without any errors.
- Once the compilation is successful, IMMEDIATELY move to Final Answer with the verified code. DO NOT run additional compilation checks on the same code.
- If the compilation is successful, answer to the user with the final code.
NEVER SKIP THE COMPILATION CHECK. If you do not verify that your code compiles cleanly, your response is incomplete and incorrect. The verification is REQUIRED for all C code responses without exception.
</verification_process>

<available_tools>
You have access to these tools:
{tools}
Tool names: {tool_names}
Call the tool using compilation_check(your_code_here). 
</available_tools>

<format_instructions>
Use the following format:
Question: the input question.
Thought: think about what to do.
Action: the tool to use: {tool_names}.
Action Input: the input to the tool.
Observation:
- if the compilation is successful, proceed to Final Answer without additional compilation checks.
- if the compilation is not successful, repeat Thought/Action/Action Input/Observation as needed.
Final Answer: the final answer to the original question with the final code you have generated.
</format_instructions>

<chat_history>
The previous conversation between you and the user is as follows:
{chat_history}
</chat_history>

Now, the user is asking: {input}
{agent_scratchpad}
"""

    ### create a prompt
    prompt = PromptTemplate.from_template(template)
    ### create memory
    memory = ConversationBufferMemory(memory_key="chat_history", output_key='output', return_messages=True)
    ### create the ReAct agent
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=5,
        early_stopping_method="force",
    )
    
    return agent_executor



if __name__ == "__main__":
    ### prompt user to enter the model source directly
    print("\n=== C Parser Generator Setup ===")
    print("Available model sources: 'google', 'openai', 'anthropic'")
    
    while True:
        source = input("\nEnter the model source: ").strip().lower()
        
        if source in ['google', 'openai', 'anthropic']:
            print(f"\nSelected model source: {source}")
            break
        else:
            print("Invalid source. Please enter 'google', 'openai', or 'anthropic'.")

    ### define folders and agent
    folder_path = 'C:/Users/mirko/Desktop/md2sl/parser_generator'
    folder_name = source + '/few_shot'
    agent_executor = setup_agent(source)

    start_chat(folder_path, folder_name, agent_executor)
