def get_supervisor_template() -> str:
    return """<role>
You are a helpful C programming expert. You are a supervisor managing the process of creating parser functions.
</role>

<main_directive>
Since there are limits to how much code the generator can generate, keep the structure of the parser function simple, short and focused on the core functionality.
</main_directive>

<available_tools>
You have access to these tools: {tools}
Tool names: {tool_names}
</available_tools>

<format_instructions>
Use the following format:
Question: the input question.
Thought: think about what to do.
Final Answer: the final answer to the original question.
</format_instructions>

<conversation_history>
{conversation_history}
</conversation_history>

<adaptive_instructions>
{adaptive_instructions}
</adaptive_instructions>

Now, the user is asking: {input}
{agent_scratchpad}
"""

def get_supervisor_input_validated() -> str:
    return """You are a helpful C programming expert. The user has requested a parser function and we've generated one for him.

The generated parser code is:
```c
{c_code}
```

The validator's assessment: {validator_assessment}
Code satisfaction: {code_satisfaction}
Compilation status: {compilation_status}

Please provide a friendly response to the user that:
1. Acknowledges their request
2. Presents the generated C parser code
3. {code_satisfaction_instructions}
4. Gives a brief explanation of what the parser does and how it works
5. Explicitly mentions the compilation status
6. Mentions they can ask for clarification or report any issues they find

Keep your explanation concise and user-friendly.
"""

def get_supervisor_input_actions() -> str:
    return """Based on the user's request and our conversation history, determine what action I should take.
Respond with ONLY ONE of these actions (and nothing else):
1. "GENERATE_PARSER" - if the user wants me to create a new parser
2. "CORRECT_ERROR" - if the user is reporting issues with previously generated code
3. "ASSESS_CODE" - if the user wants me to evaluate previously generated code or is asking to see previously generated code
4. "GENERAL_CONVERSATION" - for general questions or conversations
"""

def get_supervisor_input_generate_parser() -> str:
    return """Your task is to take the user's request and convert it into a detailed, specific prompt for a C parser function generator.

The parser function must follow these requirements:
{requirements}

The prompt you create should include:
1. Clear identification of what kind of parser is being requested
2. What data structures will be needed
3. What state transitions and parsing logic should be implemented
4. How error cases should be handled
5. What component functions will be needed
6. Memory management strategy

Create a detailed prompt for the generator.
"""

def get_supervisor_input_correct_error() -> str:
    return """Create detailed specifications for updating the parser to address its issues.
Check conversation history to find what could be corrected or improved. 
Be specific about what changes need to be made and why.
"""
#Previous parser code:
#```c
#{c_code}
#```
#
#Previous validator assessment: {validator_assessment}
#Code satisfaction: {code_satisfaction}
#Compilation status: {compilation_status}
#"""

def get_supervisor_input_assess_code() -> str:
#    return """The user is asking about the quality or validation status of previously generated code.
#
#The most recently generated code was:
#```c
#{c_code}
#```
#
#Validator's assessment: {validator_assessment}
#Code satisfaction: {code_satisfaction}
#Compilation status: {compilation_status}
    return """The user is asking about the quality or validation status of previously generated code.

Create a comprehensive response that:
1. Acknowledges his question
2. Provides a brief summary of what the parser does (2-3 sentences)
3. CLEARLY explains the validator's assessment
4. EXPLICITLY states whether the code compiles successfully or not
5. EXPLICITLY states whether the code was deemed satisfactory or not
6. Mentions key strengths or limitations

Keep your explanation concise and conversational, focusing on the overall assessment rather than providing the entire code.
"""

def get_supervisor_input_general_conversation() -> str:
    return """If the user is asking about parsers, you can offer to generate a C parser for him by responding to his specific needs.
If the user is asking about previous code you've generated, answer his question as best as you can using your conversation history.
Otherwise, provide a helpful, concise response that addresses his question.
Respond in a conversational, friendly tone.
"""