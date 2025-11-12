from typing import Any
from utils.general import requirements



def get_supervisor_input_validated(
        user_request: str, 
        c_code: str, 
        last_message: str, 
        is_satisfactory: bool,
        compilation_status: str
    ) -> str:
    return f"""You are a helpful C programming expert. The user requested a parser function, and we've generated one for them.

The user's original request was: {user_request}

The generated parser code is:
```c
{c_code}
```

The validator's assessment: {last_message}

Is the code satisfactory? {"Yes" if is_satisfactory else "No"}
Compilation status: {compilation_status}

Please provide a friendly response to the user that:
1. Acknowledges their request
2. Presents the generated C parser code
3. {"" if is_satisfactory else "IMPORTANTLY, mentions that the code is NOT FULLY SATISFACTORY according to the validator, and briefly explains why"}
4. Gives a brief explanation of what the parser does and how it works
5. Explicitly mentions the compilation status
6. Mentions they can ask for clarification or report any issues they find

Keep your explanation concise and user-friendly.
"""

def get_supervisor_input_actions(user_request: str, conversation_context: str) -> str:
    return f"""Based on the user's request and our conversation history, determine what action I should take.
Respond with ONLY ONE of these actions (and nothing else):
1. "GENERATE_PARSER" - if the user wants me to create a new parser
2. "CORRECT_ERROR" - if the user is reporting issues with previously generated code
3. "ASSESS_CODE" - if the user wants me to evaluate previously generated code or is asking to see previously generated code
4. "GENERAL_CONVERSATION" - for general questions or conversations

User's request: "{user_request}"

Recent conversation context: 
{conversation_context}
"""

def get_supervisor_input_generate_parser(user_request: str) -> str:
    return f"""Your task is to take the user's request and convert it into a detailed, specific prompt for a C parser function generator.

The parser function must follow these requirements:
{requirements}

The prompt you create should include:
1. Clear identification of what kind of parser is being requested
2. What data structures will be needed
3. What state transitions and parsing logic should be implemented
4. How error cases should be handled
5. What component functions will be needed
6. Memory management strategy

User request: {user_request}

Create a detailed prompt for the generator.
"""

def get_supervisor_input_correct_error(user_request: str, conversation_context: str, parser: dict[str, Any]) -> str:
    return f"""Create detailed specifications for updating the parser to address these issues.
Be specific about what changes need to be made and why.

Original user request: {user_request}

Previous parser code:
```c
{parser['code']}
```

Previous validator assessment: {parser.get('validator_assessment', 'Not available')}
Compilation status: {parser.get('compilation_status', 'Unknown')}
Was code satisfactory: {"Yes" if parser.get('is_satisfactory', False) else "No"}

User reported issues or requested changes: {user_request}

Conversation history (for context):
{conversation_context}
"""

def get_supervisor_input_assess_code(user_request: str, parser: dict[str, Any]) -> str:
    keywords = ["show", "code", "generated", "memory", "what was the", "see the", "provide the"]
    is_asking_for_code = any(keyword in user_request.lower() for keyword in keywords)
            
    if is_asking_for_code:
        return f"""The user is asking about previously generated code.

User request: {user_request}

The complete most recently generated code is:
```c
{parser['code']}
```

Validator's assessment: {parser.get('validator_assessment', 'No formal assessment available')}
Compilation status: {parser.get('compilation_status', 'Unknown')}
Was code satisfactory: {"Yes" if parser.get('is_satisfactory', False) else "No"}

Create a response that highlights:
1. Acknowledges their request
2. Summarizes what the parser does and its key components (3-5 sentences)
3. Explicitly mentions the validator's assessment
4. Clearly states whether the code compiles successfully
5. States whether the code was deemed satisfactory or not

Focus on explaining the code's functionality rather than showing the entire codebase. Only provide specific code snippets if the user explicitly asks for them.
"""
    else:
        return f"""The user is asking about the quality or validation status of previously generated code.

User request: {user_request}

The most recently generated code was:
```c
{parser['code']}
```

Validator's assessment: {parser.get('validator_assessment', 'No formal assessment available')}
Compilation status: {parser.get('compilation_status', 'Unknown')}
Was code satisfactory: {"Yes" if parser.get('is_satisfactory', False) else "No"}

Create a comprehensive response that:
1. Acknowledges their question
2. Provides a brief summary of what the parser does (2-3 sentences)
3. CLEARLY explains the validator's assessment
4. EXPLICITLY states whether the code compiles successfully
5. CLEARLY states whether the code was deemed satisfactory or not
6. Mentions key strengths or limitations

Keep your explanation concise and conversational, focusing on the overall assessment rather than providing the entire code.
"""

def get_supervisor_input_general_conversation(user_request: str, conversation_context: str, parser: dict[str, Any] | None) -> str:
    keywords = ["memory", "remember", "previous", "code", "generated", "parser"]
    is_asking_about_memory = parser and any(keyword in user_request.lower() for keyword in keywords)
    
    if is_asking_about_memory:                
        return f"""Respond to the user mentioning that you have a recently generated parser in memory. Focus ONLY on the most recent parser. Remember to explicitly mention:
1. Whether the code was satisfactory according to the validator
2. Whether the code compiled successfully
3. A brief summary of what the parser does (if you know this information)

Let them know they can ask to see a summary of the code by saying something like "summarize the code" or "what did the parser do?"

User: {user_request}

Recent conversation history (for context):
{conversation_context}

Information about your most recent parser:
I have a recently generated parser from {parser.get('timestamp')}
- Satisfactory: {parser.get('is_satisfactory', 'Unknown')}
- Compilation: {parser.get('compilation_status', 'Unknown')}
"""
    else:
        return f"""If they're asking about parsers, you can offer to generate a C parser for them by responding to their specific needs.
If they're asking about previous code you've generated, refer ONLY to your most recent parser.
Otherwise, provide a helpful, concise response that addresses their question.

Respond in a conversational, friendly tone.

User: {user_request}

Recent conversation history (for context):
{conversation_context}
"""