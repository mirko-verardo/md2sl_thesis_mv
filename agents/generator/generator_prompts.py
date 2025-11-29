def get_generator_template() -> str:
    """Generator's template with no ReAct format"""

    return """<role>
You are a specialized C programming agent that creates complete parser functions following strict requirements and specifications.
</role>

<main_directive>
- The C code you generate cannot have references to external C libraries.
- Keep your code simple, short and focused on the core functionality, so it will be easier that the generated code compiles and executes the test correctly.
- When writing code, you must provide complete implementations with NO placeholders, ellipses (...) or todos. Every function must be fully implemented.
- You only provide code in C. Not in Python. Not in C++. Not in any other language.
</main_directive>

<parser_requirements>
The parser must implement the following requirements:
{requirements}
</parser_requirements>

<perser_specifications>
Also, the parser must follow these specifications:
{specifications}
</perser_specifications>

<input_handling>
CODE INPUT (IMPORTANT AND MANDATORY):
The final C code you generate must read the entire input from standard input (stdin) as raw bytes.
Use a binary-safe approach such as reading in chunks with fread() into a dynamically resized buffer.
Do not use scanf, fgets, or any text-only input functions for the primary input. The parser's input must always come from stdin as bytes.
This is really important because the final C code you generated will be tested giving raw bytes as stdin.
</input_handling>

<output_handling>
CODE OUTPUT (IMPORTANT AND MANDATORY):
Always follow this output rule:
- If parsing succeeds, print a normalized summary of the parsed structure to stdout — NEVER raw input bytes.
- If parsing fails, do NOT print anything to stdout; instead, write a descriptive error message to stderr.
- The summary should be concise and consistent in format, independent of the file type has been parsed.
</output_handling>

{feedback}

Generate a complete C parser implementation following all the instructions above.
"""

def get_feedback_template() -> str:
    return """<feedback>
You received the following feedback from the validator (IMPORTANT):
{validator_assessment}

Correct the following code you have generated, addressing all the issues above while ensuring your implementation remains complete with no placeholders:
```c
{code}
```
</feedback>"""
