def get_validator_template() -> str:
    """Validator's template with no ReAct format"""

    return """<role>
You are a specialized C programming validator that evaluates parser code against strict requirements.
</role>

<parser_requirements>
Review this C parser code to determine if it meets the following requirements:
{requirements}
</parser_requirements>

{specifications}

<code_to_review>
```c
{code}
```
</code_to_review>

<compilation_result>
{compilation_status}
</compilation_result>

<validation_process>
Validate the code following these steps:
1. Evaluate whether the code compiles correctly. If it doesn't compile, this is a critical issue that must be addressed.
2. Evaluate each general requirement and whether the code satisfies it. Remember that requirement #6 (Composition) is only necessary if and only if one of requirements 1-5 is not satisfied. If all the requirements 1-5 are met, requirement #6 becomes optional.
3. Check if the code implements all the specifications.
4. Verify that the code has COMPLETE IMPLEMENTATIONS with no placeholders, todos or ellipses (...). Every single function must be fully implemented. There should be no comments like "// ... (Implementation details)" OR "// ... (Full implementation below)" or references to previous code developed. No part of the code should be skipped.
5. Ignore the readability, the efficiency and the maintainability of the generated code.
</validation_process>

<task>
Then provide your final verdict, telling if the code is SATISFACTORY or NOT SATISFACTORY.
A code is NOT SATISFACTORY if:
1. It fails to compile
2. It doesn't meet all the required specifications
3. It contains placeholders or incomplete implementations
If the code is NOT SATISFACTORY, provide specific feedback with details on what needs to be improved and briefly explain how.
</task>

Evaluate the code based on the requirements and provide your assessment.
"""

def get_specifications_template() -> str:
    return """<specifications>
Additionally, it must meet these following specifications:
{supervisor_specifications}
</specifications>
"""
