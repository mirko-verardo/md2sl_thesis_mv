def get_parser_requirements() -> str:
    return """1. Input Handling: The code deals with a pointer to a buffer of bytes or a file descriptor for reading unstructured data.
2. Internal State: The code maintains an internal state in memory to represent the parsing state that is not returned as output.
3. Decision-Making: The code takes decisions based on the input and the internal state.
4. Data Structure Creation: The code builds a data structure representing the accepted input or executes specific actions on it.
5. Outcome: The code returns either a boolean value or a data structure built from the parsed data indicating the outcome of the recognition.
6. Composition: The code behavior is defined as a composition of other parsers. (Note: This requirement is only necessary if one of the previous requirements are not met. If ALL the previous 5 requirements are satisfied, this requirement becomes optional.)"""

def get_assessment(assessment: str | None) -> str:
    if assessment is None:
        return "No assessment available"
    return assessment

def get_satisfaction(assessment: str | None) -> str:
    if assessment is None:
        return "Unknown"
    assessment = assessment.lower()
    is_satisfactory = "satisfactory" in assessment and "not satisfactory" not in assessment
    return "SATISFACTORY" if is_satisfactory else "NOT SATISFACTORY"

def get_satisfaction_instructions(assessment: str) -> str:
    assessment = assessment.lower()
    is_satisfactory = "satisfactory" in assessment and "not satisfactory" not in assessment
    return "Mentions that the code is SATISFACTORY" if is_satisfactory else "IMPORTANTLY, mentions that the code is NOT SATISFACTORY according to the validator and briefly explains why"

def get_compilation_status(assessment: str | None) -> str:
    if assessment is None:
        return "Unknown"
    assessment = assessment.lower()
    is_compiled = "successfully compiled" in assessment and "not successfully compiled" not in assessment
    return "SUCCESSFULLY COMPILED" if is_compiled else "NOT SUCCESSFULLY COMPILED"