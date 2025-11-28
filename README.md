# Mirko nothes

## Virtual environment

- **Python**: 3.13.9
- **Packages**: `requirements.txt`
- **Variables**: `.env`
    ```
    FOLDER_PATH="your-absolute-path"
    GOOGLE_API_KEY="your-key"
    OPENAI_API_KEY="your-key"
    ANTHROPIC_API_KEY="your-key"
    ```

## Improvements

- general code refactoring
    - agents prompts isolated from the logic
    - redundant message passing bugfix on multiagent
- real time entire conversation history passed at each round (loop in the while true) on multiagent
    - messages passed is used for it (not read from log)
    - supervisor memory avoided but last validation assessment kept in agent state (not read from log)
    - supervisor memory was used to track the last generated parser but it could not be the best
        - maybe a middle one compiles and the last one not
        - conversation history should be enough, if not the code could easily be updated to track the last working parser
- Supervisor and Validator doesn't need ReAct approach since they don't have tools to use
- let the user choose directly the followings:
    - file format to generate the parser for (PDF, JSON, HTML, etc...)
        - so test can be easily executed
    - the first part of the interaction: GENERATE_PARSER, CORRECT_ERROR, ASSESS_CODE, GENERAL_CONVERSATION
        - LLM didn't always understand it by itself

## TODO

- correct validator prompt so the assessment will be satisfactory not only if code compiles but also if it executes correctly on test
- test execution can be with exit status 0 (ok) but with program that captures and writes exceptions on stderr

## Prompts

- *difficult*: 
    - generate a parser function for geojson files
    - generate a parser function for geojson files that supports all geometry types
    - generate a parser function for MIME files
        - TODO
- *simple*: 
    - generate a simple parser function for json files
    - generate a simple parser function for xml files

## Ideas

- avoid generating different file format parser in the same conversation
- create specific test case linked to each input file

## Unit Testing

1. Manual file to parse and check some predefined test cases on it
    - I've tested the function but what else? Ask for what I can test or print to stdout and stderr
    - It's proper to ask the user at the beggining what type of file he needs to parse
        - Preconfigured format: xml, json, pdf, geojson, html, http, ...
        - Each format has a different testing inputs and strategy/checks
        - PRO: real testing, CONS: manual test cases (checks definition)
    - Initial simple strategy:
        - if execution OK: print a parsed input summary on stdout
        - if execution FAILS: print errors on stderr
        - PRO: a lot of files (external repository?) can be used as input (no checks definition), CONS: minimal testing
2. Generate automatically through an agent file to parse and test cases on it
    - difficult
3. Metrics:
    - accuracy: files parsed with test case ok / total files parsed
    - coverage: lines executed / total lines
    - execution time

## Questions

- "Agent stopped due to iteration limit or time limit" problem with multiagent
- Problem: last iteration can fail and the middle ones instead create a working code
    - middle: compiles, not satisfactory
    - last: doesn't compile
- **Supervisor** and **Validator** prompts misses Action, Action Input, Observation for ReAct pattern:
    - Invalid Format: Missing 'Action:' after 'Thought:'
    - What is ExceptionTool() needed for in **Supervisor** and **Validator**?

# General considerations

- understand what can be done (or better generated) by the llm and what it's far better to manually control
- real work: give the agent the right way
