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
    - supervisor memory avoided but last assessment kept in agent state (not read from log)
    - supervisor memory was used to track the last generated parser but it could not be the best
        - maybe a middle one compiles and the last one not
        - conversation history should be enough, if not the code could easily be updated to track the last working parser
- supervisor and validator doesn't need ReAct approach since they don't have tools to use
    - so they do a single LLM call
- let the user choose directly the followings:
    - file format to generate the parser for (PDF, JSON, HTML, etc...)
        - so test can be easily executed
    - the first part of the interaction: GENERATE_PARSER, CORRECT_ERROR, ASSESS_CODE, GENERAL_CONVERSATION
        - LLM didn't always understand it by itself
- **true story**: ReAct paradigm could not be ideal in a multiagent system
    - ReAct fits good in a single agent system, where one agent can do multiple things thanks to ReAct
    - in multiagent systems, the real advantage comes from roles specialization where each agent has its own (unique) job
        - overlapping roles doesn't make sense
    - also, the cool thing is the **comparison**: 1 ReAct agent that does N things vs N agents that do 1 thing each
        - the max iteration can be set the same in both systems

## TODO

- testing can be with exit status 0 (ok) but with program that captures and writes exceptions on stderr
- specific test for some formats (JSON)
    - i have to produce specific c code to apply test
- c code static analysis for vulnerabilities checking (new agent)
    - python bandit integration tool (code python analysis)
    - search for ready c tool
- sage metric paper
- better memory management between different loops in multiagent system (using last generator code)

## Static Analysis Tool

## Paper's

- Bandit: NO, only Python
- SonarQube: NO, C only commercial
- CodeQL: YES? A lot of languages: heavy?

### Others

- Splint: NO, last version 2007
- Cpplint: YES, pipx install? Only cpp?
- Cppcheck: YES
- Clang: YES
- CodeChecker: ? (aggregator, heavy?)

## Prompts

- *difficult*: 
    - generate a parser function for PDF files
    - generate a parser function for GEOJSON files that supports all geometry types
- *simple*: 
    - generate a simple parser function for JSON files
    - generate a simple parser function for CSV files

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
        - if testing OK: print a parsed input summary on stdout
        - if testing FAILS: print errors on stderr
        - PRO: a lot of files (external repository?) can be used as input (no checks definition), CONS: minimal testing
2. Generate automatically through an agent file to parse and test cases on it
    - difficult
3. Metrics:
    - accuracy: files parsed with test case ok / total files parsed
    - coverage: lines executed / total lines
    - execution time
    - sage score (paper)

## Questions

- "Agent stopped due to iteration limit or time limit" problem with multiagent
    - solved without ReAct
- Problem: last iteration can fail and the middle ones instead create a working code
    - middle: compiles, not satisfactory
    - last: doesn't compile

# General considerations

- understand what can be done (or better generated) by the llm and what it's far better to manually control
- real work: give the agent the right way
