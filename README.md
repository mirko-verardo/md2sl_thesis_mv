# Mirko nothes

## Virtual environment

- **Python**: 3.13.9
- **Packages**: `requirements.txt`
- **Variables**: `.env`
    ```
    WSL="your-wsl-if-used"
    GOOGLE_API_KEY="your-key"
    OPENAI_API_KEY="your-key"
    ANTHROPIC_API_KEY="your-key"
    ```
    **NB**: if you don't need WSL because you are already running on Linux OS, don't leave empty that variable but set it equal to "none".

## GCC (not used anymore)

- gcc.exe (Rev8, Built by MSYS2 project) 15.2.0
    - Installed with MSYS2

## WSL

- Ubuntu 24.04.3 LTS
```
wsl --install -d Ubuntu
```

### GCC

- gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
```
wsl -d Ubuntu sudo apt install -y build-essential gcc g++
```

## DONE

- general code refactoring
    - agents prompts isolated from the logic
    - redundant message passing bugfix on multiagent
- supervisor memory avoided but last code assessment kept in agent state (not read from log)
- supervisor and assessor doesn't need ReAct approach since they don't have tools to use
    - so they do a single LLM call
- also generator doesn't need ReAct at all because we are in a multiagent system where each agent does 1 thing
- let the user choose directly the followings:
    - file format to generate the parser for (PDF, JSON, HTML, etc...)
        - so test can be easily executed
    - the first part of the interaction: GENERATE_PARSER, CORRECT_ERROR, ASSESS_CODE, GENERAL_CONVERSATION
        - LLM didn't always understand it by itself
- orchestrator, compiler and tester nodes introduced
- compiler-assisted diagnostics and static analysis (light)
    - vulnerability assessment but only at build-time
- dynamic analysis
    - vulnerability assessment at run-time
    - new profile with sanitizers: created by compiler, launched by tester

### Minors

- output folders divided by file format
- user can generate 1 file format parser for each conversation (avoiding confusion)
- better stderr management with file name replaced (confounding) and line and column number specified
- better context management between different loops in multiagent system (keeping in memory last parser generated) and avoiding passing entire conversation history (confounding)

## TODO

- timeout for llm and testing
- dynamic analysis
    - fuzzing
- choose metrics
    - accuracy: 
        - files parsed without errors / total files parsed
        - parser created without errors / total parser created
    - coverage: lines executed / total lines
    - execution time
    - sage score (paper)

### Minors

- specific test for some formats (JSON)
    - I have to produce specific c code to apply test
    - I have to create specific test case linked to each input file
- static analysis (completed)
    - new agent
    - integration of CodeQL tool
    - vulnerability assessment at build-time with advanced analysis (abstract syntax trees, control-flow graphs, ...)
    - less important for parsing than dynamic analysis
- better log management

## Experimental Benchmarking

- Single ReAct agent vs Multi agent
    - for all file formats
    - metrics:
        - mean parser generation TIME
        - mean parser generation ATTEMPTS (react loop for Single, graph iterations for Multi)
        - compilation rate
        - testing rate (training set)
        - testing rate (test set)
        - cyclomatic complexity
        - code coverage

## Static Analysis Tool

### Paper's

- Bandit: NO, only Python
- SonarQube: NO, C but only commercial
- CodeQL: YES (a lot of languages, heavy?)

### Others

- **GCC**
- Splint: NO, last version 2007
- Cpplint: YES, pipx install? Only cpp?
- Cppcheck: YES
- LLVM Clang: YES
- CodeChecker: ? (aggregator, heavy?)

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

## Questions

- "Agent stopped due to iteration limit or time limit" problem with multiagent
    - solved without ReAct
- Problem: last iteration can fail and the middle ones instead create a working code
    - middle: compiles, not satisfactory
    - last: doesn't compile

## General considerations

- understand what can be done (or better generated) by the llm and what it's far better to manually control
- real work: give the agent the right way
- ReAct paradigm could not be ideal in a multiagent system
    - ReAct fits good in a single agent system, where one agent can do multiple things thanks to ReAct
    - in multiagent systems, the real advantage comes from roles specialization where each agent has its own (unique) job
        - overlapping roles doesn't make sense
    - also, the cool thing is the **comparison**: 1 ReAct agent that does N things vs N agents that do 1 thing each
        - the max iteration can be set the same in both systems
