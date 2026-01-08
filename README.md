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
- not using ReAct for multiagent
    - supervisor and assessor doesn't need ReAct since they don't have tools to use (so they do a single LLM call)
    - also generator doesn't need ReAct at all because in a multiagent system each agent can do 1 thing
    - "Agent stopped due to iteration limit or time limit" problem solved without ReAct
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
- testing, simple strategy:
    - if testing OK: print a summary of parsed input on stdout
    - if testing FAILS: print errors on stderr

### Minors

- output folders divided by file format
- user can generate 1 file format parser for each conversation (avoiding confusion)
- better stderr management with file name replaced (confounding) and line and column number specified
- better context management between different loops in multiagent system (keeping in memory last parser generated) and avoiding passing entire conversation history (confounding)

## Experimental Benchmarking

- Single ReAct agent vs Multi agent (2)
- for all file formats (6)
- for all LLM (3)
- metrics:
    - mean parser compilation/testing/validation TIME
        - inside 1 round
    - mean parser compilation/testing/validation ITERATIONS
        - ReAct loops for Single, Graph iterations for Multi
        - inside 1 round, from 1 to max_iterations
    - compilation/testing/validation rate
        - record first parser ok
    - testing rate (on test set)
    - cyclomatic complexity
    - code coverage

## Future developments

- more metrics
- specific test for some formats
    - TDD
    - I have to produce specific c code to apply test
    - I have to create specific test case linked to each input file
- static analysis (completed)
    - new agent
    - integration of CodeQL tool
    - vulnerability assessment at build-time with advanced analysis (abstract syntax trees, control-flow graphs, ...)
    - less important for parsing than dynamic analysis
- dynamic analysis
    - fuzzing

## General considerations

- understand what can be done (or better generated) by the llm and what it's far better to manually control
- real work: give the agent the right way
- ReAct paradigm could not be ideal in a multiagent system
    - ReAct fits good in a single agent system, where one agent can do multiple things thanks to ReAct
    - in multiagent systems, the real advantage comes from roles specialization where each agent has its own (unique) job
        - overlapping roles doesn't make sense
    - also, the cool thing is the **comparison**: 1 ReAct agent that does N things vs N agents that do 1 thing each
        - the max iteration can be set the same in both systems
