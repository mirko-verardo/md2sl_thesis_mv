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
    - supervisor memory was used to track the last generated parser but it could not be the best (maybe a middle one compiles and the last one not)

## Prompt

- *difficult*: 
    - generate a parser function for geojson files
    - generate a parser function for geojson files that supports all geometry types
    - generate a parser function for MIME files
- *simple*: 
    - generate a simple parser function for xml files
    - generate a simple parser function in C for pdf files

## Ideas

- let the user choose directly the followings:
    - file type to generate the parser for (PDF, JSON, HTML, etc...)
    - the first part of the interaction: GENERATE_PARSER, CORRECT_ERROR, ASSESS_CODE, GENERAL_CONVERSATION
    - (both things are let the llm to undestand, which is not ideal)

## Unit Testing

1. Manual file to parse and check some predefined test cases on it
2. Generate automatically through an agent file to parse and test cases on it
3. Metrics:
    - accuracy: files parsed with test case ok / total files parsed
    - coverage: lines executed / total lines
    - execution time

## Questions

- "Agent stopped due to iteration limit or time limit" problem with multiagent
- Problem: last iteration can fail and the middle ones instead create a working code
- **Supervisor** and **Validator** prompts misses these ones for ReAct patter:
    - Action
    - Action Input
    - Observation
- What is ExceptionTool() needed for in **Supervisor** and **Validator**?
- Invalid Format: Missing 'Action:' after 'Thought:'

# Sam nothes (old)

1. `zero_shot agent.py`
    - provato a modificare il prompt di inizializzazione in modo tale che l'agente risponda sempre e solo con funzioni complete, senza placeholder o con parti di codice che devono ancora essere sviluppate.

2. `few_shot agent.py`
    - con i 3 esempi di Marco
    - possiamo ora dare lo stesso identico prompt a entrambi e vedere come varia la risposta.
    - per entrambi, ogni tanto l'agente risponde chiedendo più specifiche sulla funzione da implementare, che non so fornirgli.

3. `multi_agent.py`
    - supervisore: chatta con l'utente e capisce quando rivolgersi al generatore, pensando ad un prompt adatto per creare una funzione parser a partire da una richiesta generica dell'utente, e manda il prompt al generatore.
    - generatore: genera il codice in C e lo manda al validatore.
    - validatore: legge il codice generato e sulla base dei requisiti e della completezza, valuta se è soddisfacente (terminando il loop) o no (ritornando dal generatore).
