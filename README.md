# Mirko nothes

## Virtual environment

- **Python**: 3.13.9
- **Packages**: `requirements.txt`
- **Variables**: `.env`
    ```
    FOLDER_PATH="your-absolute-path"
    GOOGLE_API_KEY="your-key"
    OPENAI_API_KEY="your-key"
    HUGGING_FACE_API_KEY="your-key"
    ANTHROPIC_API_KEY="your-key"
    ```

## Prompt

- *difficult*: generate a parser function for parsing files in geojson format
- *simple*: generate a simple parser function for parsing files in json format

## Unit Testing

1. Manual file to parse and check some predefined test case on it
2. Generate automatically through an agent file to parse and test case on it to test
3. Metrics:
    - accuracy: files parsed with test case ok / total files parsed
    - coverage: lines executed / total lines
    - execution time

## Questions

### Supervisor

- compilation_check: agents make confusion between them? can i avoid it?
- conversation_context: passed in 2 different modes?
- system_metrics complete_round(): not in GENERATE_PARSER and CORRECT_ERROR?

# Sam nothes (old)

1. `zero_shot agent.py`
    - provato a modificare il prompt di inizializzazione in modo tale che l’agente risponda sempre e solo con funzioni complete, senza placeholder o con parti di codice che devono ancora essere sviluppate.

2. `few_shot agent.py`
    - con i 3 esempi di Marco
    - possiamo ora dare lo stesso identico prompt a entrambi e vedere come varia la risposta.
    - per entrambi, ogni tanto l’agente risponde chiedendo più specifiche sulla funzione da implementare, che non so fornirgli.

3. `multi_agent.py`
    - supervisore: chatta con l'utente e capisce quando rivolgersi al generatore, pensando ad un prompt adatto per creare una funzione parser a partire da una richiesta generica dell’utente, e manda il prompt al generatore.
    - generatore: genera il codice in C e lo manda al validatore.
    - validatore: legge il codice generato e sulla base dei requisiti e della completezza, valuta se è soddisfacente (terminando il loop) o no (ritornando dal generatore).
