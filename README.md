# Mirko nothes

## Virtual environment

- **Python**: 3.13.9
- **Packages**: `requirements.txt`

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
