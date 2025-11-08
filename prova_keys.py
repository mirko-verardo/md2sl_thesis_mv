from utils import initialize_llm

who = "google"
#who = "openai"
#who = "anthropic"
#who = "huggingface"

llm = initialize_llm(who)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to Italian. Translate the user sentence.",
    ),
    ("human", "Hey, how are you today? Maybe it seems to work here..."),
]

ai_msg = llm.invoke(messages)
print(ai_msg.content)
