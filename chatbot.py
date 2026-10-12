"""
Minimal CLI chatbot: LangChain + a local Ollama model, with multi-turn memory.

Flow: load config from .env -> build a ChatOllama model -> wrap it in a
prompt template ("chain") -> wrap the chain in RunnableWithMessageHistory so
prior turns are replayed into the prompt on every call -> run a REPL loop.

Tracing: LangSmith tracing is turned on purely by the LANGCHAIN_* env vars
below; no LangSmith code is called directly anywhere in this file.
See README.md for a full explanation of every concept used here.
"""

import os

from dotenv import load_dotenv  # loads variables from .env into os.environ
from langchain_core.chat_history import InMemoryChatMessageHistory  # in-RAM store for one conversation's messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # builds the reusable prompt structure
from langchain_core.runnables.history import RunnableWithMessageHistory  # auto-reads/writes history around a chain
from langchain_ollama import ChatOllama  # chat-model wrapper that talks to the local Ollama server

load_dotenv()

# Single source of truth for which Ollama model tag to use. Change it in
# .env, not here, if the tag ever needs correcting.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")

# The LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY / LANGCHAIN_PROJECT vars from
# .env are read automatically by LangChain's own tracing callback once
# load_dotenv() has populated os.environ -- no tracing code needed here.

# Talks to the local Ollama server (default http://localhost:11434).
# No API key needed: the model is running on this machine, not a cloud API.
llm = ChatOllama(model=OLLAMA_MODEL)

# A ChatPromptTemplate is itself a runnable: it turns a dict of variables
# into a list of chat messages. MessagesPlaceholder is where prior turns
# from history get spliced in before the new human message.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful, concise assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# The "|" pipe is LangChain Expression Language (LCEL): it builds a
# RunnableSequence that feeds prompt's output (a list of messages) into
# llm, which returns an AIMessage. This is a "chain".
chain = prompt | llm

# In-memory store of conversation histories, keyed by session_id. A real
# app could serve many independent conversations this way; this script
# only ever uses one, but the API shape requires a session_id regardless.
store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# Wraps the chain so every invoke() automatically loads prior messages into
# "history" and appends the new turn afterward. This history lives only in
# RAM -- it's gone when the script exits. That's a deliberate simplification
# for this learning project, not a bug (see README.md).
chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

if __name__ == "__main__":
    SESSION_ID = "local-cli-session"
    print("Chatbot ready. Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        response = chatbot.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": SESSION_ID}},
        )
        # ChatOllama returns an AIMessage; .content is just the text reply.
        print(f"Bot: {response.content}")
