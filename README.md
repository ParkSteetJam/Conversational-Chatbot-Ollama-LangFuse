# Conversational Chatbot — LangChain + Ollama + LangSmith

A minimal command-line chatbot that talks to a local Ollama model through
LangChain, remembers earlier turns in the conversation, and has every call
traced in LangSmith. This project exists to **learn** LangChain and
LangSmith fundamentals, so this README explains every concept used — assume
nothing, look nothing up elsewhere to get started.

## 1. What this project is

`chatbot.py` is a single Python script that runs a REPL (read-eval-print
loop) in your terminal: you type a message, it's sent to a local Gemma
model running under Ollama, and the reply is printed back. The bot
remembers everything said earlier in the same run, and every exchange is
recorded as a trace in LangSmith so you can see exactly what was sent to
the model and when.

## 2. What is LangChain?

LangChain is a framework for building applications around LLM calls. Its
core idea is the **`Runnable`** interface: every building block (a prompt
template, a chat model, a parser, ...) implements the same
`invoke(input) -> output` shape. Because everything is a `Runnable`, pieces
can be composed into pipelines and swapped out — e.g. you could replace
`ChatOllama` with `ChatOpenAI` later without changing the rest of the
pipeline, since both expose the same interface.

## 3. What is a "chain" / runnable?

In `chatbot.py`:

```python
chain = prompt | llm
```

The `|` (pipe) operator is **LangChain Expression Language (LCEL)**. It
takes two runnables and returns a new runnable — a `RunnableSequence` —
that feeds the output of the left side into the input of the right side.
Here, `prompt` turns your input into a list of chat messages, and `llm`
sends that list to the model and returns its reply. This composed object,
`chain`, is itself a runnable: calling `chain.invoke(...)` runs the whole
pipeline in order. `invoke()` is the one method you call to "run" any
runnable, whether it's a single step or an entire pipeline.

## 4. What is "memory" here?

LLMs are **stateless per call** — each request to the model only knows
what's in that single request. There is no server-side conversation state.
"Memory" in a chatbot like this just means: **re-send the prior messages
as part of every new request**, so the model can see what was said before.

This project does that with two pieces from `langchain-core`:

- **`InMemoryChatMessageHistory`** — the simplest possible history store:
  a Python list of messages, held in RAM.
- **`RunnableWithMessageHistory`** — a wrapper around `chain` that, on
  every `invoke()`, automatically loads the right history, splices it into
  the `history` slot of the prompt (via `MessagesPlaceholder`), runs the
  chain, and then appends the new human message + AI reply back into that
  history for next time.

Because the history is just a Python dict in RAM (`store` in `chatbot.py`),
**it's lost the moment the script exits.** That's a deliberate
simplification for a learning project, not a bug — a real application
would persist history to a database. See [§12](#12-what-to-learn-next) for
the modern recommended path beyond this.

## 5. What is Ollama, and what is `gemma4:e4b`?

**Ollama** is a local runtime for running LLMs on your own machine (no
cloud API, no API key for Ollama itself). Models are referred to as
`name:tag` — e.g. `gemma4:e4b`, where `gemma4` is the model family and
`e4b` is the specific variant/size tag. `ChatOllama(model=...)` talks to
Ollama's local HTTP server (`http://localhost:11434` by default) and asks
it to run that exact tag.

This project assumes Ollama is already installed and `gemma4:e4b` is
already pulled. To verify:

```powershell
ollama list
```

You should see `gemma4:e4b` in the output. If you ever need to re-pull it:

```powershell
ollama pull gemma4:e4b
```

The tag is read from the `OLLAMA_MODEL` environment variable (see §7), so
if a different tag is ever needed, change it in `.env` — never in the code.

## 6. What is LangSmith, and what does a trace capture?

**LangSmith** is LangChain's observability/tracing platform. Once tracing
is turned on (via environment variables — see §7), **every `invoke()` call
on a LangChain runnable is automatically recorded** as a "run" in LangSmith,
with no extra code required anywhere in `chatbot.py`.

Each run captures:

- The exact input and output of that step.
- Latency and (where the model reports it) token counts.
- A **nested trace tree**: for `chatbot.invoke(...)`, you'll see a parent
  span for the whole call, with child spans for the prompt-formatting step
  and the model call itself.

This is especially useful for understanding "memory": open the model-call
span on the second or third message of a conversation, and you'll see the
**full list of messages actually sent** — including all the earlier turns
that `RunnableWithMessageHistory` resent. It makes memory concrete instead
of abstract.

## 7. Environment variables reference

Set in `.env` (copy `.env.example` to `.env` and fill in real values — see
§8). `.env` is gitignored; `.env.example` is the committed template.

| Variable | Purpose |
|---|---|
| `OLLAMA_MODEL` | The exact Ollama model tag to use, e.g. `gemma4:e4b`. Read by `chatbot.py` directly. |
| `LANGCHAIN_TRACING_V2` | Set to `true` to turn on LangSmith tracing. Read automatically by LangChain — no code needed. |
| `LANGCHAIN_API_KEY` | Your LangSmith API key, from smith.langchain.com → Settings → API Keys. |
| `LANGCHAIN_PROJECT` | Groups traces under a named project in the LangSmith UI instead of `default`. |
| `LANGCHAIN_ENDPOINT` *(optional)* | Only needed for EU-region or self-hosted LangSmith orgs. Standard US accounts default to `https://api.smith.langchain.com` and can omit this. If you need it, your LangSmith "API Keys" page shows the exact value. |

Note: newer LangChain/LangSmith SDKs also accept `LANGSMITH_*` names
(`LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`) as aliases.
This project sticks to the longer-established `LANGCHAIN_*` names. Don't
set both forms inconsistently (e.g. `true` in one and `false` in the
other) — pick one family and use it everywhere.

## 8. Setup instructions

Run these from inside this project folder, in PowerShell.

```powershell
# 1. Verify the model tag resolves locally (flagged risk — do this first)
ollama pull gemma4:e4b
ollama list

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your real .env from the template and fill in your LangSmith API key
copy .env.example .env
notepad .env
```

If step 1 shows a different tag than `gemma4:e4b`, update `OLLAMA_MODEL` in
`.env` to match — don't edit `chatbot.py`.

## 9. Running it

```powershell
python chatbot.py
```

Example session:

```
Chatbot ready. Type 'exit' or 'quit' to stop.
You: My name is Sameer.
Bot: Nice to meet you, Sameer! How can I help you today?
You: What's my name?
Bot: Your name is Sameer.
You: exit
```

The second answer only works because `RunnableWithMessageHistory` resent
the first exchange as part of the second request — that's "memory" in
action.

## 10. Verifying LangSmith traces

1. Go to [smith.langchain.com](https://smith.langchain.com).
2. Open the `conversational-chatbot-ollama-langsmith` project (matches
   `LANGCHAIN_PROJECT`).
3. You should see one trace per message you sent.
4. Click into a trace (ideally the second message onward) and expand the
   model-call span — confirm you can see the full message list, including
   the earlier turn, being sent to the model.

## 11. Git setup

```powershell
git init
git status   # confirm .env and venv/ are NOT listed (they're gitignored)
git add .gitignore requirements.txt .env.example chatbot.py README.md
git commit -m "Initial conversational chatbot with Ollama + LangChain + LangSmith tracing"
```

## 12. What to learn next

This project deliberately stays at "fundamentals" level. Natural next
steps once these concepts feel solid:

- **LangGraph persistence** — the modern recommended way to persist
  conversation state beyond a toy script (survives restarts, supports
  branching/checkpoints). `RunnableWithMessageHistory` remains supported,
  but LangChain's own docs steer new production code toward LangGraph.
- **Streaming responses** — print tokens as they arrive instead of waiting
  for the full reply.
- **Swapping providers** — since `ChatOllama` implements the same
  `Runnable` interface as every other chat model, try swapping in
  `ChatOpenAI` or another provider and see that the rest of the pipeline
  (`prompt`, `chain`, memory wiring) needs no changes.
