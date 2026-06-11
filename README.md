# Project Jarvis 🤖

A highly autonomous, multi-agent personal assistant running locally.

## Architecture

```
jarvis/
├── orchestrator.py          ← Central brain & routing (Step 1 ✅)
├── memory/
│   └── memory.py            ← ChromaDB persistence layer (Step 1 ✅)
├── agents/
│   ├── communication_agent.py  ← Gmail + WhatsApp (Step 2 🔜)
│   ├── coder_agent.py          ← Docker sandbox coding (Step 3 🔜)
│   └── memory_agent.py         ← Memory writes (Step 4 🔜)
├── integrations/               ← API clients (Step 2 🔜)
├── docker_sandbox/             ← Container management (Step 3 🔜)
├── config/
│   └── settings.py          ← Pydantic settings from .env
├── utils/
│   ├── logger.py             ← Structured logging
│   └── schemas.py            ← Pydantic message schemas
└── tests/                   ← pytest suite
```

## Step 1 — What's implemented

| Component | Status | Description |
|---|---|---|
| `orchestrator.py` | ✅ | AutoGen multi-agent router with heuristic + LLM classification |
| `memory/memory.py` | ✅ | ChromaDB client, availability CRUD, interaction logging |
| `config/settings.py` | ✅ | Pydantic v2 settings from `.env` |
| `utils/schemas.py` | ✅ | Typed message envelopes |
| `utils/logger.py` | ✅ | Structlog + rotating file handler |
| `tests/` | ✅ | Unit tests (no API key needed) |

## Setup

```bash
# 1. Clone and enter the project
cd jarvis

# 2. Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure secrets
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY at minimum

# 5. Run the CLI (terminal REPL)
python orchestrator.py

# 6. Run tests (no API key needed)
pytest tests/ -v
```

## Key Design Decisions

**AutoGen over raw LangChain** — AutoGen's native agent-to-agent messaging,
built-in termination conditions, and async support make multi-agent delegation
far cleaner than LangChain's older agent executor pattern.

**Two-stage routing** — Keyword heuristics first (zero latency, zero cost),
LLM fallback only for ambiguous inputs.  Prevents unnecessary API calls for
obvious tasks like "build me a script".

**tenacity retries on every LLM call** — Exponential back-off (2s → 30s, 3
attempts) prevents a single Anthropic API timeout from crashing the main loop.

**Local embeddings** — `all-MiniLM-L6-v2` runs entirely on-device, so memory
reads/writes never require an API call or send data to a third party.
