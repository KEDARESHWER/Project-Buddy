"""
orchestrator.py
───────────────
The central brain of Project Jarvis.

Responsibilities
────────────────
1. Accept inbound triggers from any channel (terminal, WhatsApp webhook,
   email poll, voice) wrapped in a JarvisMessage.
2. Query ChromaDB for the user's current context and availability.
3. Classify the task intent using a lightweight router LLM call.
4. Delegate to the appropriate sub-agent via AutoGen's conversation API.
5. Persist the outcome back to memory.

Agent topology (AutoGen GroupChat)
────────────────────────────────────

         ┌─────────────────────────────────┐
         │         Orchestrator            │  ← This file
         │  (GroupChatManager / Router)    │
         └────────┬───────┬───────┬────────┘
                  │       │       │
         ┌────────▼──┐ ┌──▼───┐ ┌▼──────────────┐
         │  Comms    │ │Coder │ │ Memory Manager │
         │  Agent    │ │Agent │ │    Agent       │
         └───────────┘ └──────┘ └────────────────┘

AutoGen was chosen over vanilla LangChain because:
  • Native agent-to-agent message passing with termination conditions.
  • Built-in code execution + reply loop (ideal for the Coder Agent).
  • First-class async support via AsyncGroupChat.
  • Human-in-the-loop hooks with is_termination_msg callbacks.

Error handling strategy
────────────────────────
  • Every agent call is wrapped in a tenacity retry decorator to handle
    transient LLM API timeouts without crashing the main loop.
  • All exceptions are caught, logged, and surfaced as a graceful
    error response rather than an unhandled traceback.
  • The Orchestrator never raises — it always returns a JarvisMessage
    with status=FAILED and a populated .error field on unexpected input.
"""
from __future__ import annotations

import asyncio
import json
import webbrowser  # <--- Add this line here
from typing import Any

import asyncio
import json
from typing import Any

import autogen
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings
from memory.memory import (
    get_availability,
    get_context_for_query,
    log_interaction,
    set_availability,
)
from utils.logger import configure_logging, get_logger
from utils.schemas import (
    AgentName,
    InputChannel,
    JarvisMessage,
    TaskStatus,
    UserAvailability,
)

# Bootstrap logging as early as possible so every line below is captured.
configure_logging()
log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────
# LLM configuration shared by all AutoGen agents
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# LLM configuration shared by all AutoGen agents
# ─────────────────────────────────────────────────────────────

_LLM_CONFIG: dict[str, Any] = {
    "config_list": [
        {
            # Pointing directly to your local LM Studio instance
            "model": "local-model",
            "api_key": "lm-studio",
            "base_url": "http://localhost:1234/v1",
        }
    ],
    "temperature": settings.jarvis_llm_temperature,
    "max_tokens": settings.jarvis_llm_max_tokens,
    # Retry settings inside AutoGen's own HTTP layer
    "timeout": 60,
    "max_retries": 3,
    "cache_seed": None,   # Disable caching so live context always reaches the LLM
}

# ─────────────────────────────────────────────────────────────
# Intent classification keywords
# The router first checks simple keyword heuristics before spending
# a full LLM call — fast, free, and good enough for obvious cases.
# ─────────────────────────────────────────────────────────────

_CODER_KEYWORDS = {
    "build", "code", "script", "prototype", "implement", "write a function",
    "create an app", "automate", "debug", "fix the bug", "program",
}
_COMMS_KEYWORDS = {
    "email", "message", "whatsapp", "reply", "send", "decline", "respond",
    "tell them", "ping", "draft", "notify",
}
_MEMORY_KEYWORDS = {
    "remember", "forget", "preference", "update my status", "availability",
    "i'm busy", "set my status", "focus mode", "do not disturb",
}


def _classify_intent_heuristic(text: str) -> AgentName | None:
    """
    Fast keyword-based intent classification.

    Returns the target AgentName if confident, or None to fall through
    to the slower LLM classifier.
    """
    lower = text.lower()
    if any(kw in lower for kw in _CODER_KEYWORDS):
        return AgentName.CODER
    if any(kw in lower for kw in _COMMS_KEYWORDS):
        return AgentName.COMMUNICATION
    if any(kw in lower for kw in _MEMORY_KEYWORDS):
        return AgentName.MEMORY_MANAGER
    return None


# ─────────────────────────────────────────────────────────────
# AutoGen agent definitions
# ─────────────────────────────────────────────────────────────

def _build_orchestrator_agent() -> autogen.AssistantAgent:
    """
    The Orchestrator agent.  Its system prompt positions it as a
    dispatcher that must produce a structured JSON routing decision.
    """
    system_prompt = """\
You are the Orchestrator of Project Jarvis, a highly autonomous personal assistant.
Your ONLY job is to read the user's request plus memory context and output a JSON
routing decision — nothing else.

Output format (strict JSON, no markdown fences):
{
  "target_agent": "<communication_agent | coder_agent | memory_manager>",
  "refined_task": "<concise restatement of the task for the target agent>",
  "reasoning": "<one sentence explanation>"
}

Rules:
- communication_agent: any incoming message to reply to, email task, WhatsApp task.
- coder_agent: any request to build, code, script, prototype, or debug.
- memory_manager: any request to update preferences, availability, or reminders.
- When uncertain, default to communication_agent.
"""
    return autogen.AssistantAgent(
        name="Orchestrator",
        system_message=system_prompt,
        llm_config=_LLM_CONFIG,
        max_consecutive_auto_reply=1,   # Router replies once, then hands off
    )


def _build_communication_agent() -> autogen.AssistantAgent:
    system_prompt = """\
You are the Communication Agent for Project Jarvis.
You handle messaging and local system navigation tasks.

If the user asks to open a website (like Google, GitHub, or YouTube) or search for something, you have a tool called 'open_browser_to_url' available. Call this tool with the correct URL.

Always end your response with: TASK_COMPLETE
"""
    return autogen.AssistantAgent(
        name="CommunicationAgent",
        system_message=system_prompt,
        llm_config=_LLM_CONFIG,
        max_consecutive_auto_reply=3,
    )


def _build_coder_agent() -> autogen.AssistantAgent:
    """
    Coder Agent. Writes code and relies on the UserProxy to execute it.
    """
    system_prompt = """\
You are the Coder Agent for Project Jarvis.
You write executable Python code to solve the user's request.

Process:
1. Write the complete code in a single ```python code block.
2. The user will automatically execute your code and reply with the terminal output.
3. If the terminal output shows an error, analyze it, rewrite the code, and provide the fixed 
```python block.
4. Once the code runs successfully and achieves the goal, you MUST reply with EXACTLY the text: TASK_COMPLETE
"""
    return autogen.AssistantAgent(
        name="CoderAgent",
        system_message=system_prompt,
        llm_config=_LLM_CONFIG,
    )


def _build_memory_agent() -> autogen.AssistantAgent:
    """
    Memory Manager Agent.  Updates ChromaDB with preferences, tasks, and
    availability.  In Step 1 this agent is a stub — full implementation
    is in agents/memory_agent.py.
    """
    system_prompt = """\
You are the Memory Manager Agent for Project Jarvis.
You extract and store meaningful information from conversations.

When updating availability, map the user's words to one of:
  available | focused | in_meeting | away | sleeping

Always end your response with: TASK_COMPLETE
"""
    return autogen.AssistantAgent(
        name="MemoryManagerAgent",
        system_message=system_prompt,
        llm_config=_LLM_CONFIG,
        max_consecutive_auto_reply=2,
    )

def _build_user_proxy() -> autogen.UserProxyAgent:
    """
    The UserProxyAgent represents the human in AutoGen's conversation model.
    It has been granted permission to execute local Python code.
    """
    return autogen.UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        # Stop the conversation when any agent outputs TASK_COMPLETE
        is_termination_msg=lambda msg: "TASK_COMPLETE" in str(msg.get("content")).upper() and "```python" not in str(msg.get("content")).lower(),
        max_consecutive_auto_reply=10,
        code_execution_config={
            "work_dir": "workspace",  # Saves generated scripts here
            "use_docker": False,      # Executes directly on your Windows machine
        },
    )


# ─────────────────────────────────────────────────────────────
# Orchestrator class
# ─────────────────────────────────────────────────────────────

class Orchestrator:
    """
    The central routing brain for Project Jarvis.

    Usage:
        orch = Orchestrator()
        response_msg = await orch.handle(jarvis_message)
    """

    def __init__(self) -> None:
        log.info("Initialising Orchestrator")

        # Build all agents once and reuse across requests
        self._orchestrator_agent = _build_orchestrator_agent()
        self._comm_agent = _build_communication_agent()
        self._coder_agent = _build_coder_agent()
        self._memory_agent = _build_memory_agent()
        self._user_proxy = _build_user_proxy()

        # Map agent names → agent objects for dynamic routing
        self._agent_map: dict[AgentName, autogen.AssistantAgent] = {
            AgentName.COMMUNICATION: self._comm_agent,
            AgentName.CODER: self._coder_agent,
            AgentName.MEMORY_MANAGER: self._memory_agent,
        }

        # ─────────────────────────────────────────────────────────────
        # Register Physical System Tools
        # ─────────────────────────────────────────────────────────────
        
        def open_browser_to_url(url: str) -> str:
            """
            Opens the system's default web browser to a specified website URL.
            Use this whenever the user wants to open a website, search something, or look at a page.
            """
            # Clean up input if the model provides a raw domain
            if not url.startswith("http://") and not url.startswith("https://"):
                url = "https://" + url
            webbrowser.open(url)
            return f"System Action: Successfully opened browser to {url}"

        # Register the function so CommAgent can call it and UserProxy can execute it
        autogen.agentchat.register_function(
            open_browser_to_url,
            caller=self._comm_agent,
            executor=self._user_proxy,
            name="open_browser_to_url",
            description="Opens a web browser on the user's computer to a given URL.",
        )

        log.info("Orchestrator ready", agents=list(self._agent_map.keys()))
    # ── Public API ──────────────────────────────────────────

    async def handle(self, message: JarvisMessage) -> JarvisMessage:
        """
        Process an inbound JarvisMessage end-to-end.

        Steps:
          1. Retrieve relevant memory context.
          2. Classify intent (heuristic → LLM fallback).
          3. Delegate to the target sub-agent via AutoGen.
          4. Store the interaction in memory.
          5. Return a populated response JarvisMessage.

        This method never raises — all exceptions are caught and returned
        as a FAILED status message so the caller's main loop stays alive.
        """
        log.info(
            "Handling message",
            msg_id=str(message.id),
            channel=message.channel,
            preview=message.user_input[:80],
        )

        try:
            # ── Step 1: Enrich with memory context ──────────
            context = get_context_for_query(message.user_input)
            message.context_summary = context
            log.debug("Context retrieved", chars=len(context))

            # ── Step 2: Classify intent ──────────────────────
            target_agent_name = await self._classify_intent(message)
            log.info("Intent classified", target=target_agent_name.value)

            # ── Step 3: Delegate to sub-agent ───────────────
            result = await self._delegate(message, target_agent_name)

            # ── Step 4: Log to memory ────────────────────────
            log_interaction(
                summary=f"Task handled by {target_agent_name.value}: {message.user_input[:120]}",
                agent=AgentName.ORCHESTRATOR.value,
                channel=message.channel,
                metadata={"target_agent": target_agent_name.value},
            )

            return result

        except Exception as exc:
            log.error(
                "Orchestrator caught unhandled exception",
                error=str(exc),
                exc_info=True,
            )
            return JarvisMessage(
                source_agent=AgentName.ORCHESTRATOR,
                channel=message.channel,
                user_input=message.user_input,
                status=TaskStatus.FAILED,
                error=f"Orchestrator error: {exc}",
            )

    # ── Private helpers ─────────────────────────────────────

    async def _classify_intent(self, message: JarvisMessage) -> AgentName:
        """
        Determine which sub-agent should handle this message.

        First tries cheap keyword matching; falls back to a single
        Orchestrator LLM call for ambiguous cases.
        """
        # Fast path — keyword heuristics
        heuristic_result = _classify_intent_heuristic(message.user_input)
        if heuristic_result:
            log.debug("Heuristic routing succeeded", target=heuristic_result.value)
            return heuristic_result

        # Slow path — ask the Orchestrator LLM to decide
        return await self._llm_classify(message)

    @retry(
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _llm_classify(self, message: JarvisMessage) -> AgentName:
        """
        Use the Orchestrator LLM to classify ambiguous intents.
        Wrapped in tenacity retry for transient API failures.
        """
        prompt = (
            f"User input: {message.user_input}\n\n"
            f"Memory context:\n{message.context_summary or 'No context available.'}"
        )

        log.debug("Falling back to LLM classification")

        # Run the synchronous AutoGen initiate_chat in a thread pool so
        # we don't block the async event loop.
        loop = asyncio.get_event_loop()
        chat_result = await loop.run_in_executor(
            None,
            lambda: self._user_proxy.initiate_chat(
                self._orchestrator_agent,
                message=prompt,
                max_turns=1,
            ),
        )

        # Extract the last assistant message and parse JSON
        last_msg = chat_result.chat_history[-1].get("content", "{}")
        routing = self._parse_routing_json(last_msg)
        return routing

    def _parse_routing_json(self, raw: str) -> AgentName:
        """
        Parse the Orchestrator's JSON routing decision.

        Falls back to COMMUNICATION agent if parsing fails — it's the
        safest default (drafting a message is less destructive than
        running arbitrary code).
        """
        try:
            data = json.loads(raw)
            agent_str = data.get("target_agent", "")
            return AgentName(agent_str)
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning(
                "Failed to parse routing JSON — defaulting to communication_agent",
                raw=raw[:200],
                error=str(exc),
            )
            return AgentName.COMMUNICATION

    @retry(
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _delegate(
        self,
        message: JarvisMessage,
        target_agent_name: AgentName,
    ) -> JarvisMessage:
        """
        Hand off the enriched JarvisMessage to the appropriate sub-agent
        and wait for the result.

        Wrapped in tenacity retry for transient LLM API timeouts.
        """
        target_agent = self._agent_map.get(target_agent_name)
        if target_agent is None:
            raise ValueError(f"No agent registered for: {target_agent_name}")

        # Build the full prompt including memory context
        full_prompt = (
            f"{message.user_input}\n\n"
            f"=== Available Context ===\n"
            f"{message.context_summary or 'No prior context.'}\n"
            f"=== User Availability ===\n"
            f"{self._format_availability()}"
        )

        log.info(
            "Delegating to sub-agent",
            target=target_agent_name.value,
            prompt_chars=len(full_prompt),
        )

        loop = asyncio.get_event_loop()
        chat_result = await loop.run_in_executor(
            None,
            lambda: self._user_proxy.initiate_chat(
                target_agent,
                message=full_prompt,
                max_turns=10,
            ),
        )

        # Extract the final agent response
        final_content = ""
        for chat_msg in reversed(chat_result.chat_history):
            if chat_msg.get("role") == "assistant":
                final_content = chat_msg.get("content", "")
                break

        return JarvisMessage(
            source_agent=target_agent_name,
            target_agent=AgentName.ORCHESTRATOR,
            channel=message.channel,
            user_input=message.user_input,
            context_summary=message.context_summary,
            status=TaskStatus.COMPLETED,
            payload={"response": final_content},
        )

    def _format_availability(self) -> str:
        """Fetch and format current availability for injection into prompts."""
        avail = get_availability()
        if avail is None:
            return "No availability status set."
        return (
            f"Status: {avail.status.value} | "
            f"Reason: {avail.reason or 'n/a'} | "
            f"Since: {avail.updated_at.strftime('%H:%M UTC')}"
        )


# ─────────────────────────────────────────────────────────────
# Convenience function: process a raw text input from any channel
# ─────────────────────────────────────────────────────────────

async def process_input(
    text: str,
    channel: InputChannel = InputChannel.TERMINAL,
) -> JarvisMessage:
    """
    Thin wrapper used by CLI, webhook server, and tests.

    Args:
        text:    Raw user input string.
        channel: Where the input came from.

    Returns:
        A completed JarvisMessage with the agent's response in .payload.
    """
    message = JarvisMessage(
        source_agent=AgentName.ORCHESTRATOR,
        channel=channel,
        user_input=text,
        status=TaskStatus.PENDING,
    )
    orch = Orchestrator()
    return await orch.handle(message)


# ─────────────────────────────────────────────────────────────
# CLI entry point  (python -m jarvis.orchestrator)
# ─────────────────────────────────────────────────────────────

async def _cli_loop() -> None:
    """
    Simple interactive REPL for testing Jarvis from the terminal.
    Type 'quit' or Ctrl-C to exit.
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print(Panel("[bold green]Project Jarvis — Orchestrator CLI[/bold green]"))
    console.print("Type your request. Commands: [bold]status[/bold] | [bold]quit[/bold]\n")

    # Seed availability on first run so the system has context
    set_availability(UserAvailability.AVAILABLE, reason="Just started Jarvis")

    orch = Orchestrator()

    while True:
        try:
            raw = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Goodbye.[/yellow]")
            break

        if not raw:
            continue

        if raw.lower() == "quit":
            console.print("[yellow]Shutting down Jarvis.[/yellow]")
            break

        if raw.lower() == "status":
            avail = get_availability()
            if avail:
                console.print(f"[blue]Status:[/blue] {avail.status.value} — {avail.reason}")
            else:
                console.print("[blue]Status:[/blue] Unknown")
            continue

        # Process through the full Orchestrator pipeline
        msg = JarvisMessage(
            source_agent=AgentName.ORCHESTRATOR,
            channel=InputChannel.TERMINAL,
            user_input=raw,
        )

        result = await orch.handle(msg)

        if result.status == TaskStatus.FAILED:
            console.print(f"[red]Error:[/red] {result.error}")
        else:
            response = result.payload.get("response", "(no response)")
            console.print(Panel(response, title=f"[green]{result.source_agent}[/green]"))


if __name__ == "__main__":
    asyncio.run(_cli_loop())
