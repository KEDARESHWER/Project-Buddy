"""
tests/test_orchestrator.py
───────────────────────────
Unit tests for the Orchestrator's intent classification and routing.

The LLM calls are mocked via pytest-mock so these tests run instantly
and without any API key.

Run with:
    pytest tests/test_orchestrator.py -v
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from utils.schemas import (
    AgentName,
    InputChannel,
    JarvisMessage,
    TaskStatus,
    UserAvailability,
)


@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    """Inject dummy env vars so settings validation passes."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))


# ─────────────────────────────────────────────────────────────
# Heuristic classification (fast path — no LLM)
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("build me a web scraper in Python",      AgentName.CODER),
    ("write a script to rename these files",  AgentName.CODER),
    ("prototype a REST API",                  AgentName.CODER),
    ("reply to that WhatsApp message",        AgentName.COMMUNICATION),
    ("send an email to John",                 AgentName.COMMUNICATION),
    ("decline the lunch invite politely",     AgentName.COMMUNICATION),
    ("I'm in focus mode for 2 hours",         AgentName.MEMORY_MANAGER),
    ("set my status to busy",                 AgentName.MEMORY_MANAGER),
    ("remember that I prefer dark mode",      AgentName.MEMORY_MANAGER),
])
def test_heuristic_classification(text, expected):
    """Keyword heuristics should classify common inputs without an LLM call."""
    from orchestrator import _classify_intent_heuristic
    result = _classify_intent_heuristic(text)
    assert result == expected


def test_heuristic_returns_none_for_ambiguous_input():
    """Ambiguous text should return None (triggers LLM fallback)."""
    from orchestrator import _classify_intent_heuristic
    result = _classify_intent_heuristic("what time is it?")
    assert result is None


# ─────────────────────────────────────────────────────────────
# JSON routing parser
# ─────────────────────────────────────────────────────────────

def test_parse_routing_json_valid():
    """Valid JSON routing decision should be parsed correctly."""
    from orchestrator import Orchestrator

    orch = _make_orchestrator_no_init()
    result = orch._parse_routing_json(
        '{"target_agent": "coder_agent", "refined_task": "build scraper", "reasoning": "code task"}'
    )
    assert result == AgentName.CODER


def test_parse_routing_json_invalid_falls_back_to_communication():
    """Malformed JSON should fall back to communication_agent."""
    from orchestrator import Orchestrator

    orch = _make_orchestrator_no_init()
    result = orch._parse_routing_json("not valid json at all")
    assert result == AgentName.COMMUNICATION


def test_parse_routing_json_unknown_agent_falls_back():
    """Unknown agent name in JSON should fall back to communication_agent."""
    from orchestrator import Orchestrator

    orch = _make_orchestrator_no_init()
    result = orch._parse_routing_json('{"target_agent": "ghost_agent"}')
    assert result == AgentName.COMMUNICATION


# ─────────────────────────────────────────────────────────────
# Message construction
# ─────────────────────────────────────────────────────────────

def test_jarvis_message_defaults():
    """JarvisMessage should have sensible defaults."""
    msg = JarvisMessage(
        source_agent=AgentName.ORCHESTRATOR,
        user_input="hello",
    )
    assert msg.status == TaskStatus.PENDING
    assert msg.channel == InputChannel.TERMINAL
    assert msg.error is None
    assert msg.id is not None


def test_jarvis_message_unique_ids():
    """Each JarvisMessage should have a unique UUID."""
    m1 = JarvisMessage(source_agent=AgentName.ORCHESTRATOR, user_input="a")
    m2 = JarvisMessage(source_agent=AgentName.ORCHESTRATOR, user_input="b")
    assert m1.id != m2.id


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _make_orchestrator_no_init():
    """
    Create an Orchestrator instance without calling __init__
    (which would try to connect to ChromaDB and build AutoGen agents).
    Only used for testing pure parsing / routing helpers.
    """
    from orchestrator import Orchestrator
    orch = object.__new__(Orchestrator)
    return orch
