"""
tests/test_memory.py
─────────────────────
Unit tests for the Memory Manager module (memory/memory.py).

These tests use a temporary directory for ChromaDB so they never touch
the real persistent database.  No LLM API key is needed — only the
local SentenceTransformer embedding model is used.

Run with:
    pytest tests/test_memory.py -v
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from utils.schemas import UserAvailability


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_chroma(tmp_path: Path, monkeypatch):
    """
    Redirect ChromaDB to a fresh temp directory for every test.
    Also reset the module-level singletons so tests don't share state.
    """
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")

    # Force reimport with new settings
    import importlib
    import config.settings as settings_mod
    import memory.memory as memory_mod

    importlib.reload(settings_mod)
    importlib.reload(memory_mod)

    # Reset singletons
    memory_mod._chroma_client = None
    memory_mod._embed_model = None

    yield

    # Cleanup
    memory_mod._chroma_client = None


# ─────────────────────────────────────────────────────────────
# Tests: ChromaDB client initialisation
# ─────────────────────────────────────────────────────────────

def test_chroma_client_initialises(tmp_path):
    """get_chroma_client() should return a working PersistentClient."""
    from memory.memory import get_chroma_client
    client = get_chroma_client()
    assert client is not None


def test_chroma_client_is_singleton():
    """Calling get_chroma_client() twice should return the same object."""
    from memory.memory import get_chroma_client
    c1 = get_chroma_client()
    c2 = get_chroma_client()
    assert c1 is c2


# ─────────────────────────────────────────────────────────────
# Tests: Availability
# ─────────────────────────────────────────────────────────────

def test_get_availability_returns_none_when_empty():
    """Before any availability is set, get_availability() should return None."""
    from memory.memory import get_availability
    result = get_availability()
    assert result is None


def test_set_and_get_availability_focused():
    """Setting availability to FOCUSED should persist and retrieve correctly."""
    from memory.memory import set_availability, get_availability

    set_availability(UserAvailability.FOCUSED, reason="Deep work on Jarvis")

    result = get_availability()
    assert result is not None
    assert result.status == UserAvailability.FOCUSED
    assert result.reason == "Deep work on Jarvis"


def test_set_and_get_availability_available():
    """Setting availability to AVAILABLE should overwrite FOCUSED."""
    from memory.memory import set_availability, get_availability

    set_availability(UserAvailability.FOCUSED, reason="Working")
    set_availability(UserAvailability.AVAILABLE, reason="Done for now")

    result = get_availability()
    assert result is not None
    assert result.status == UserAvailability.AVAILABLE
    assert result.reason == "Done for now"


def test_availability_has_timestamp():
    """Availability record should carry a UTC timestamp."""
    from memory.memory import set_availability, get_availability

    before = datetime.now(timezone.utc)
    set_availability(UserAvailability.MEETING)
    after = datetime.now(timezone.utc)

    result = get_availability()
    assert result is not None
    assert before <= result.updated_at <= after


def test_set_availability_without_reason():
    """set_availability should work with an empty reason string."""
    from memory.memory import set_availability, get_availability

    set_availability(UserAvailability.AWAY)   # No reason kwarg
    result = get_availability()
    assert result is not None
    assert result.status == UserAvailability.AWAY
    assert result.reason == ""


# ─────────────────────────────────────────────────────────────
# Tests: upsert and query
# ─────────────────────────────────────────────────────────────

def test_upsert_and_query_finds_document():
    """A document upserted into a collection should be retrievable by semantic query."""
    from memory.memory import upsert_document, query_similar

    upsert_document(
        collection_name="test_collection",
        doc_id="test:001",
        text="User prefers dark mode for all interfaces",
        metadata={"type": "preference"},
    )

    results = query_similar(
        collection_name="test_collection",
        query_text="What are the user's UI preferences?",
        n_results=3,
    )

    assert len(results) >= 1
    assert results[0]["id"] == "test:001"
    assert "dark mode" in results[0]["document"]


def test_upsert_overwrites_existing_document():
    """Upserting the same doc_id twice should update, not duplicate."""
    from memory.memory import upsert_document, query_similar, get_chroma_client
    from config.settings import settings

    doc_id = "pref:theme"
    upsert_document("test_coll", doc_id, "User likes light mode")
    upsert_document("test_coll", doc_id, "User likes dark mode")

    client = get_chroma_client()
    coll = client.get_collection("test_coll")
    result = coll.get(ids=[doc_id])

    # Should be exactly one document with the latest content
    assert len(result["ids"]) == 1
    assert "dark mode" in result["documents"][0]


def test_query_returns_empty_list_for_empty_collection():
    """Querying a collection with no data should return an empty list, not raise."""
    from memory.memory import query_similar

    results = query_similar("empty_collection_xyz", "anything", n_results=5)
    assert results == []


# ─────────────────────────────────────────────────────────────
# Tests: interaction logging
# ─────────────────────────────────────────────────────────────

def test_log_interaction_returns_doc_id():
    """log_interaction should return a non-empty string doc_id."""
    from memory.memory import log_interaction

    doc_id = log_interaction(
        summary="User asked to build a web scraper",
        agent="coder_agent",
        channel="terminal",
    )
    assert isinstance(doc_id, str)
    assert len(doc_id) > 0


def test_get_context_for_query_returns_string():
    """get_context_for_query should always return a string (never raise)."""
    from memory.memory import get_context_for_query, log_interaction

    log_interaction("User is working on a Python project", "orchestrator", "terminal")
    context = get_context_for_query("What is the user working on?")

    assert isinstance(context, str)
