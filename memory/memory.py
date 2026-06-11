"""
memory/memory.py
────────────────
The Memory Manager module.  This is the single source of truth for
everything Jarvis remembers between sessions:

  • User availability status ("User is currently in deep work mode")
  • Long-term preferences extracted from conversations
  • Task history and outcomes
  • Summaries of past email/WhatsApp threads

Architecture
────────────
  ┌────────────────────────────────────────────────────────────┐
  │  ChromaDB (local persistent)                               │
  │  ┌──────────────────┐  ┌──────────┐  ┌───────────────────┐│
  │  │  user_context    │  │  tasks   │  │   conversations   ││
  │  │  (preferences,   │  │ (pending/│  │  (email/WA        ││
  │  │   availability)  │  │  done)   │  │   summaries)      ││
  │  └──────────────────┘  └──────────┘  └───────────────────┘│
  └────────────────────────────────────────────────────────────┘

All text is embedded with a local SentenceTransformer model
("all-MiniLM-L6-v2") so no API call is needed for retrieval.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from config.settings import settings
from utils.logger import get_logger
from utils.schemas import AvailabilityPayload, UserAvailability

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────
# Embedding model (runs entirely locally — no API key needed)
# ─────────────────────────────────────────────────────────────
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level singletons so the model and DB client are loaded
# only once per process, not on every function call.
_embed_model: SentenceTransformer | None = None
_chroma_client: chromadb.PersistentClient | None = None


# ─────────────────────────────────────────────────────────────
# Initialisation helpers
# ─────────────────────────────────────────────────────────────

def get_embed_model() -> SentenceTransformer:
    """
    Lazy-load the SentenceTransformer embedding model.
    Downloads the model on first run; cached locally after that.
    """
    global _embed_model
    if _embed_model is None:
        log.info("Loading embedding model", model=_EMBED_MODEL_NAME)
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
        log.info("Embedding model loaded")
    return _embed_model


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Return (or initialise) the ChromaDB persistent client.

    The database is stored on disk at settings.chroma_persist_dir so
    that memory survives process restarts.
    """
    global _chroma_client
    if _chroma_client is None:
        persist_dir = str(settings.chroma_persist_dir)
        log.info("Initialising ChromaDB", persist_dir=persist_dir)

        _chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(
                anonymized_telemetry=False,   # No usage data sent to Chroma
                allow_reset=True,             # Needed for tests
            ),
        )
        log.info("ChromaDB client ready")
    return _chroma_client


def _get_or_create_collection(name: str) -> chromadb.Collection:
    """
    Get a named collection from ChromaDB, creating it if it doesn't
    exist yet.  All collections share the same local embedding function
    so cross-collection similarity searches stay consistent.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},   # Cosine similarity for semantic search
    )
    log.debug("Collection ready", collection=name)
    return collection


# ─────────────────────────────────────────────────────────────
# Low-level embed / upsert / query helpers
# ─────────────────────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    """
    Convert a string into a vector embedding.

    Args:
        text: Any UTF-8 string.

    Returns:
        A list of floats suitable for storing in ChromaDB.
    """
    model = get_embed_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def upsert_document(
    collection_name: str,
    doc_id: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Embed *text* and store (or update) the document in *collection_name*.

    Uses upsert so that re-embedding an existing key simply overwrites it —
    no duplicate entries accumulate over time.

    Args:
        collection_name: Target ChromaDB collection.
        doc_id:          Stable identifier for this record (e.g. "availability:current").
        text:            Human-readable content to embed.
        metadata:        Optional dict of filterable scalar fields.
    """
    try:
        collection = _get_or_create_collection(collection_name)
        embedding = embed_text(text)

        collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}],
        )
        log.debug("Document upserted", collection=collection_name, doc_id=doc_id)

    except Exception as exc:
        # Log but don't crash — a memory write failure should not kill the agent loop.
        log.error(
            "Failed to upsert document",
            collection=collection_name,
            doc_id=doc_id,
            error=str(exc),
        )


def query_similar(
    collection_name: str,
    query_text: str,
    n_results: int = 5,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Perform a semantic similarity search against a collection.

    Args:
        collection_name: Which ChromaDB collection to search.
        query_text:      Natural-language query string.
        n_results:       Maximum number of results to return.
        where:           Optional metadata filter dict (ChromaDB WHERE syntax).

    Returns:
        A list of dicts, each with keys: 'id', 'document', 'metadata', 'distance'.
        Empty list on error or no results.
    """
    try:
        collection = _get_or_create_collection(collection_name)
        query_embedding = embed_text(query_text)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Flatten ChromaDB's nested list structure into a clean list of dicts
        hits: list[dict[str, Any]] = []
        if results and results.get("ids"):
            for i, doc_id in enumerate(results["ids"][0]):
                hits.append({
                    "id": doc_id,
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                })
        return hits

    except Exception as exc:
        log.error(
            "Similarity query failed",
            collection=collection_name,
            query=query_text[:80],
            error=str(exc),
        )
        return []


# ─────────────────────────────────────────────────────────────
# User availability — the most critical piece of context
# ─────────────────────────────────────────────────────────────

# Stable document ID so there is always exactly ONE availability record.
_AVAILABILITY_DOC_ID = "availability:current"


def set_availability(status: UserAvailability, reason: str = "") -> None:
    """
    Persist the user's current availability status in ChromaDB.

    This is called by the Orchestrator whenever the user says something
    like "I'm going into deep focus for 2 hours" or when a calendar
    event starts.  The Communication Agent reads this before deciding
    whether to auto-decline an incoming message.

    Args:
        status: One of the UserAvailability enum values.
        reason: Optional human-readable context (e.g. "working on Jarvis v1").

    Example:
        set_availability(UserAvailability.FOCUSED, reason="deep work on Jarvis v1")
    """
    payload = AvailabilityPayload(
        status=status,
        reason=reason,
        updated_at=datetime.now(timezone.utc),
    )

    # The text we embed is human-readable so it can be retrieved by
    # natural-language queries like "is the user available right now?"
    text = (
        f"User availability status: {status.value}. "
        f"Reason: {reason or 'not specified'}. "
        f"Updated at: {payload.updated_at.isoformat()}."
    )

    metadata = {
        "status": status.value,
        "reason": reason,
        "updated_at": payload.updated_at.isoformat(),
        "type": "availability",
    }

    upsert_document(
        collection_name=settings.chroma_collection_user_context,
        doc_id=_AVAILABILITY_DOC_ID,
        text=text,
        metadata=metadata,
    )
    log.info("Availability updated", status=status.value, reason=reason)


def get_availability() -> AvailabilityPayload | None:
    """
    Retrieve the most recently stored availability status.

    Returns:
        An AvailabilityPayload or None if no status has been set yet.

    Example:
        avail = get_availability()
        if avail and avail.status == UserAvailability.FOCUSED:
            send_decline_message(...)
    """
    try:
        collection = _get_or_create_collection(settings.chroma_collection_user_context)

        results = collection.get(
            ids=[_AVAILABILITY_DOC_ID],
            include=["metadatas", "documents"],
        )

        if not results or not results.get("ids"):
            log.debug("No availability record found in memory")
            return None

        metadata = results["metadatas"][0]
        return AvailabilityPayload(
            status=UserAvailability(metadata["status"]),
            reason=metadata.get("reason", ""),
            updated_at=datetime.fromisoformat(metadata["updated_at"]),
        )

    except Exception as exc:
        log.error("Failed to retrieve availability", error=str(exc))
        return None


# ─────────────────────────────────────────────────────────────
# Conversation / interaction logging
# ─────────────────────────────────────────────────────────────

def log_interaction(
    summary: str,
    agent: str,
    channel: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Embed and store a summary of any meaningful agent interaction so that
    future context lookups can reference past events.

    Args:
        summary:  Human-readable description of what happened.
        agent:    Which agent produced this log entry.
        channel:  Input channel (email, whatsapp, terminal, …).
        metadata: Additional key-value pairs to attach.

    Returns:
        The auto-generated document ID for this entry.
    """
    doc_id = f"interaction:{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()

    full_metadata = {
        "agent": agent,
        "channel": channel,
        "timestamp": now,
        "type": "interaction",
        **(metadata or {}),
    }

    text = f"[{now}] ({agent} via {channel}): {summary}"

    upsert_document(
        collection_name=settings.chroma_collection_conversations,
        doc_id=doc_id,
        text=text,
        metadata=full_metadata,
    )
    log.debug("Interaction logged", doc_id=doc_id, agent=agent)
    return doc_id


def get_context_for_query(query: str, n_results: int = 5) -> str:
    """
    Fetch the most semantically relevant memory entries for a given
    query and return them as a single formatted string.

    The Orchestrator calls this to inject relevant context into the
    system prompt before delegating to a sub-agent.

    Args:
        query:     Natural-language question or task description.
        n_results: Max number of memory entries to include.

    Returns:
        A multi-line string of relevant memory entries, or an empty
        string if nothing relevant is found.
    """
    hits = query_similar(
        collection_name=settings.chroma_collection_user_context,
        query_text=query,
        n_results=n_results,
    )

    # Also search conversation history
    convo_hits = query_similar(
        collection_name=settings.chroma_collection_conversations,
        query_text=query,
        n_results=n_results,
    )

    all_hits = sorted(hits + convo_hits, key=lambda h: h["distance"])

    if not all_hits:
        return ""

    lines = ["=== Relevant Memory Context ==="]
    for hit in all_hits[:n_results]:
        lines.append(f"[relevance: {1 - hit['distance']:.2f}] {hit['document']}")

    return "\n".join(lines)
