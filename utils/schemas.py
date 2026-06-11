"""
utils/schemas.py
────────────────
Pydantic v2 models that define the canonical message structures passed
between agents.  Keeping these in one place means:

  • Every agent knows exactly what shape of data it will receive.
  • Serialisation / deserialisation is handled automatically.
  • Runtime validation surfaces data bugs early (before an agent acts on them).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────

class AgentName(str, Enum):
    """Canonical identifiers for every agent in the system."""
    ORCHESTRATOR = "orchestrator"
    COMMUNICATION = "communication_agent"
    CODER = "coder_agent"
    MEMORY_MANAGER = "memory_manager"


class TaskStatus(str, Enum):
    """Life-cycle states a task can be in."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


class InputChannel(str, Enum):
    """Where the original user input came from."""
    VOICE = "voice"
    WHATSAPP = "whatsapp"
    EMAIL = "email"
    TERMINAL = "terminal"   # Direct CLI interaction (dev/test)


class UserAvailability(str, Enum):
    """High-level user availability states stored in memory."""
    AVAILABLE = "available"
    FOCUSED = "focused"         # Deep work — decline non-urgent interruptions
    MEETING = "in_meeting"
    AWAY = "away"
    SLEEPING = "sleeping"


# ─────────────────────────────────────────────────────────────
# Core message envelope
# ─────────────────────────────────────────────────────────────

class JarvisMessage(BaseModel):
    """
    The universal message envelope that flows through Jarvis.
    Every agent speaks this language when handing off work.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique message identifier")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of message creation",
    )
    source_agent: AgentName = Field(description="The agent that created this message")
    target_agent: AgentName | None = Field(
        default=None,
        description="Intended recipient; None means broadcast / Orchestrator decides",
    )
    channel: InputChannel = Field(
        default=InputChannel.TERMINAL,
        description="Original input channel",
    )
    user_input: str = Field(description="Raw text input or summarised trigger event")
    context_summary: str = Field(
        default="",
        description="Relevant memory context retrieved from ChromaDB",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific data (e.g. code, email body, docker output)",
    )
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    error: str | None = Field(
        default=None,
        description="Error message if processing failed; None on success",
    )

    class Config:
        use_enum_values = True


# ─────────────────────────────────────────────────────────────
# Specialised payloads (stored inside JarvisMessage.payload)
# ─────────────────────────────────────────────────────────────

class EmailPayload(BaseModel):
    """Structured data for an incoming email handed to the Orchestrator."""
    message_id: str
    sender: str
    subject: str
    body_snippet: str       # First ~500 chars
    received_at: datetime
    needs_reply: bool = False
    suggested_reply: str = ""


class WhatsAppPayload(BaseModel):
    """Structured data for an incoming WhatsApp message."""
    from_number: str
    to_number: str
    body: str
    media_url: str | None = None
    received_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    suggested_reply: str = ""


class CoderPayload(BaseModel):
    """Structured data produced by the Coder Agent."""
    language: str = "python"
    source_code: str
    docker_image: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    iterations: int = 0       # How many fix-and-retry loops were needed
    success: bool = False


class AvailabilityPayload(BaseModel):
    """Payload for availability updates stored in ChromaDB."""
    status: UserAvailability
    reason: str = ""
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
