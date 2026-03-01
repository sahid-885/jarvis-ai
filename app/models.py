"""
DATA MODELS MODULE
==================

This file defines the Pydantic models used for API requests, responses, and
internal chat storage. FastAPI uses these to validate incoming JSON and to
serialize responses; the chat service uses them when saving/loading sessions.

MODELS:
  ChatRequest   - Body of POST /chat and POST /chat/realtime (message + optional session_id).
  ChatResponse  - Body returned by both chat endpoints (response text + session_id).
  ChatMessage   - One message in a conversation (role + content). Used inside ChatHistory.
  ChatHistory   - Full conversation: session_id + list of ChatMessage. Used when saving to disk.

─────────────────────────────────────────────────────────────────────────────
WHAT ARE PYDANTIC MODELS?
─────────────────────────────────────────────────────────────────────────────
Pydantic is a data-validation library for Python. You define a class that
inherits from BaseModel, declare its fields with type hints, and Pydantic
automatically:
  1. VALIDATES incoming data — wrong types, missing fields, or constraint
     violations (like min_length) raise a clear error instead of silently passing.
  2. CONVERTS types where possible — e.g. if you send the string "42" for an
     int field, Pydantic converts it to 42.
  3. SERIALIZES to JSON — model.model_dump() / model.model_dump_json() give you
     a dict / JSON string.

WHY DOES FASTAPI USE PYDANTIC?
FastAPI is built on top of Pydantic. When you declare a route like:

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest): ...

FastAPI will:
  - Parse the incoming JSON body and validate it against ChatRequest.
  - If validation fails, return a 422 Unprocessable Entity response with details.
  - Serialize the return value against ChatResponse before sending the response.
  - Auto-generate OpenAPI (Swagger) docs from these models.

This means you get input validation, output serialization, and documentation
for free just by defining these small classes.

─────────────────────────────────────────────────────────────────────────────
HOW THESE MODELS MAP TO API ENDPOINTS
─────────────────────────────────────────────────────────────────────────────
  POST /chat           →  body: ChatRequest   →  response: ChatResponse
  POST /chat/realtime  →  body: ChatRequest   →  response: ChatResponse
  POST /tts            →  body: TTSRequest     →  response: audio bytes
  (internal disk I/O)  →  ChatHistory contains a list of ChatMessages
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# =============================================================================
# MESSAGE AND REQUEST/RESPONSE MODELS
# =============================================================================

class ChatMessage(BaseModel):
    """
    A single message in a conversation (user or assistant).

    Stored in chronological order inside a ChatHistory. No timestamp field;
    the position in the list defines chronology.

    This model is used internally when saving/loading chat sessions to disk
    (as part of ChatHistory). It is NOT directly exposed as an API request or
    response body.

    Fields:
        role:    "user" (human) or "assistant" (JARVIS). Mirrors the OpenAI/Groq
                 message format so we can pass messages directly to the LLM.
        content: The message text. Can be any length (no validation here because
                 this model is only used internally, not for user input).
    """
    role: str      # Either "user" (human) or "assistant" (Jarvis).
    content: str   # The message text.


class ChatRequest(BaseModel):
    """
    Request body for POST /chat and POST /chat/realtime.

    This is the model that FastAPI parses from the JSON body of every incoming
    chat request. Pydantic validates the data *before* your endpoint code runs.

    Fields:
        message:    Required (the ... default means "no default — must be provided").
                    min_length=1 prevents empty strings.
                    max_length=32,000 prevents absurdly long inputs that would
                    blow up the LLM's token limit or waste resources.
                    If validation fails, FastAPI returns HTTP 422 with an error
                    message explaining which constraint was violated.

        session_id: Optional. If omitted (None), the server creates a brand-new
                    session and returns the generated ID in the response.
                    If provided, the server loads that session's history from disk
                    and continues the conversation.

        tts:        If True, the server will also generate a text-to-speech audio
                    response alongside the text reply.
    """
    # Field(...) means "required" — the `...` is Pydantic's sentinel for "no default value".
    # min_length and max_length are Pydantic field validators that run automatically.
    message: str = Field(..., min_length=1, max_length=32_000)
    session_id: Optional[str] = None
    tts: bool = False


class ChatResponse(BaseModel):
    """
    Response body for POST /chat and POST /chat/realtime.

    FastAPI serializes this model to JSON before sending it to the client.

    Fields:
        response:   The assistant's reply text.
        session_id: The session this message belongs to. The client should store
                    this and send it back in the next ChatRequest to continue the
                    same conversation thread.
    """
    response: str
    session_id: str


class ChatHistory(BaseModel):
    """
    Internal model for a full conversation: session id plus ordered list of messages.

    Used when saving a session to disk — the chat service calls
    .model_dump_json() to serialize this to a JSON file, and
    ChatHistory(**json.load(f)) to deserialize it back.

    This model is NOT directly used as an API request or response; it's purely
    for internal storage.

    Fields:
        session_id: Unique identifier for this conversation (UUID string).
        messages:   Ordered list of ChatMessage objects (alternating user/assistant).
    """
    session_id: str
    messages: List[ChatMessage]


class TTSRequest(BaseModel):
    """
    Request body for POST /tts. Contains the text to convert to speech.

    Fields:
        text: The text to synthesize. min_length=1 prevents empty requests.
              max_length=5000 prevents excessively long audio generation, which
              would be slow and consume unnecessary resources.
    """
    text: str = Field(..., min_length=1, max_length=5000)
