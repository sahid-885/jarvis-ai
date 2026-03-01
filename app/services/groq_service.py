"""
GROQ SERVICE MODULE
===================

This module handles general chat: no web search, only the Groq LLM plus context
from the vector store (learning data + past chats). Used by ChatService for
POST /chat.

ARCHITECTURE OVERVIEW:
  GroqService is the core LLM interface for JARVIS. It does three things:
    1. RETRIEVE: Ask the vector store for relevant context chunks.
    2. BUILD: Assemble the full system prompt (personality + time + context + addendum).
    3. CALL: Send the prompt + history + question to Groq and return the response.

  This class is also the parent of RealtimeGroqService, which adds web search.
  The inheritance design means realtime can reuse _build_prompt_and_messages,
  _invoke_llm, and _stream_llm — it only overrides get_response/stream_response
  to inject search results.

MULTIPLE API KEYS (primary-first with fallback):
  - You can set multiple Groq API keys in .env: GROQ_API_KEY, GROQ_API_KEY_2,
    GROQ_API_KEY_3, ... (no limit).
  - PRIMARY-FIRST: Every request tries the first API key first. If it fails
    (rate limit 429, timeout, network error, etc.), we immediately try the
    second key, then the third, until one succeeds.
  - Each key gets 1 retry for transient failures before falling back to the next.
  - If ALL keys fail, we raise AllGroqApisFailedError with a user-friendly message.
  - All API key usage is logged with masked keys for security and debugging.

  WHY PRIMARY-FIRST (not round-robin):
    Round-robin distributes load evenly, but for a single-user app, we want to
    maximize usage of the primary key (best quota) and only touch backup keys
    when the primary is rate-limited. This keeps billing predictable and avoids
    unnecessary key rotation.

  HOW THE FALLBACK LOOP WORKS (visual):
    Request arrives
      ├─ Try Key #1 (primary) ─── success? → return response
      │                           failure? ↓
      ├─ Try Key #2 (backup)  ─── success? → return response
      │                           failure? ↓
      ├─ Try Key #3 (backup)  ─── success? → return response
      │                           failure? ↓
      └─ All keys exhausted → raise AllGroqApisFailedError

FLOW:
  1. get_response(question, chat_history) is called.
  2. We ask the vector store for the top-k chunks most similar to the question (retrieval).
  3. We build a system message: JARVIS_SYSTEM_PROMPT + current time + retrieved context.
  4. We send to Groq using the first key; on failure, try second, third, etc.
  5. We return the assistant's reply, or raise AllGroqApisFailedError if all fail.

Context is only what we retrieve (not a full dump of learning data), so token usage stays bounded.
"""

from typing import List, Optional, Iterator
# ─── LANGCHAIN IMPORTS ───────────────────────────────────────────────────────
# LangChain is a framework that provides abstractions for working with LLMs.
# Instead of making raw HTTP requests to the Groq API, we use LangChain's
# wrappers which handle serialization, retries, and response parsing.
#
# ChatGroq: LangChain's wrapper around the Groq API. It handles HTTP calls,
#   retries, and response parsing so we don't have to use raw requests.
#   Each ChatGroq instance is bound to one API key.
#
# ChatPromptTemplate: Builds the full prompt from a system message, history
#   placeholder, and the human question. LangChain compiles this into the
#   exact message format the Groq API expects (an array of role-tagged messages).
#   Think of it as a "template" with slots for dynamic content.
#
# MessagesPlaceholder: A slot in the template where we inject the chat history
#   (a list of HumanMessage/AIMessage objects). At render time, LangChain
#   expands this into the correct sequence of role-tagged messages. Without
#   this, we'd have to manually format the history into the message array.
#
# HumanMessage / AIMessage: LangChain's typed message objects. We convert our
#   (user_text, assistant_text) tuples into these for the history placeholder.
#   HumanMessage becomes {"role": "user", "content": "..."} in the API call,
#   and AIMessage becomes {"role": "assistant", "content": "..."}.
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

import logging
import time

from config import GROQ_API_KEYS, GROQ_MODEL, JARVIS_SYSTEM_PROMPT, GENERAL_CHAT_ADDENDUM
from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information
from app.utils.retry import with_retry

logger = logging.getLogger("J.A.R.V.I.S")

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
# Request timeout: if the Groq API doesn't respond within 60 seconds, we
# consider the request failed and move on (either retry or fallback to next key).
# This prevents the server from hanging indefinitely on a stuck connection.
# 60 seconds is generous — Groq typically responds in 1-5 seconds. The high
# timeout accounts for rare cases where the API is under heavy load.
GROQ_REQUEST_TIMEOUT = 60

# User-friendly message when all API keys fail. This is shown to the end user,
# so it avoids technical jargon and suggests trying again later.
# It's defined as a constant so both _invoke_llm and _stream_llm use the same text.
ALL_APIS_FAILED_MESSAGE = (
    "I'm unable to process your request at the moment. All API services are "
    "temporarily unavailable. Please try again in a few minutes."
)


class AllGroqApisFailedError(Exception):
    """
    Raised when every configured Groq API key has been tried and all failed.

    This is a custom exception so the API layer can catch it specifically and
    return a 503 Service Unavailable response (rather than a generic 500).
    The error message (ALL_APIS_FAILED_MESSAGE) is user-friendly and safe to
    display directly.

    USAGE IN THE API LAYER:
      try:
          response = groq_service.get_response(question, history)
      except AllGroqApisFailedError as e:
          return JSONResponse(status_code=503, content={"detail": str(e)})
    """
    pass


# =============================================================================
# HELPER: ESCAPE CURLY BRACES FOR LANGCHAIN
# =============================================================================
# LangChain prompt templates use {variable_name} for string interpolation.
# For example, {question} gets replaced with the actual question text.
#
# PROBLEM: If the learning data or chat history contains literal { or } characters
# (very common in code snippets, JSON data, math notation, etc.), LangChain's
# template engine will try to interpret them as variables and throw a KeyError.
#
# EXAMPLE OF THE PROBLEM:
#   Context contains: "def foo() { return 1; }"
#   LangChain sees {, tries to find a variable named " return 1; }", crashes.
#
# SOLUTION: Double every { and } → {{ and }}. In Python's string formatting
# (which LangChain uses internally), {{ renders as a literal { in the output.
# We apply this to ALL user-provided content before injecting it into templates.
#
# This is similar to SQL parameterization — we sanitize user input before
# injecting it into a template to prevent "injection" errors.

def escape_curly_braces(text: str) -> str:
    """
    Double every { and } so LangChain does not treat them as template variables.

    Examples:
      "def foo() { return 1; }"  →  "def foo() {{ return 1; }}"
      "{name}"                    →  "{{name}}"
      "no braces here"            →  "no braces here" (unchanged)

    This is critical for safety: without it, any user message containing
    curly braces could crash the LLM chain with a template formatting error.
    """
    if not text:
        return text
    return text.replace("{", "{{").replace("}", "}}")


def _is_rate_limit_error(exc: BaseException) -> bool:
    """
    Check if an exception indicates a Groq rate limit (HTTP 429 or quota exceeded).

    We use string matching because Groq's SDK may raise different exception types
    for rate limits vs quota limits. Checking the error message is the most
    reliable way to detect all variants.

    COMMON RATE LIMIT INDICATORS IN GROQ ERROR MESSAGES:
      - "429" — HTTP status code for Too Many Requests.
      - "rate limit" — explicit rate limit message.
      - "tokens per day" — daily quota exceeded.

    Used for LOGGING ONLY — the actual fallback logic tries the next key on ANY
    failure, not just rate limits. This means even unexpected errors (network
    timeouts, 500s) trigger a fallback, making the system more resilient.
    """
    msg = str(exc).lower()
    return "429" in str(exc) or "rate limit" in msg or "tokens per day" in msg


def _log_timing(label: str, elapsed: float, extra: str = ""):
    """
    Log timing in consistent format for performance monitoring.

    All timing logs use the [TIMING] prefix so they can be easily filtered
    with grep: `grep "[TIMING]" jarvis.log` shows all performance data.
    """
    msg = f"[TIMING] {label}: {elapsed:.3f}s"
    if extra:
        msg += f" ({extra})"
    logger.info(msg)


def _mask_api_key(key: str) -> str:
    """
    Mask an API key for safe logging: show first 8 and last 4 characters only.

    WHY: API keys must never appear in plain text in logs (security risk). But
    we need to distinguish WHICH key was used for debugging multi-key setups.
    Showing the prefix and suffix is enough to identify the key without exposing it.

    Example: "gsk_abc123xyz789def456" → "gsk_abc1...f456"

    EDGE CASE: Keys shorter than 12 characters are fully masked to "***masked***"
    because there isn't enough entropy to safely show partial content.
    """
    if not key or len(key) <= 12:
        return "***masked***"
    return f"{key[:8]}...{key[-4:]}"


# =============================================================================
# GROQ SERVICE CLASS
# =============================================================================

class GroqService:
    """
    General chat service: retrieves context from the vector store and calls the Groq LLM.

    MULTI-KEY FALLBACK STRATEGY (PRIMARY-FIRST):
      Unlike round-robin (which cycles through keys evenly), primary-first always
      starts with key #1. This is intentional:
        - Key #1 is the "main" key with the best quota/tier.
        - Keys #2, #3, etc. are backups for when #1 is rate-limited.
        - If key #1 works, we never touch the others (saves their quota).
        - If key #1 fails, we try #2 immediately (no delay between keys).
        - Each key gets 1 retry (via with_retry) for transient errors before
          we give up on it and move to the next.
        - If ALL keys fail, we raise AllGroqApisFailedError.

    INHERITANCE:
      RealtimeGroqService extends this class. It inherits:
        - __init__ (creates LLM clients + stores vector store)
        - _invoke_llm / _stream_llm (multi-key fallback)
        - _build_prompt_and_messages (system prompt assembly)
      And overrides:
        - get_response / stream_response (to add web search results)

    WHY LANGCHAIN (not raw HTTP):
      LangChain provides:
        - Automatic message formatting (system/human/assistant roles).
        - The prompt template system (with variable substitution).
        - The "|" pipe operator for chaining (prompt | llm).
        - Streaming support via .stream() method.
      Without LangChain, we'd need ~100 lines of boilerplate for HTTP calls,
      message formatting, and stream parsing.
    """

    def __init__(self, vector_store_service: VectorStoreService):
        """
        Create one ChatGroq LLM client per API key and store the vector store reference.

        HOW IT WORKS:
          - GROQ_API_KEYS is a list loaded from .env (e.g. [key1, key2, key3]).
          - We create a ChatGroq instance for each key. self.llms[0] uses key #1,
            self.llms[1] uses key #2, etc.
          - All clients share the same model, temperature, and timeout settings.
          - The vector_store_service is used later for context retrieval.

        PARAMETERS EXPLAINED:
          - temperature=0.6: Controls randomness in the LLM's output.
            0.0 = fully deterministic (same input always gives same output).
            1.0 = highly random (creative but potentially incoherent).
            0.6 gives JARVIS a personality while staying coherent. This is
            tunable — lower for factual tasks, higher for creative ones.
          - request_timeout=60: Max seconds to wait for Groq's response.
            After this, the request is considered failed and we try the next key.

        WHY ONE CLIENT PER KEY:
          ChatGroq binds the API key at construction time. To try different keys,
          we need different ChatGroq instances. The fallback loop in _invoke_llm
          simply indexes into self.llms[i] to pick the right client.

        Raises:
            ValueError: If no API keys are configured in .env.
        """
        if not GROQ_API_KEYS:
            raise ValueError(
                "No Groq API keys configured. Set GROQ_API_KEY (and optionally GROQ_API_KEY_2, GROQ_API_KEY_3, ...) in .env"
            )
        # Create one LLM client per API key — self.llms[i] uses GROQ_API_KEYS[i].
        # List comprehension creates all clients in one pass.
        self.llms = [
            ChatGroq(
                groq_api_key=key,
                model_name=GROQ_MODEL,
                temperature=0.6,
                request_timeout=GROQ_REQUEST_TIMEOUT,
            )
            for key in GROQ_API_KEYS
        ]
        # Store the vector store reference for retrieval in _build_prompt_and_messages.
        self.vector_store_service = vector_store_service
        logger.info(f"Initialized GroqService with {len(GROQ_API_KEYS)} API key(s) (primary-first fallback)")

    # ─── LLM INVOCATION WITH MULTI-KEY FALLBACK ─────────────────────────────
    # These two methods (_invoke_llm and _stream_llm) implement the core
    # fallback logic. They are used by get_response/stream_response (and their
    # realtime overrides) to actually call the Groq API.
    #
    # The pattern is the same for both:
    #   for each key (starting from #1):
    #       try to call Groq with this key
    #       if success → return result
    #       if failure → log it, move to next key
    #   all keys failed → raise AllGroqApisFailedError
    #
    # BLOCKING vs STREAMING:
    #   _invoke_llm: Waits for the complete response before returning. Simpler,
    #     and supports with_retry (because the whole call is atomic).
    #   _stream_llm: Returns an iterator that yields chunks as they arrive.
    #     Does NOT use with_retry because you can't retry a partially consumed
    #     stream (some tokens were already sent to the client).

    def _invoke_llm(
        self,
        prompt: ChatPromptTemplate,
        messages: list,
        question: str,
    ) -> str:
        """
        Call the LLM (blocking) using PRIMARY-FIRST fallback across all API keys.

        DETAILED FLOW:
          1. Start with i=0 (first/primary API key).
          2. Build a LangChain chain: prompt | self.llms[i]
             This means: render the prompt template, then pass it to the LLM.
          3. Call chain.invoke() wrapped in with_retry(max_retries=2):
             - First attempt: call the API.
             - If it fails with a transient error: wait 0.5s, retry once.
             - If the retry also fails: give up on this key.
          4. If the call succeeds: return response.content (the text).
          5. If the call fails: log the error, increment i, go to step 2.
          6. If all keys exhausted: raise AllGroqApisFailedError.

        WHY max_retries=2:
          "2 attempts total" means 1 initial + 1 retry. This catches brief
          network blips without wasting too much time on a truly dead key.
          The delay (0.5s) is short because we'd rather try the next key quickly.

        THE PIPE OPERATOR (prompt | self.llms[i]):
          LangChain's "|" operator creates a "chain" — a pipeline where the
          output of one step feeds into the next. Here:
            1. `prompt` renders the template with the provided variables.
            2. The rendered messages are passed to `self.llms[i]` (the Groq client).
            3. The Groq client sends them to the API and returns the response.
          This is equivalent to: self.llms[i].invoke(prompt.format_messages(...))

        Args:
            prompt: The compiled LangChain prompt template (system + history + question).
            messages: List of HumanMessage/AIMessage objects for the history placeholder.
            question: The user's current question (fills {question} in the template).

        Returns:
            The LLM's response text (str).

        Raises:
            AllGroqApisFailedError: If every API key fails after retries.
        """
        n = len(self.llms)
        last_exc = None
        # Track which keys we tried (for the diagnostic log if all fail).
        keys_tried = []

        for i in range(n):
            keys_tried.append(i)
            masked_key = _mask_api_key(GROQ_API_KEYS[i])
            logger.info(f"Trying API key #{i + 1}/{n}: {masked_key}")

            def _invoke_with_key():
                # The "|" operator (pipe) is LangChain's way of chaining:
                # prompt template → LLM. It renders the template first, then
                # sends the rendered messages to the LLM.
                chain = prompt | self.llms[i]
                # .invoke() sends the request and blocks until the full response arrives.
                # We pass the template variables: "history" for the chat history
                # and "question" for the current user question.
                return chain.invoke({"history": messages, "question": question})

            try:
                # with_retry wraps the call with automatic retry logic:
                # - max_retries=2: try up to 2 times (1 initial + 1 retry).
                # - initial_delay=0.5: wait 0.5 seconds before the retry.
                # If both attempts fail, the exception propagates to our except block.
                response = with_retry(
                    _invoke_with_key,
                    max_retries=2,  # 2 attempts total (1 initial + 1 retry) per key
                    initial_delay=0.5,
                )
                # If we got here via a fallback key (i > 0), log it for visibility.
                # This helps operators know that the primary key was rate-limited.
                if i > 0:
                    logger.info(f"Fallback successful: API key #{i + 1}/{n} succeeded: {masked_key}")
                # response is a LangChain AIMessage object; .content is the text string.
                return response.content
            except Exception as e:
                last_exc = e
                # Log whether it's a rate limit or a different error (helps debugging).
                if _is_rate_limit_error(e):
                    logger.warning(f"API key #{i + 1}/{n} rate limited: {masked_key}")
                else:
                    logger.warning(f"API key #{i + 1}/{n} failed: {masked_key} - {str(e)[:100]}")
                if i < n - 1:
                    # More keys to try — continue the fallback loop.
                    logger.info(f"Falling back to next API key...")
                    continue
                break  # Last key failed — exit the loop.

        # All keys failed — build a diagnostic log and raise.
        # We log ALL keys that were tried so the operator can investigate.
        masked_all = ", ".join([_mask_api_key(GROQ_API_KEYS[j]) for j in keys_tried])
        logger.error(f"All {n} API key(s) failed. Tried: {masked_all}")
        # Chain the original exception (from last_exc) for full traceback context.
        raise AllGroqApisFailedError(ALL_APIS_FAILED_MESSAGE) from last_exc

    def _stream_llm(
        self,
        prompt: ChatPromptTemplate,
        messages: list,
        question: str,
    ) -> Iterator[str]:
        """
        Stream the LLM response token-by-token using PRIMARY-FIRST fallback.

        HOW STREAMING DIFFERS FROM BLOCKING:
          - chain.stream() returns an iterator of chunk objects instead of one response.
          - Each chunk may have a .content attribute (str) or be a dict with "content".
          - We extract the text from each chunk and yield it to the caller.
          - We only get ONE attempt per key (no with_retry) because you can't
            "retry" a partially consumed stream — the LLM has already generated
            some tokens. If the stream breaks mid-way, we fall back to the next key
            and start fresh (the caller sees a clean stream from the new key).

        WHY NO with_retry FOR STREAMING:
          In blocking mode, a failed call is atomic — nothing was sent to the client.
          In streaming mode, we may have already yielded some chunks to the caller
          (which forwarded them to the client via SSE). If we retried, the client
          would receive the beginning of the response twice. So instead, we fail
          over to the next key and start a completely fresh stream.

        TIMING INSTRUMENTATION:
          - first_chunk: Time from request start to first token received. This is
            the user-perceived latency (Time To First Token / TTFT). For Groq,
            this is typically 200-500ms.
          - groq_stream_total: Total time from start to last token. Includes all
            token generation time. For a medium response, this is typically 2-8s.

        CHUNK FORMAT:
          LangChain's streaming chunks can come in two formats depending on the
          provider and version:
            1. Object with .content attribute (most common): chunk.content = "Hello"
            2. Dict with "content" key: chunk["content"] = "Hello"
          We handle both formats for robustness.

        Args:
            prompt: The compiled LangChain prompt template.
            messages: Chat history as LangChain message objects.
            question: The user's current question.

        Yields:
            Text chunks (str) as they arrive from the LLM.

        Raises:
            AllGroqApisFailedError: If every API key fails.
        """
        n = len(self.llms)
        last_exc = None

        for i in range(n):
            masked_key = _mask_api_key(GROQ_API_KEYS[i])
            logger.info(f"Streaming with API key #{i + 1}/{n}: {masked_key}")

            try:
                chain = prompt | self.llms[i]
                chunk_count = 0
                first_chunk_time = None
                stream_start = time.perf_counter()

                # chain.stream() returns a generator that yields chunk objects
                # as the LLM produces tokens. The HTTP connection stays open
                # (Server-Sent Events) until the LLM finishes generating.
                for chunk in chain.stream({"history": messages, "question": question}):
                    # Extract text content from the chunk. LangChain chunks can be
                    # either objects with a .content attribute or plain dicts.
                    content = ""
                    if hasattr(chunk, "content"):
                        content = chunk.content or ""
                    elif isinstance(chunk, dict) and "content" in chunk:
                        content = chunk.get("content", "") or ""

                    if isinstance(content, str) and content:
                        # Log the Time To First Token — critical for UX.
                        # This tells us how long the user waited before seeing
                        # any response text appear on screen.
                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter() - stream_start
                            _log_timing("first_chunk", first_chunk_time)
                        chunk_count += 1
                        # Yield the text to our caller (ChatService), which forwards
                        # it to the client via SSE for the real-time typing effect.
                        yield content

                total_stream = time.perf_counter() - stream_start
                _log_timing("groq_stream_total", total_stream, f"chunks: {chunk_count}")

                if i > 0 and chunk_count > 0:
                    logger.info(f"Fallback successful: API key #{i + 1}/{n} streamed: {masked_key}")
                return  # Stream completed successfully — exit the fallback loop.

            except Exception as e:
                last_exc = e
                if _is_rate_limit_error(e):
                    logger.warning(f"API key #{i + 1}/{n} rate limited: {masked_key}")
                else:
                    logger.warning(f"API key #{i + 1}/{n} failed: {masked_key} - {str(e)[:100]}")
                if i < n - 1:
                    # More keys available — try the next one with a fresh stream.
                    logger.info("Falling back to next API key for stream...")
                    continue
                break

        logger.error(f"All {n} API key(s) failed during stream.")
        raise AllGroqApisFailedError(ALL_APIS_FAILED_MESSAGE) from last_exc

    # ─── PROMPT ASSEMBLY ─────────────────────────────────────────────────────
    # This method builds everything the LLM needs: system message, history, and
    # the prompt template. It's used by both get_response and stream_response,
    # and also by the realtime subclass (which passes extra_system_parts for
    # web search results).
    #
    # THE SYSTEM MESSAGE IS BUILT IN 5 LAYERS:
    #   Layer 1: Base personality (JARVIS_SYSTEM_PROMPT)
    #   Layer 2: Current time/date (so the LLM knows "today" and "this week")
    #   Layer 3: RAG context (vector store retrieval results)
    #   Layer 4: Extra parts (web search results, only in realtime mode)
    #   Layer 5: Mode addendum (role-specific instructions)
    #
    # Each layer is optional (except Layer 1) and is only added if content exists.

    def _build_prompt_and_messages(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
        extra_system_parts: Optional[List[str]] = None,
        mode_addendum: str = "",
    ) -> tuple:
        """
        Retrieve context from the vector store and assemble the full LLM prompt.

        THIS IS WHERE THE MAGIC HAPPENS. The system message is built layer by layer:

        LAYER 1 — BASE PERSONALITY (JARVIS_SYSTEM_PROMPT):
          The core personality and instructions for JARVIS. Defines tone, behavior,
          and capabilities. Loaded from config.py. This is the "soul" of JARVIS —
          it tells the LLM who it is, how it should behave, and what it can/can't do.

        LAYER 2 — CURRENT TIME:
          "Current time and date: Saturday, February 21, 2026, 3:45 PM IST"
          This lets the LLM give time-aware answers ("today", "this week", etc.).
          Without this, the LLM has no idea what "today" means — it only knows
          its training data cutoff date.

        LAYER 3 — VECTOR STORE CONTEXT (RAG — Retrieval-Augmented Generation):
          We query the vector store for the k=10 most similar chunks to the user's
          question. These chunks come from:
            - Learning data files (PDFs, text files the user uploaded)
            - Past chat sessions (saved JSON files)
          The chunks are concatenated and injected into the system message as
          "Relevant context from your learning data and past conversations: ..."
          This is what makes JARVIS personalized — it answers based on YOUR data.

          WHAT IS RAG?
            RAG stands for "Retrieval-Augmented Generation". Instead of relying
            solely on the LLM's training data, we RETRIEVE relevant documents
            first, then AUGMENT the prompt with them, so the LLM can GENERATE
            answers grounded in your actual data. This dramatically reduces
            hallucination and makes the LLM "know" things it was never trained on.

          WHY k=10: More chunks = more context for the LLM, but also more tokens
          (and cost). 10 is a good balance: enough to find relevant info, not so
          many that we flood the prompt with irrelevant content or hit token limits.

        LAYER 4 — EXTRA SYSTEM PARTS (optional):
          Used by RealtimeGroqService to inject Tavily web search results. The
          general chat path passes None here. This is the extension point that
          makes the inheritance design work — the subclass doesn't need to
          duplicate prompt assembly, it just passes extra content.

        LAYER 5 — MODE ADDENDUM:
          A role-specific instruction block. For general chat, it might say
          "Answer based on your knowledge and the provided context." For realtime
          chat, it says "Incorporate the web search results into your answer."
          This steers the LLM's behavior based on which mode is active.

        PROMPT TEMPLATE STRUCTURE:
          The final prompt template looks like:
            [system]  <everything above, concatenated>
            [history]  <MessagesPlaceholder — expands to HumanMessage/AIMessage pairs>
            [human]   {question}  <the current user question>

          This three-part structure (system → history → question) is the standard
          pattern for chat LLMs. The system message sets the rules, the history
          provides conversation context, and the human message is the current query.

        CHAT HISTORY CONVERSION:
          The chat_history comes in as [(user_text, ai_text), ...] tuples from
          ChatService.format_history_for_llm(). We convert each tuple into a
          HumanMessage + AIMessage pair for LangChain's MessagesPlaceholder.
          This conversion is necessary because LangChain expects its own message
          types, not plain tuples.

        Args:
            question: The user's current question.
            chat_history: List of (user_text, assistant_text) tuples.
            extra_system_parts: Optional strings to append (e.g. search results).
            mode_addendum: Role-specific instructions (general vs realtime).

        Returns:
            (prompt, messages) where:
              - prompt: A ChatPromptTemplate ready to be piped into an LLM.
              - messages: List of HumanMessage/AIMessage for the history placeholder.
        """
        # ── Step 1: Retrieve context from the vector store (RAG) ──
        # This is the "R" in RAG. We search for document chunks that are
        # semantically similar to the user's question.
        context = ""
        context_sources = []
        t0 = time.perf_counter()
        try:
            # get_retriever(k=10) returns a LangChain retriever that, when called,
            # performs a similarity search in the vector store and returns the
            # k nearest document chunks. Under the hood, this computes the cosine
            # similarity between the question's embedding and all stored embeddings.
            retriever = self.vector_store_service.get_retriever(k=10)
            # .invoke(question) runs the actual similarity search against the
            # embedded question and returns a list of Document objects.
            # Each Document has .page_content (the text) and .metadata (source info).
            context_docs = retriever.invoke(question)
            if context_docs:
                # Concatenate all chunk texts. Each doc.page_content is a text chunk
                # from a learning file or past chat session. We join with newlines
                # so the LLM sees them as separate paragraphs.
                context = "\n".join([doc.page_content for doc in context_docs])
                # Track sources for logging (helps debug "where did that answer come from?").
                context_sources = [doc.metadata.get("source", "unknown") for doc in context_docs]
                logger.info("[CONTEXT] Retrieved %d chunks from sources: %s", len(context_docs), context_sources)
            else:
                logger.info("[CONTEXT] No relevant chunks found for query")
        except Exception as retrieval_err:
            # If the vector store is broken, we proceed with empty context.
            # The LLM can still answer from its training data — just without
            # personalized context. This is a graceful degradation pattern:
            # partial functionality is better than a complete failure.
            logger.warning("Vector store retrieval failed, using empty context: %s", retrieval_err)
        finally:
            _log_timing("vector_db", time.perf_counter() - t0)

        # ── Step 2: Build the system message layer by layer ──
        # Each layer is concatenated to form one large system message string.
        time_info = get_time_information()
        system_message = JARVIS_SYSTEM_PROMPT  # Layer 1: base personality

        system_message += f"\n\nCurrent time and date: {time_info}"  # Layer 2: time awareness

        if context:
            # Layer 3: RAG context. Note: we escape curly braces because this
            # text is injected into a LangChain template (see escape_curly_braces docs).
            # Without escaping, code snippets in the context would crash the template engine.
            system_message += f"\n\nRelevant context from your learning data and past conversations:\n{escape_curly_braces(context)}"

        if extra_system_parts:
            # Layer 4: additional content (e.g. web search results from realtime mode).
            # extra_system_parts is a list of strings; we join them with double newlines.
            system_message += "\n\n" + "\n\n".join(extra_system_parts)

        if mode_addendum:
            # Layer 5: mode-specific instructions (e.g. "use the search results").
            system_message += f"\n\n{mode_addendum}"

        # ── Step 3: Build the LangChain prompt template ──
        # ChatPromptTemplate.from_messages() creates a template with three slots:
        #   1. ("system", ...) — the system message (rendered as-is, no variables inside
        #      because we already escaped curly braces in the context).
        #   2. MessagesPlaceholder("history") — expands to the chat history messages.
        #      At invoke time, LangChain replaces this with the actual messages list.
        #   3. ("human", "{question}") — the current user question. The {question}
        #      placeholder is filled by chain.invoke({"question": "..."}).
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        # ── Step 4: Convert chat history tuples to LangChain message objects ──
        # ChatService gives us tuples: [("Hello", "Hi!"), ("How are you?", "I'm good")]
        # LangChain needs: [HumanMessage("Hello"), AIMessage("Hi!"), HumanMessage("How are you?"), AIMessage("I'm good")]
        messages = []
        if chat_history:
            for human_msg, ai_msg in chat_history:
                messages.append(HumanMessage(content=human_msg))
                messages.append(AIMessage(content=ai_msg))

        logger.info("[PROMPT] System message length: %d chars | History pairs: %d | Question: %.100s",
                     len(system_message), len(chat_history) if chat_history else 0, question)

        return prompt, messages

    # ─── PUBLIC API ──────────────────────────────────────────────────────────
    # These are the methods that ChatService calls. They orchestrate the full
    # flow: build prompt → call LLM → return result.
    #
    # RealtimeGroqService overrides these to add web search results before
    # calling the LLM. The base implementations here do NOT include web search.
    #
    # ERROR HANDLING STRATEGY:
    #   - AllGroqApisFailedError: re-raised as-is (the API layer handles it).
    #   - Any other exception: wrapped with context for easier debugging.
    #   This two-tier approach means the API layer can distinguish between
    #   "all keys failed" (503) and "something unexpected broke" (500).

    def get_response(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None
    ) -> str:
        """
        Return the assistant's reply for a general chat question (no web search).

        FLOW:
          1. _build_prompt_and_messages: retrieve context + build system prompt.
          2. _invoke_llm: call Groq with primary-first key fallback.
          3. Return the response text.

        The GENERAL_CHAT_ADDENDUM is passed as mode_addendum — it contains
        instructions specific to general chat mode (e.g. "rely on your knowledge
        and the provided context, do not make up URLs").

        Error handling:
          - AllGroqApisFailedError: re-raised as-is (all keys exhausted).
          - Any other exception: wrapped with a descriptive message so the caller
            knows the error originated from the Groq service.
        """
        try:
            prompt, messages = self._build_prompt_and_messages(
                question, chat_history, mode_addendum=GENERAL_CHAT_ADDENDUM,
            )
            t0 = time.perf_counter()
            result = self._invoke_llm(prompt, messages, question)
            _log_timing("groq_api", time.perf_counter() - t0)
            logger.info("[RESPONSE] General chat | Length: %d chars | Preview: %.120s", len(result), result)
            return result
        except AllGroqApisFailedError:
            raise
        except Exception as e:
            raise Exception(f"Error getting response from Groq: {str(e)}") from e

    def stream_response(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
    ) -> Iterator[str]:
        """
        Stream the assistant's reply token-by-token for general chat (no web search).

        Same as get_response but uses _stream_llm instead of _invoke_llm.
        Returns a generator that yields text chunks as the LLM produces them.

        This is used by ChatService.process_message_stream to implement
        Server-Sent Events (SSE) for real-time typing effect in the UI.

        GENERATOR DELEGATION (yield from):
          `yield from self._stream_llm(...)` delegates the entire generator to
          _stream_llm. Every chunk that _stream_llm yields is automatically
          yielded by this method. This is cleaner than a manual for loop:
            for chunk in self._stream_llm(...):
                yield chunk
          Both are equivalent, but `yield from` is the Pythonic way.
        """
        try:
            prompt, messages = self._build_prompt_and_messages(
                question, chat_history, mode_addendum=GENERAL_CHAT_ADDENDUM,
            )
            yield from self._stream_llm(prompt, messages, question)
        except AllGroqApisFailedError:
            raise
        except Exception as e:
            raise Exception(f"Error streaming response from Groq: {str(e)}") from e
