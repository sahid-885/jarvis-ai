"""
REALTIME GROQ SERVICE MODULE
=============================

Extends GroqService to add Tavily web search before calling the LLM. Used by
ChatService for POST /chat/realtime. Same session and history as general chat;
the only difference is we run a Tavily search for the user's question and add
the results to the system message, then call Groq.

INHERITANCE MODEL:
  RealtimeGroqService extends GroqService (Python class inheritance). This means
  it automatically gets ALL of GroqService's functionality:
    - __init__: Creates LLM clients for each API key + stores vector store.
    - _invoke_llm / _stream_llm: Multi-key fallback logic.
    - _build_prompt_and_messages: System prompt assembly (5-layer approach).
  RealtimeGroqService only OVERRIDES get_response() and stream_response() to
  inject web search results into the system prompt before calling the LLM.

  WHY INHERITANCE (not composition):
    Since RealtimeGroqService IS-A GroqService with extra functionality (web
    search), inheritance is the natural fit. Composition would require wrapping
    every method, which adds boilerplate without benefit. The override pattern
    (call search → inject results → call parent's prompt builder) is clean.

KEY FEATURES:
  - Smart query extraction: converts messy conversational text into focused
    search queries using a fast LLM call with chat history context.
  - Advanced Tavily search: deep search with AI-generated answer synthesis.
  - Rich result formatting: structured results with Tavily AI answer + individual
    source details for the LLM to build comprehensive replies.
  - Inherits primary-first API key fallback from GroqService.

FLOW:
  1. _extract_search_query(question, chat_history): fast LLM call to produce a
     clean, focused search query from the user's raw conversational text.
  2. search_tavily(query): call Tavily with advanced depth, AI answer, and
     format results as structured text.
  3. get_response / stream_response: inject search results into system message,
     then call Groq as usual.

  VISUAL FLOW:
    User: "What about him?" (referencing someone from chat history)
      │
      ├─ _extract_search_query ──→ "Elon Musk latest news 2026"
      │   (uses chat history to resolve "him" → "Elon Musk")
      │
      ├─ search_tavily ──→ formatted search results (titles, URLs, AI answer)
      │
      ├─ _build_prompt_and_messages ──→ system prompt WITH search results
      │   (inherited from GroqService, extra_system_parts = search results)
      │
      └─ _invoke_llm / _stream_llm ──→ LLM response
          (inherited from GroqService, multi-key fallback)

GRACEFUL DEGRADATION:
  If TAVILY_API_KEY is not set, tavily_client is None and search_tavily returns "";
  the user still gets an answer from Groq with no search results. Similarly, if
  the Tavily API call fails, we log the error and proceed without search results.
  The user always gets a response — it's just less informed without web data.
"""

from typing import List, Optional, Iterator, Tuple, Any
from tavily import TavilyClient
import logging
import os
import time

from app.services.groq_service import GroqService, escape_curly_braces, AllGroqApisFailedError
from app.services.vector_store import VectorStoreService
from app.utils.retry import with_retry
from config import REALTIME_CHAT_ADDENDUM, GROQ_API_KEYS, GROQ_MODEL


logger = logging.getLogger("J.A.R.V.I.S")

# ─── CONSTANTS ───────────────────────────────────────────────────────────────

# Timeout for the fast LLM used in query extraction (15 seconds).
# This is much lower than the main LLM timeout (60s) because query extraction
# is a simple task — if it takes more than 15 seconds, something is wrong and
# we should fall back to using the raw question.
GROQ_REQUEST_TIMEOUT_FAST = 15

# ─── QUERY EXTRACTION PROMPT ────────────────────────────────────────────────
# This prompt is sent to the fast LLM to convert messy conversational text into
# a clean search query. The key instructions are:
#   - "max 12 words": keeps the query focused and search-engine-friendly.
#   - "Resolve any references": uses chat history to expand pronouns and
#     references like "that website", "him", "it" into concrete terms.
#   - "Output ONLY the search query": prevents the LLM from adding explanations
#     or formatting that would pollute the search.
#
# EXAMPLE:
#   Chat history: "User: Tell me about Elon Musk"
#   User's message: "What's his net worth?"
#   Extracted query: "Elon Musk net worth 2026"
#   (The LLM resolved "his" → "Elon Musk" using the chat history.)
_QUERY_EXTRACTION_PROMPT = (
    "You are a search query optimizer. Given the user's message and recent conversation, "
    "produce a single short, focused web search query (max 12 words) that will find the "
    "information the user needs. Resolve any references (like 'that website', 'him', 'it') "
    "using the conversation history. Output ONLY the search query, nothing else."
)


# =============================================================================
# REALTIME GROQ SERVICE CLASS (extends GroqService)
# =============================================================================

class RealtimeGroqService(GroqService):
    """
    Extends GroqService to add Tavily web search before calling the LLM.

    HOW IT WORKS:
      1. Inherits ALL capabilities from GroqService (LLM clients, multi-key
         fallback, prompt building, vector store retrieval).
      2. Adds two new capabilities:
         a. _fast_llm: A lightweight LLM client dedicated to query extraction
            (small max_tokens, low temperature, short timeout).
         b. tavily_client: A Tavily API client for web search.
      3. Overrides get_response() and stream_response() to:
         - Extract a clean search query from the user's message.
         - Run a Tavily web search with that query.
         - Pass the search results as extra_system_parts to the parent's
           _build_prompt_and_messages() method.
         - Call the parent's _invoke_llm() / _stream_llm() as usual.

    GRACEFUL DEGRADATION HIERARCHY:
      If TAVILY_API_KEY is missing → no web search, answer from LLM + vector store only.
      If query extraction fails → use raw user message as search query.
      If Tavily search fails → answer from LLM + vector store only.
      If all Groq keys fail → raise AllGroqApisFailedError.
      The user ALWAYS gets a response unless all LLM keys are exhausted.

    WHY A SEPARATE _fast_llm:
      Query extraction is a tiny task (input: one sentence, output: ~10 words).
      Using the main LLM (temperature=0.6, timeout=60s, unlimited tokens) would
      be wasteful. The _fast_llm is tuned for speed and determinism:
        - temperature=0.0: We want the same query every time (no creativity needed).
        - max_tokens=50: The output is at most 12 words; 50 tokens is plenty.
        - timeout=15s: If it's slower than this, something's wrong — fall back.
      This makes query extraction fast (~200-500ms) and cheap (~10 tokens).
    """

    def __init__(self, vector_store_service: VectorStoreService):
        """
        Initialize the realtime service: parent LLM clients + Tavily + fast LLM.

        INITIALIZATION ORDER:
          1. super().__init__(): Creates the main LLM clients (one per API key)
             and stores the vector store reference. This is the GroqService init.
          2. Tavily client: Created if TAVILY_API_KEY is set in the environment.
             If not set, self.tavily_client = None and search_tavily() returns "".
          3. _fast_llm: A lightweight ChatGroq client used ONLY for query extraction.
             Uses the primary API key (GROQ_API_KEYS[0]) since query extraction
             is very cheap and unlikely to trigger rate limits.

        Args:
            vector_store_service: The vector store for context retrieval (passed
                                  through to GroqService.__init__).
        """
        # Call GroqService.__init__ to create self.llms and self.vector_store_service.
        # super().__init__() is how Python calls the parent class's constructor.
        super().__init__(vector_store_service)

        # ── Set up Tavily web search client ──
        # os.getenv reads from environment variables (loaded from .env by the config module).
        # If TAVILY_API_KEY is empty or missing, we set tavily_client to None.
        tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        if tavily_api_key:
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
            logger.info("Tavily search client initialized successfully")
        else:
            self.tavily_client = None
            logger.warning("TAVILY_API_KEY not set. Realtime search will be unavailable.")

        # ── Set up the fast LLM for query extraction ──
        # This is a separate, lightweight LLM client optimized for the small task
        # of converting user messages into search queries.
        if GROQ_API_KEYS:
            from langchain_groq import ChatGroq
            self._fast_llm = ChatGroq(
                groq_api_key=GROQ_API_KEYS[0],  # Uses primary key only (tiny request, won't rate-limit).
                model_name=GROQ_MODEL,
                temperature=0.0,  # Fully deterministic — we want consistent query extraction.
                request_timeout=GROQ_REQUEST_TIMEOUT_FAST,  # 15s timeout (fail fast).
                max_tokens=50,  # Output is ~10 words max, 50 tokens is more than enough.
            )
        else:
            self._fast_llm = None

    # ─── QUERY EXTRACTION ────────────────────────────────────────────────────
    # Raw user messages are often conversational and contain pronouns, references,
    # and incomplete sentences. Search engines work best with concise, keyword-rich
    # queries. This method bridges that gap.
    #
    # EXAMPLES OF WHY THIS IS NEEDED:
    #   Raw: "Yeah tell me more about what he did yesterday"
    #   Extracted: "Elon Musk activities February 20 2026"
    #
    #   Raw: "Can you look up that Python library we talked about?"
    #   Extracted: "LangChain Python library"
    #
    #   Raw: "what's the weather like"
    #   Extracted: "weather today" (simple messages pass through mostly unchanged)

    def _extract_search_query(
        self, question: str, chat_history: Optional[List[tuple]] = None
    ) -> str:
        """
        Use a fast LLM call to convert the user's raw conversational message into
        a clean, focused search query. Resolves references ("that website", "him")
        using recent chat history. Falls back to the raw question on any failure.

        HOW IT WORKS:
          1. Take the last 3 turns of chat history (to keep the context small).
          2. Format them as "User: ... / Assistant: ..." pairs.
          3. Send them along with the user's current message to the fast LLM,
             using _QUERY_EXTRACTION_PROMPT as instructions.
          4. The LLM returns a clean search query (e.g. "Python 3.12 new features").
          5. Validate the result: must be 3-200 characters and non-empty.
          6. If anything goes wrong (LLM error, bad output), return the raw question.

        WHY ONLY LAST 3 TURNS:
          We truncate to the last 3 history pairs because:
          a. Query extraction is a lightweight task — it doesn't need full context.
          b. Fewer tokens = faster response and lower cost.
          c. Recent context is most relevant for resolving references ("him", "it").
          d. We also truncate each message to 200 chars to further limit tokens.

        FALLBACK BEHAVIOR:
          This method NEVER raises exceptions. If the fast LLM fails (network error,
          rate limit, bad output), we silently return the raw question. The search
          may be less focused, but the user still gets results.

        Args:
            question: The user's raw conversational message.
            chat_history: List of (user_text, assistant_text) tuples for context.

        Returns:
            A clean search query string (or the raw question as fallback).
        """
        # If no fast LLM is available (no API keys configured), skip extraction.
        if not self._fast_llm:
            return question

        try:
            t0 = time.perf_counter()

            # Build a condensed version of recent chat history for context.
            # We only include the last 3 turns and truncate each message to 200 chars
            # to keep the prompt small (fast + cheap).
            history_context = ""
            if chat_history:
                recent = chat_history[-3:]  # Last 3 conversation turns only.
                parts = []
                for h, a in recent:
                    parts.append(f"User: {h[:200]}")      # Truncate long messages.
                    parts.append(f"Assistant: {a[:200]}")
                history_context = "\n".join(parts)

            # Build the full prompt for the fast LLM.
            # If we have history, include it so the LLM can resolve references.
            # If not, just send the user's message directly.
            if history_context:
                full_prompt = (
                    f"{_QUERY_EXTRACTION_PROMPT}\n\n"
                    f"Recent conversation:\n{history_context}\n\n"
                    f"User's latest message: {question}\n\n"
                    f"Search query:"
                )
            else:
                full_prompt = (
                    f"{_QUERY_EXTRACTION_PROMPT}\n\n"
                    f"User's message: {question}\n\n"
                    f"Search query:"
                )

            # Call the fast LLM directly (no chain, no template — just a simple invoke).
            # This is much simpler than the main LLM call because we don't need
            # history placeholders or system message layering.
            response = self._fast_llm.invoke(full_prompt)
            # Clean up the response: strip whitespace and quotes that the LLM might add.
            extracted = response.content.strip().strip('"').strip("'")

            # Validate: the extracted query must be between 3 and 200 characters.
            # Too short (< 3 chars) means the LLM returned garbage.
            # Too long (> 200 chars) means the LLM wrote an explanation instead of a query.
            if extracted and 3 <= len(extracted) <= 200:
                logger.info(
                    "[REALTIME] Query extraction: '%s' -> '%s' (%.3fs)",
                    question[:80], extracted[:80], time.perf_counter() - t0,
                )
                return extracted

            logger.warning("[REALTIME] Query extraction returned unusable result, using raw question")
            return question

        except Exception as e:
            # Catch ALL exceptions — query extraction is optional, not critical.
            # If it fails, the raw question is still a reasonable search query.
            logger.warning("[REALTIME] Query extraction failed (%s), using raw question", e)
            return question

    # ─── TAVILY WEB SEARCH ───────────────────────────────────────────────────
    # Tavily is a search API designed for LLM applications. Unlike Google Search
    # API which returns links, Tavily returns structured content (extracted text,
    # relevance scores) and can even synthesize an AI answer from the results.
    #
    # WHY TAVILY (not Google Search API / Bing API):
    #   - Returns extracted page content (not just URLs) — no need to scrape.
    #   - "Advanced" search depth does deeper crawling for better results.
    #   - include_answer=True gives us an AI-synthesized answer we can use
    #     as the primary source, with individual results as supporting evidence.
    #   - Designed for RAG pipelines — the output is LLM-friendly.

    def search_tavily(self, query: str, num_results: int = 7) -> str:
        """
        Call Tavily API with advanced search depth and AI answer synthesis.
        Returns richly formatted results for the LLM to consume.

        HOW IT WORKS:
          1. Call self.tavily_client.search() with these key parameters:
             - search_depth="advanced": Tavily crawls deeper into pages for
               better content extraction (vs "basic" which only reads metadata).
             - max_results=7: Return up to 7 source pages. 7 is a balance between
               comprehensiveness and token cost.
             - include_answer=True: Tavily's AI synthesizes a short answer from
               all results. This is extremely useful — it gives the LLM a
               pre-digested summary to build on.
             - include_raw_content=False: We don't need full page HTML (too large).
          2. The with_retry wrapper retries up to 3 times with 1-second initial
             delay if the Tavily API is temporarily unavailable.
          3. Format the results into a structured text block:
             - Header: "=== WEB SEARCH RESULTS FOR: {query} ==="
             - AI answer (if available): "AI-SYNTHESIZED ANSWER: ..."
             - Individual sources: numbered, with title, content, URL, and
               relevance score.
             - Footer: "=== END SEARCH RESULTS ==="

        RESULT FORMATTING:
          The structured format (headers, labels, numbered sources) helps the LLM
          understand the search results clearly. The relevance score (0.0-1.0)
          tells the LLM which sources are most trustworthy. The AI-synthesized
          answer is marked as "primary source" so the LLM prioritizes it.

        EXAMPLE OUTPUT:
          === WEB SEARCH RESULTS FOR: Python 3.12 new features ===
          AI-SYNTHESIZED ANSWER (use this as your primary source):
          Python 3.12 introduces improved error messages, a new type parameter...

          INDIVIDUAL SOURCES:
          [Source 1] (relevance: 0.95)
          Title: What's New in Python 3.12
          Content: Python 3.12 was released on October 2, 2023...
          URL: https://docs.python.org/3/whatsnew/3.12.html
          ...
          === END SEARCH RESULTS ===

        Args:
            query: The search query (ideally from _extract_search_query).
            num_results: Maximum number of results to fetch (default 7).

        Returns:
            A tuple (formatted_string, payload_dict).
            - formatted_string: Text block for the LLM prompt, or "" if no results.
            - payload_dict: JSON-serializable dict for the frontend widget, or None.
              Contains "query", "answer", "results" (list of {title, content, url, score}).
        """
        # Guard: if Tavily client wasn't initialized (no API key), return empty.
        if not self.tavily_client:
            logger.warning("Tavily client not initialized. TAVILY_API_KEY not set.")
            return ("", None)

        try:
            t0 = time.perf_counter()

            # Call Tavily with retry logic for resilience against transient failures.
            response = with_retry(
                lambda: self.tavily_client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=num_results,
                    include_answer=True,
                    include_raw_content=False,
                ),
                max_retries=3,
                initial_delay=1.0,
            )

            results = response.get("results", [])
            ai_answer = response.get("answer", "")

            if not results and not ai_answer:
                logger.warning("No Tavily search results for query: %s", query)
                return ("", None)

            # Build payload for the frontend (right-side search results widget).
            payload: Optional[dict] = {
                "query": query,
                "answer": ai_answer,
                "results": [
                    {
                        "title": r.get("title", "No title"),
                        "content": (r.get("content") or "")[:500],  # Snippet for widget
                        "url": r.get("url", ""),
                        "score": round(float(r.get("score", 0)), 2),
                    }
                    for r in results[:num_results]
                ],
            }

            # Format results into a structured text block for the LLM.
            parts = [f"=== WEB SEARCH RESULTS FOR: {query} ===\n"]
            if ai_answer:
                parts.append(f"AI-SYNTHESIZED ANSWER (use this as your primary source):\n{ai_answer}\n")
            if results:
                parts.append("INDIVIDUAL SOURCES:")
                for i, result in enumerate(results[:num_results], 1):
                    title = result.get("title", "No title")
                    content = result.get("content", "")
                    url = result.get("url", "")
                    score = result.get("score", 0)
                    parts.append(f"\n[Source {i}] (relevance: {score:.2f})")
                    parts.append(f"Title: {title}")
                    if content:
                        parts.append(f"Content: {content}")
                    if url:
                        parts.append(f"URL: {url}")
            parts.append("\n=== END SEARCH RESULTS ===")
            formatted = "\n".join(parts)

            logger.info(
                "[TAVILY] %d results, AI answer: %s, formatted: %d chars (%.3fs)",
                len(results), "yes" if ai_answer else "no",
                len(formatted), time.perf_counter() - t0,
            )
            return (formatted, payload)

        except Exception as e:
            logger.error("Error performing Tavily search: %s", e)
            return ("", None)

    # ─── PUBLIC API OVERRIDES ────────────────────────────────────────────────
    # These methods OVERRIDE the parent GroqService's get_response/stream_response.
    # The override pattern is:
    #   1. Extract a clean search query (new functionality).
    #   2. Run Tavily web search (new functionality).
    #   3. Call parent's _build_prompt_and_messages with extra_system_parts (reuse).
    #   4. Call parent's _invoke_llm / _stream_llm (reuse).
    #
    # This is the "Template Method" design pattern: the parent defines the
    # skeleton (build prompt → call LLM), and the child fills in extra steps
    # (search query extraction → web search → inject results).

    def get_response(self, question: str, chat_history: Optional[List[tuple]] = None) -> str:
        """
        Get a realtime response: extract query → search web → call LLM with results.

        This OVERRIDES GroqService.get_response() to add web search functionality.
        The parent's version only uses vector store context; this version ALSO
        includes Tavily web search results in the system prompt.

        DETAILED FLOW:
          1. _extract_search_query: Convert the user's raw message into a clean
             search query using the fast LLM + chat history for reference resolution.
          2. search_tavily: Call Tavily API with the clean query. Returns formatted
             results (AI answer + individual sources) or "" if search failed.
          3. escape_curly_braces: Sanitize the search results for LangChain template
             injection (same reason as in GroqService — curly braces would crash it).
          4. _build_prompt_and_messages: Build the system prompt with all 5 layers,
             passing search results as extra_system_parts (Layer 4).
          5. _invoke_llm: Call Groq with primary-first multi-key fallback.
          6. Return the LLM's response.

        DIFFERENCE FROM PARENT'S get_response:
          Parent:   _build_prompt_and_messages(extra_system_parts=None)  → no search
          This:     _build_prompt_and_messages(extra_system_parts=[search_results])  → with search

        Args:
            question: The user's raw message.
            chat_history: List of (user_text, assistant_text) tuples.

        Returns:
            The LLM's response text, informed by web search results.

        Raises:
            AllGroqApisFailedError: If all Groq API keys fail.
        """
        try:
            # Step 1: Extract a clean, focused search query from the user's message.
            # Example: "tell me more about him" → "Elon Musk latest news 2026"
            search_query = self._extract_search_query(question, chat_history)
            logger.info("[REALTIME] Searching Tavily for: %s", search_query)

            # Step 2: Run Tavily web search (returns formatted string for prompt + payload for UI).
            formatted_results, _ = self.search_tavily(search_query, num_results=7)
            if formatted_results:
                logger.info("[REALTIME] Tavily returned results (length: %d chars)", len(formatted_results))
            else:
                logger.warning("[REALTIME] Tavily returned no results for: %s", search_query)

            # Step 3: Build the prompt with search results injected as extra_system_parts.
            extra_parts = [escape_curly_braces(formatted_results)] if formatted_results else None
            prompt, messages = self._build_prompt_and_messages(
                question, chat_history,
                extra_system_parts=extra_parts,
                mode_addendum=REALTIME_CHAT_ADDENDUM,  # Realtime-specific instructions for the LLM.
            )

            # Step 4: Call the LLM with multi-key fallback (inherited from GroqService).
            t0 = time.perf_counter()
            response_content = self._invoke_llm(prompt, messages, question)
            logger.info("[TIMING] groq_api: %.3fs", time.perf_counter() - t0)
            logger.info(
                "[RESPONSE] Realtime chat | Length: %d chars | Preview: %.120s",
                len(response_content), response_content,
            )
            return response_content

        except AllGroqApisFailedError:
            raise
        except Exception as e:
            logger.error("Error in realtime get_response: %s", e, exc_info=True)
            raise

    def stream_response(self, question: str, chat_history: Optional[List[tuple]] = None) -> Iterator[Any]:
        """
        Stream the realtime response: first yield search_results payload for the
        frontend widget, then yield text chunks from the LLM.

        Yields:
            First: a dict {"_search_results": payload} if search returned data.
            Then: text chunks (str) from the LLM.
        """
        try:
            search_query = self._extract_search_query(question, chat_history)
            logger.info("[REALTIME] Searching Tavily for: %s", search_query)

            formatted_results, payload = self.search_tavily(search_query, num_results=7)
            if formatted_results:
                logger.info("[REALTIME] Tavily returned results (length: %d chars)", len(formatted_results))
            else:
                logger.warning("[REALTIME] Tavily returned no results for: %s", search_query)

            # Send search results to the client for the right-side widget (before any text).
            if payload:
                yield {"_search_results": payload}

            extra_parts = [escape_curly_braces(formatted_results)] if formatted_results else None
            prompt, messages = self._build_prompt_and_messages(
                question, chat_history,
                extra_system_parts=extra_parts,
                mode_addendum=REALTIME_CHAT_ADDENDUM,
            )
            yield from self._stream_llm(prompt, messages, question)
            logger.info("[REALTIME] Stream completed for: %s", search_query)

        except AllGroqApisFailedError:
            raise
        except Exception as e:
            logger.error("Error in realtime stream_response: %s", e, exc_info=True)
            raise
