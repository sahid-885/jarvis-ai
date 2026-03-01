"""
RETRY UTILITY
=============

Calls a function and, if it raises, retries a few times with exponential backoff.
Used for Groq and Tavily API calls so temporary rate limits or network blips
don't immediately fail the request.

Example:
  response = with_retry(lambda: groq_client.chat(...), max_retries=3, initial_delay=1.0)

─────────────────────────────────────────────────────────────────────────────
WHAT IS "RETRY WITH EXPONENTIAL BACKOFF"?
─────────────────────────────────────────────────────────────────────────────
When calling external APIs (Groq for LLM inference, Tavily for web search),
transient failures are common:
  - Rate limits (HTTP 429): "slow down, you're sending too many requests."
  - Network timeouts: a packet got lost.
  - Server errors (HTTP 500/503): the remote service is temporarily overloaded.

A simple retry (try again immediately) can make things worse — if the server
is overloaded and thousands of clients all retry instantly, the server stays
overloaded.

**Exponential backoff** solves this by waiting longer between each retry:
  - Attempt 1 fails → wait 1 second
  - Attempt 2 fails → wait 2 seconds  (delay doubles)
  - Attempt 3 fails → wait 4 seconds  (delay doubles again)
  - Attempt 4 fails → give up and raise the error

This spreading-out of retries gives the server time to recover.

─────────────────────────────────────────────────────────────────────────────
WHERE THIS IS USED IN JARVIS
─────────────────────────────────────────────────────────────────────────────
  - GroqService wraps its chat-completion calls with with_retry() so that a
    momentary Groq rate limit doesn't kill the user's request.
  - RealtimeGroqService wraps Tavily web-search calls the same way.
"""

import logging
import time
from typing import TypeVar, Callable


logger = logging.getLogger("J.A.R.V.I.S")

# ─── Generic type variable ──────────────────────────────────────────────────
# TypeVar("T") lets us write a *generic* function signature:
#
#   def with_retry(fn: Callable[[], T], ...) -> T
#
# This tells the type checker: "whatever type fn() returns, with_retry returns
# the same type."  So if fn returns a string, with_retry returns a string; if fn
# returns a ChatCompletion, with_retry returns a ChatCompletion.
# Without TypeVar, we'd have to use `Any`, losing type safety.
T = TypeVar("T")


def with_retry(
    fn: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
) -> T:
    """
    Execute fn(). If it raises, wait and try again with exponential backoff.
    After max_retries total attempts (including the first), re-raise the last exception.

    HOW THE RETRY LOOP WORKS:
      1. Call fn(). If it succeeds, return the result immediately.
      2. If fn() raises an exception:
         a. If this was the last allowed attempt, re-raise — we've exhausted retries.
         b. Otherwise, log a warning, sleep for `delay` seconds, then double the
            delay and try the next attempt.
      3. The delay sequence is: initial_delay, initial_delay*2, initial_delay*4, ...
         e.g. with initial_delay=1.0: 1s → 2s → 4s → 8s → ...

    Args:
        fn:            A zero-argument callable to execute. Typically a lambda wrapping
                       an API call, e.g. lambda: client.chat.completions.create(...).
        max_retries:   Total number of attempts (including the first). Default 3 means
                       "try once, and if it fails, retry up to 2 more times."
        initial_delay: Seconds to wait after the first failure. Default 1.0 second.

    Returns:
        Whatever fn() returns on a successful call.

    Raises:
        The exception from the last failed attempt if all retries are exhausted.
    """
    last_exception = None
    delay = initial_delay  # will be doubled after each failure

    for attempt in range(max_retries):
        try:
            return fn()  # success → return immediately, skip remaining retries
        except Exception as e:
            last_exception = e

            # If this was the final attempt, don't sleep — just re-raise.
            if attempt == max_retries - 1:
                raise

            # Log which attempt failed and how long we'll wait before the next one.
            logger.warning(
                "Attempt %s/%s failed (%s). Retrying in %.1fs: %s",
                attempt + 1,
                max_retries,
                fn.__name__ if hasattr(fn, "__name__") else "call",
                delay,
                e,
            )

            time.sleep(delay)  # pause before retrying
            delay *= 2  # Exponential backoff: 1s → 2s → 4s → 8s → ...

    # This line is technically unreachable (the `raise` inside the loop handles
    # the final attempt), but it satisfies the type checker and acts as a safety net.
    raise last_exception
