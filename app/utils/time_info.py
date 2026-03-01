"""
TIME INFORMATION UTILITY
========================

Returns a short, readable string with the current date and time. This is
injected into the system prompt so the LLM can answer "what day is it?"
and similar questions. Called by both GroqService and RealtimeGroqService.

─────────────────────────────────────────────────────────────────────────────
WHY INJECT TIME INTO THE SYSTEM PROMPT?
─────────────────────────────────────────────────────────────────────────────
LLMs have NO built-in awareness of the current date or time. Their training
data has a cutoff, and they don't have access to a system clock. So if a user
asks "What day is today?" or "Is it morning or evening?", the LLM would
either hallucinate an answer or say "I don't know."

By including a block of text like:
    Current Real-time Information:
    Day: Saturday
    Date: 21
    Month: February
    Year: 2026
    Time: 14 hours, 30 minutes, 05 seconds

...in the system prompt, we give the LLM factual time data it can reference.
This is a form of **prompt augmentation** — enriching the prompt with external
data the model wouldn't otherwise have.

─────────────────────────────────────────────────────────────────────────────
FORMATTING CHOICES
─────────────────────────────────────────────────────────────────────────────
  - Each piece of information is on its own labeled line (Day:, Date:, etc.)
    so the LLM can easily parse and reference individual components.
  - The day name is fully spelled out ("Saturday", not "Sat") for natural
    language responses.
  - Time uses 24-hour format with words ("14 hours, 30 minutes") rather than
    "14:30:05" to help the LLM produce more natural-sounding answers like
    "It's about 2:30 in the afternoon."
  - The heading "Current Real-time Information:" clearly signals to the LLM
    what this block of text represents.
"""

import datetime


def get_time_information() -> str:
    """
    Return a human-readable, multi-line string with the current date and time.

    This string is appended to the LLM's system prompt so it can answer
    time-related questions accurately.

    Uses Python's datetime.now() to get the local system time, then formats
    each component with strftime directives:
        %A  → full weekday name   (e.g. "Monday")
        %d  → zero-padded day     (e.g. "05")
        %B  → full month name     (e.g. "February")
        %Y  → four-digit year     (e.g. "2026")
        %H  → 24-hour hour        (e.g. "14")
        %M  → minute              (e.g. "30")
        %S  → second              (e.g. "05")

    Returns:
        A formatted string like:
            Current Real-time Information:
            Day: Saturday
            Date: 21
            Month: February
            Year: 2026
            Time: 14 hours, 30 minutes, 05 seconds
    """
    now = datetime.datetime.now()
    return (
        f"Current Real-time Information:\n"
        f"Day: {now.strftime('%A')}\n"      # e.g. Monday
        f"Date: {now.strftime('%d')}\n"      # e.g. 05
        f"Month: {now.strftime('%B')}\n"     # e.g. February
        f"Year: {now.strftime('%Y')}\n"      # e.g. 2026
        f"Time: {now.strftime('%H')} hours, {now.strftime('%M')} minutes, {now.strftime('%S')} seconds\n"
    )
