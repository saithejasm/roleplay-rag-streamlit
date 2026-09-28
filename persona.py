"""Turns a scenario + retrieved chunks + conversation history into the system
prompt for in-character roleplay, and into the message list for the separate
end-of-session feedback call."""

import re

from config import FEEDBACK_TRAILER
from scenarios import DEFAULT_MOOD, MOOD_INSTRUCTIONS

NO_DOCUMENT_SENTINEL = "(No background document was provided. Improvise realistic, plausible details for this scenario.)"

SHORT_REPLY_WORD_THRESHOLD = 6

# Generic chat-template speaker labels some models fall back to (instead of the
# persona's actual role names) when they try to write out both sides of the
# conversation -- caught in addition to the scenario's own ai_role/user_role.
GENERIC_SPEAKER_LABELS = ["user", "assistant", "human", "ai", "bot", "system"]


def _speaker_labels(scenario: dict) -> list[str]:
    return list(dict.fromkeys([scenario["ai_role"], scenario["user_role"], *GENERIC_SPEAKER_LABELS]))


def build_agenda_block(scenario: dict) -> str:
    """Bullet list of what the authority figure needs to find out this session,
    injected into the system prompt so the AI directs the conversation toward
    concrete goals instead of asking generic, unfocused questions."""
    return "\n".join(f"  - {goal}" for goal in scenario["information_goals"])


def build_scene_block(location: str, mood: str) -> str:
    """Describe where the conversation is taking place and what tone to hold it in."""
    mood_instruction = MOOD_INSTRUCTIONS.get(mood, MOOD_INSTRUCTIONS[DEFAULT_MOOD])
    return (
        f"Setting: this conversation is taking place at/via: {location}.\n"
        f"Mood/tone to maintain throughout: {mood} -- {mood_instruction}"
    )


def build_retrieval_query(scenario: dict, messages: list[dict]) -> str:
    """Decide what to search the uploaded document for on this turn.

    Turn zero (no user messages yet) grounds on the scenario's fixed opening
    query. Later turns use the user's last message, so the AI's follow-up
    questions probe deeper into what the user actually said -- except right
    after a throwaway first reply (e.g. "hi", "ready when you are"), where
    it's blended with the opening query so the AI's first real question isn't
    starved of a useful embedding.
    """
    user_messages = [m for m in messages if m["role"] == "user"]
    if not user_messages:
        return scenario["opening_retrieval_query"]

    last_user_message = user_messages[-1]["content"]
    if len(user_messages) == 1 and len(last_user_message.split()) < SHORT_REPLY_WORD_THRESHOLD:
        return f"{last_user_message} {scenario['opening_retrieval_query']}"
    return last_user_message


def build_context_block(chunks: list[dict]) -> str:
    """Join retrieved chunk text into a plain narrative block. No inline
    citations -- the AI should know this material, not cite documents mid-roleplay."""
    if not chunks:
        return NO_DOCUMENT_SENTINEL
    return "\n\n".join(chunk["text"] for chunk in chunks)


def build_system_prompt(scenario: dict, context_block: str, scene_block: str) -> str:
    return scenario["system_prompt_template"].format(
        context_block=context_block,
        scene_block=scene_block,
        ai_role=scenario["ai_role"],
        user_role=scenario["user_role"],
        agenda_block=build_agenda_block(scenario),
    )


def build_stop_sequences(scenario: dict) -> list[str]:
    """Stop sequences that cut generation off if the model starts writing the
    other side of the conversation (or labels its own turn) instead of stopping
    after its single line -- a backstop for models that don't fully follow the
    system prompt's single-turn instruction. Capped at 4 entries (OpenAI's
    stop-sequence limit), prioritizing the generic "User:"/"Assistant:"
    chat-template labels models fall back to most often, ahead of the
    scenario's own persona labels."""
    return [
        "\nUser:",
        "\nAssistant:",
        f"\n{scenario['user_role']}:",
        f"\n{scenario['ai_role']}:",
    ]


def trim_to_single_turn(answer: str, scenario: dict) -> str:
    """Defensive cleanup for models that still don't follow the single-turn
    instruction: strips a leading self-label (e.g. "Doctor: ..." or a markdown
    heading like "# User:") and truncates at the first later line that looks
    like a speaker label -- either the scenario's own persona names or a
    generic chat-template label ("User:", "Assistant:", etc.) -- so a model
    that writes out both sides of the conversation only contributes its own
    first turn."""
    label_pattern = "|".join(re.escape(label) for label in _speaker_labels(scenario))
    # Tolerates optional markdown heading hashes and/or bold asterisks around
    # the label, e.g. "Doctor:", "**Doctor:**", "# User:", "### **Assistant:**".
    label_line = rf"^\s*#{{0,6}}\s*\**\s*({label_pattern})\s*\**\s*:\s*"

    answer = re.sub(label_line, "", answer, count=1, flags=re.IGNORECASE)

    match = re.search(rf"(?m){label_line}", answer, flags=re.IGNORECASE)
    if match:
        answer = answer[: match.start()]

    return answer.strip()


def build_feedback_messages(scenario: dict, messages: list[dict]) -> list[dict]:
    """Build the message list for the end-of-session evaluation call: the
    scenario's feedback rubric as the system prompt, the full transcript
    (role/content only), and a trailing instruction to produce the evaluation."""
    transcript = [{"role": m["role"], "content": m["content"]} for m in messages]
    return [
        {"role": "system", "content": scenario["feedback_rubric_template"]},
        *transcript,
        {"role": "user", "content": FEEDBACK_TRAILER},
    ]
