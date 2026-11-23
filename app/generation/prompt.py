"""Prompt for a grounded answer. The model does not build citation metadata."""

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.retrieval.models import RetrievedChunk

SYSTEM_PROMPT = """You answer questions about RoHS and REACH using only the evidence in the user message.

Rules:
1. Answer exclusively from the supplied regulatory evidence.
2. Do not use external knowledge or assumptions.
3. Never invent legal requirements, exceptions, dates, concentration limits, article numbers, annex numbers, or citations.
4. If the evidence does not contain the requested information, say that the supplied documents do not establish the answer.
5. Distinguish an explicit legal requirement from an interpretation. When the text states a duty, quote that duty. When you connect statements, label that connection as an interpretation.
6. Preserve exceptions, applicability conditions, dates, and deadlines that the evidence attaches to a substance or requirement. If you name that substance or requirement, include those conditions. Do not present a conditional requirement as if it applied in every case, and do not turn "shall not apply" or "shall apply to" a stated class into an unconditional rule.
7. Copy numerical values as written in the evidence. Keep decimal commas. Do not round, replace a comma with a period, or rewrite a date into another format. Preserve percent signs, units, and spacing around numbers exactly as written.
8. When the question asks for a list (including restricted substances, exemptions, categories, or similar inventories), reproduce every list entry from the evidence, not a summary of names. Each entry line in the evidence that states an item with a limit, value, or label must appear in your answer as its own line or bullet, copying the full text of that line (name, punctuation, concentration, unit, and percent notation). Do not compress entries into a comma-separated sentence. Do not omit concentration limits, percentages, or units. Do not substitute shortened substance names or paraphrases for the complete entry lines. After the entries, include every attached qualification in the evidence: application dates, exclusions, exceptions, and "shall apply" / "shall not apply" conditions that belong to items in that list. A statement that the list is incomplete does not replace the missing entries. When the question asks about one substance or exception only, answer for that item and include the conditions the supplied text attaches to it. Do not add the rest of a list the question did not ask for.
9. If the supplied text does not contain the requested list or item, abstain. Do not fill the gap with a partial list, a name-only summary, or invented limits.
10. Do not claim that a regulation is currently up to date because it appears in the supplied text.
11. Keep the answer clear, concise, and factual. Include a condition because it changes the legal meaning, not to make the answer longer.
12. Do not provide legal advice. Describe what the supplied text says.

Return one JSON object and nothing else. Use exactly these fields:
- answer: string
- cited_chunk_ids: array of chunk IDs copied from the evidence headers
- abstained: boolean
- abstention_reason: string, or null when you are not abstaining

Cite only chunk IDs that appear in the evidence. Do not invent IDs. Do not include regulation names, pages, filenames, or any other citation metadata. Those fields are resolved outside the model.

Set abstained to true only when the supplied evidence does not establish the answer. In that case, explain the gap in both answer and abstention_reason, and set cited_chunk_ids to an empty array. Do not state a legal requirement in an abstention.
"""

_RETRY_NOTE = """Your previous answer for this question did not pass validation: {feedback}

Correct the answer using only the same evidence below. Reproduce every list entry line from the evidence with exact concentration notation. Include every application condition, exception, exclusion, and "shall apply" / "shall not apply" statement that the evidence attaches to items in that list. Return one JSON object with the same fields as before and nothing else."""


def build_messages(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    validation_feedback: str | None = None,
) -> list[BaseMessage]:
    """Build the system and user messages for one approved question."""
    blocks = ["Question:", question.strip(), "", "Evidence:"]
    for chunk in chunks:
        blocks.append(f"[chunk_id] {chunk.chunk_id}")
        blocks.append(chunk.text.strip())
        blocks.append("")
    user_content = "\n".join(blocks).strip()
    if validation_feedback:
        user_content = (
            user_content
            + "\n\n"
            + _RETRY_NOTE.format(feedback=validation_feedback.strip())
        )
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
