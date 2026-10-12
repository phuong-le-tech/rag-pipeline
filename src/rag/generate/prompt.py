"""Grounded prompt: numbered context blocks plus strict answering rules.

Numbering the blocks gives the model a cheap, checkable way to cite: ``[2]`` must refer to
block 2, so citations can be validated mechanically after generation.
"""

from collections.abc import Sequence

from rag.retrieval.search import Hit

REFUSAL = "I don't know based on the provided documentation."

SYSTEM_PROMPT = f"""You answer questions about Kubernetes using ONLY the numbered context \
passages provided by the user.

Rules:
- Use only facts stated in the context. Do not use outside knowledge.
- After every sentence that states a fact, cite its passage number(s) like [n] or [n, m].
- Cite only passage numbers that exist in the context.
- If the context does not contain the answer, reply with exactly: "{REFUSAL}"
- Be concise: a short answer, then commands or YAML if the context provides them."""


def format_numbered(pairs: Sequence[tuple[int, Hit]]) -> str:
    """Render ``(number, hit)`` pairs; the numbers are used as given, not renumbered."""
    blocks = []
    for number, hit in pairs:
        heading = hit.heading or hit.title
        blocks.append(f"[{number}] {heading}\nSource: {hit.url}\n{hit.content}")
    return "\n\n---\n\n".join(blocks)


def format_context(hits: Sequence[Hit]) -> str:
    return format_numbered(list(enumerate(hits, start=1)))


def build_prompt(question: str, hits: Sequence[Hit]) -> tuple[str, str]:
    """Return ``(system, user)`` messages for a grounded, cited answer."""
    if not hits:
        raise ValueError("cannot build a grounded prompt without context passages")
    user = f"Context passages:\n\n{format_context(hits)}\n\n---\n\nQuestion: {question}"
    return SYSTEM_PROMPT, user
