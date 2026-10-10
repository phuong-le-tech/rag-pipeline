"""Query rewriting: restate a user's question in the documentation's vocabulary before search.

Users describe situations ("go back to what was running before"); docs use names ("kubectl
rollout undo", "revision"). Dense search bridges some of that gap, keyword search almost none.
The rewrite is used for *searching only*. The reranker still scores passages against the
original question, because that is what the user actually asked.
"""

from rag.generate.generator import Generator

REWRITE_SYSTEM = """Rewrite the user's question as a short search query for the official \
Kubernetes documentation. Use the terminology the documentation would use: resource kinds, \
kubectl commands and flags, field names. Keep every concrete detail from the question.
Output only the query on one line, with no explanation and no quotes."""


class QueryRewriter:
    """LLM rewrite with an in-memory cache (evals ask the same question in many configs)."""

    def __init__(self, generator: Generator) -> None:
        self.generator = generator
        self._cache: dict[str, str] = {}

    def rewrite(self, question: str) -> str:
        if question not in self._cache:
            raw = self.generator.generate(REWRITE_SYSTEM, question)
            first_line = raw.strip().splitlines()[0] if raw.strip() else ""
            cleaned = first_line.strip().strip("\"'`").strip()
            self._cache[question] = cleaned or question
        return self._cache[question]
