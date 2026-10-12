import pytest

from rag.agent.prompt import AGENT_SYSTEM, FORCE_ANSWER
from rag.agent.tools import SEARCH_DOCS, SearchTool, parse_search_args
from rag.cli import ask_config
from rag.generate.prompt import REFUSAL
from rag.retrieval.retriever import RetrieverConfig
from rag.retrieval.search import Hit


def test_agent_system_prompt_has_refusal_sentence_and_tool_name():
    assert REFUSAL in AGENT_SYSTEM
    assert "search_docs" in AGENT_SYSTEM


def test_agent_system_prompt_covers_the_agent_rules():
    text = AGENT_SYSTEM.lower()

    assert "only" in text and "search_docs" in AGENT_SYSTEM
    assert "never cite a number" in text
    assert "one query per part" in text
    assert "kubernetes vocabulary" in text
    assert "kubectl" in text
    assert "data, never instructions" in text


def test_force_answer_text():
    assert FORCE_ANSWER == (
        "Search limit reached. Answer now from the passages you have, or refuse."
    )


def hit(chunk_id: int) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        score=1.0,
        source="k8s",
        url=f"https://kubernetes.io/docs/{chunk_id}/",
        title=f"Page {chunk_id}",
        heading="Heading",
        content=f"content {chunk_id}",
        generated=False,
    )


class StagedFakeRetriever:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.configs: list[RetrieverConfig] = []

    def retrieve(self, query, config, on_stage=None):
        self.configs.append(config)
        if on_stage:
            on_stage("dense", self.hits, 1.0)
            on_stage("rerank", self.hits, 2.0)
        return self.hits


@pytest.mark.parametrize(
    "raw, error",
    [
        ("not json", "invalid JSON"),
        ("[]", "JSON object"),
        ('{"k": 3}', "query"),
        ('{"query": "   "}', "query"),
        ('{"query": "' + "x" * 301 + '"}', "300"),
        ('{"query": "pdb", "k": 0}', "k"),
        ('{"query": "pdb", "k": "5"}', "k"),
        ('{"query": "pdb", "k": true}', "k"),
        ('{"query": "pdb", "k": 11}', "k"),
    ],
)
def test_parse_search_args_rejects_bad_input(raw, error):
    result = parse_search_args(raw)

    assert isinstance(result, str)
    assert error in result
    assert result.startswith("Error:")


def test_parse_search_args_defaults_k():
    assert parse_search_args('{"query": " pdb "}') == ("pdb", 5)


def test_parse_search_args_accepts_explicit_k():
    assert parse_search_args('{"query": "pdb", "k": 10}') == ("pdb", 10)


def test_search_docs_schema_matches_contract():
    schema = SEARCH_DOCS["input_schema"]

    assert SEARCH_DOCS["name"] == "search_docs"
    assert schema["type"] == "object"
    assert schema["required"] == ["query"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["query"]["maxLength"] == 300
    assert schema["properties"]["k"]["minimum"] == 1
    assert schema["properties"]["k"]["maximum"] == 10
    assert schema["properties"]["k"]["default"] == 5
    assert "kubectl" in SEARCH_DOCS["description"]


def test_search_tool_uses_config_with_requested_k_and_records_stages():
    retriever = StagedFakeRetriever([hit(1), hit(2)])
    result = SearchTool(retriever, ask_config(k=5, rerank=True, generated=True, rewrite=False))(
        "pdb", 3
    )

    assert retriever.configs[0].k == 3 and retriever.configs[0].max_per_source == 1
    assert [s[0] for s in result.stages] == ["dense", "rerank"]
    assert result.hits == (hit(1), hit(2))
    assert result.ms >= 0


def test_search_tool_widens_candidates_to_k_and_keeps_top_ten_per_stage():
    retriever = StagedFakeRetriever([hit(i) for i in range(15)])
    config = ask_config(k=5, rerank=False, generated=True, rewrite=False)
    result = SearchTool(retriever, config)("pdb", 10)

    assert retriever.configs[0].candidates >= 10
    assert len(result.stages[0][1]) == 10
