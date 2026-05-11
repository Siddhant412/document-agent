from document_agent.search.engine import DocumentSearchEngine
from document_agent.search.models import SearchQuery
from document_agent.search.text import clean_headline, normalize_query, plain_snippet


class FakeRepository:
    class settings:
        hybrid_search_alpha = 0.5
        search_semantic_enabled = False

    def search_chunks_keyword(self, **kwargs):
        assert kwargs["query"] == "campus energy"
        return []


def test_normalize_query_collapses_whitespace() -> None:
    assert normalize_query("  campus\n  energy\tpilot  ") == "campus energy pilot"


def test_plain_snippet_centers_match() -> None:
    snippet = plain_snippet("alpha beta gamma delta epsilon", "gamma", radius=6)
    assert "gamma" in snippet
    assert snippet.startswith("...")


def test_clean_headline_falls_back_for_empty_headline() -> None:
    snippet = clean_headline("", "alpha beta gamma", "beta")
    assert "beta" in snippet


def test_search_engine_normalizes_query_before_repository_call() -> None:
    response = DocumentSearchEngine(repository=FakeRepository()).search(
        SearchQuery(query="  campus\nenergy  ")
    )
    assert response.query == "campus energy"
    assert response.total == 0
