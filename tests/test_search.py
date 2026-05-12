from uuid import uuid4

from document_agent.search.engine import DocumentSearchEngine
from document_agent.search.engine import reciprocal_rank_fusion
from document_agent.search.models import SearchHit
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
    assert "<mark>gamma</mark>" in snippet
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


def _hit(
    *,
    filename: str,
    chunk_index: int,
    score: float,
    keyword_score: float = 0.0,
    semantic_score: float = 0.0,
) -> SearchHit:
    return SearchHit(
        library_item_id=uuid4(),
        job_id=uuid4(),
        asset_id=uuid4(),
        filename=filename,
        detected_type="pdf",
        score=score,
        keyword_score=keyword_score,
        semantic_score=semantic_score,
        chunk_index=chunk_index,
        snippet=filename,
        markdown_url="http://test/markdown",
        preview_url="http://test/preview",
        processed_at=None,
    )


def test_hybrid_ranking_keeps_keyword_hits_before_semantic_only_expansion() -> None:
    keyword_hit = _hit(filename="literal-match.pdf", chunk_index=0, score=0.2, keyword_score=0.2)
    semantic_only_hit = _hit(filename="semantic-neighbor.pdf", chunk_index=0, score=0.9, semantic_score=0.9)

    ranked = reciprocal_rank_fusion(
        keyword_hits=[keyword_hit],
        semantic_hits=[semantic_only_hit],
        alpha=0.9,
    )

    assert ranked[0].filename == "literal-match.pdf"
    assert ranked[1].filename == "semantic-neighbor.pdf"
