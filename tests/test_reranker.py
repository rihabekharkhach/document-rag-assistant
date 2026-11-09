import pytest

from app.rag import reranker

CHUNKS = ["a", "b", "c", "d", "e", "f"]


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code = status
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise reranker.requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def fast_and_keyed(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    monkeypatch.setenv("JINA_API_KEY", "test-key")


def test_returns_indices_ordered_by_relevance(monkeypatch):
    payload = {"results": [{"index": 4, "relevance_score": 0.2}, {"index": 2, "relevance_score": 0.9}]}
    monkeypatch.setattr(reranker.requests, "post", lambda *a, **k: FakeResponse(payload=payload))
    assert reranker.rerank_indices("q", CHUNKS, top_n=2) == [2, 4]
    assert reranker.rerank("q", CHUNKS, top_n=2) == ["c", "e"]


def test_falls_back_to_vector_order_when_the_api_keeps_failing(monkeypatch):
    monkeypatch.setattr(reranker.requests, "post", lambda *a, **k: FakeResponse(status=429))
    assert reranker.rerank_indices("q", CHUNKS, top_n=3) == [0, 1, 2]


def test_without_api_key_uses_vector_order(monkeypatch):
    monkeypatch.delenv("JINA_API_KEY")
    assert reranker.rerank_indices("q", CHUNKS, top_n=2) == [0, 1]


def test_few_chunks_are_returned_as_they_are():
    assert reranker.rerank_indices("q", ["a", "b"], top_n=4) == [0, 1]