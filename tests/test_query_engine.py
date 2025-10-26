import pytest

from src.query_engine import EnhancedQueryEngine


class Dummy:
    pass


class DummyEmbedder:
    def __init__(self):
        self.model_name = "voyage-3"

    def embed(self, text, input_type=None):
        return [0.0] * 1024


class DummyReranker:
    def __init__(self):
        self.model_name = "rerank-2.5"

    def rerank(self, query, documents, top_k=None):
        # simple rerank: return top_k first documents with indices 0..k-1
        results = []
        for i, _ in enumerate(documents[: top_k or len(documents)]):
            results.append({"index": i, "relevance_score": 1.0 - i * 0.01})
        return type("R", (), {"results": results})()


class DummySynth:
    async def synthesize_answer(self, question, context=""):
        return "dummy answer"


@pytest.mark.asyncio
async def test_prepare_documents_with_none_contexts():
    # Create engine with dummy components
    engine = EnhancedQueryEngine(
        mongo_handler=Dummy(),
        text_embedder=DummyEmbedder(),
        multimodal_embedder=Dummy(),
        reranker=DummyReranker(),
        synthesis_llm=DummySynth(),
        use_query_cache=False,
    )

    # contexts list with some None entries
    contexts = [
        {"ku_type": "text", "raw_content": {"text": "A"}},
        None,
        {"ku_type": "text", "raw_content": {"text": "B"}},
        None,
    ]

    documents, mapping = engine._prepare_documents_for_reranking(contexts)

    # Only two non-None contexts => two documents
    assert len(documents) == 2
    # mapping keys should be 0 and 1 (document indices)
    assert list(sorted(mapping.keys())) == [0, 1]
    # mapping values should be the original context dicts (not None)
    assert mapping[0]["raw_content"]["text"] == "A"
    assert mapping[1]["raw_content"]["text"] == "B"
