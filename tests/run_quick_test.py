import asyncio
import sys
import pathlib

# Ensure repo root is on sys.path so `src` imports work when running tests directly
repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

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
        results = []
        for i, _ in enumerate(documents[: top_k or len(documents)]):
            results.append({"index": i, "relevance_score": 1.0 - i * 0.01})
        return type("R", (), {"results": results})()


class DummySynth:
    async def synthesize_answer(self, question, context=""):
        return "dummy answer"


async def main():
    engine = EnhancedQueryEngine(
        mongo_handler=Dummy(),
        text_embedder=DummyEmbedder(),
        multimodal_embedder=Dummy(),
        reranker=DummyReranker(),
        synthesis_llm=DummySynth(),
        use_query_cache=False,
    )

    contexts = [
        {"ku_type": "text", "raw_content": {"text": "A"}},
        None,
        {"ku_type": "text", "raw_content": {"text": "B"}},
        None,
    ]

    documents, mapping = engine._prepare_documents_for_reranking(contexts)

    ok = True
    if len(documents) != 2:
        print("FAIL: documents length", len(documents))
        ok = False
    if sorted(mapping.keys()) != [0, 1]:
        print("FAIL: mapping keys", mapping.keys())
        ok = False
    if mapping[0]["raw_content"]["text"] != "A":
        print("FAIL: mapping[0] wrong")
        ok = False
    if mapping[1]["raw_content"]["text"] != "B":
        print("FAIL: mapping[1] wrong")
        ok = False

    if ok:
        print("OK: quick test passed")
        return 0
    else:
        return 2


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
