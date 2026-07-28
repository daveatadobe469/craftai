from rag import embedder


def test_encode_falls_back_when_sentence_transformers_is_unavailable(monkeypatch):
    monkeypatch.setattr(embedder, "_model", None)

    def raise_model_error(*args, **kwargs):
        raise RuntimeError("offline model download failed")

    monkeypatch.setattr(embedder, "SentenceTransformer", raise_model_error)

    vectors = embedder.encode(["hello world", "marketing content"])

    assert len(vectors) == 2
    assert all(len(vector) == 384 for vector in vectors)
    assert all(abs(sum(vector)) > 0 for vector in vectors)
