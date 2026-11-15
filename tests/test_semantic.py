from datetime import datetime
from types import SimpleNamespace

from app.semantic import rank_agents_for_query


class FakeEmbeddingModel:
    def __init__(self):
        self.query = None
        self.documents = []

    def query_embed(self, query):
        self.query = query
        return iter([[1.0, 0.0]])

    def passage_embed(self, documents):
        self.documents = list(documents)
        return iter([[0.8, 0.6], [0.2, 0.98]])


def make_agent(agent_id, company_name, description, posts=None):
    profile = SimpleNamespace(
        company_name=company_name,
        industry="Agriculture",
        description=description,
        products="",
        services="",
        requirements="",
        interests="",
        location="",
    )
    return SimpleNamespace(
        id=agent_id,
        user_id=agent_id,
        user=SimpleNamespace(profile=profile),
        posts=posts or [],
        name=f"Agent {agent_id}",
        industry="Agriculture",
        summary="",
        business_goals="",
        interests="",
        location="",
    )


def test_rank_agents_uses_local_embedding_similarity(monkeypatch):
    from app import semantic

    model = FakeEmbeddingModel()
    monkeypatch.setattr(semantic, "get_embedding_model", lambda: model)
    recent_post = SimpleNamespace(
        title="Regional harvest partnership",
        description="Seeking distribution support",
        requirements="",
        category="Logistics",
        created_at=datetime(2026, 1, 1),
    )

    results = rank_agents_for_query(
        "sustainable cultivation of edible crops",
        [
            make_agent(1, "Organic growers", "Fresh produce and farm services", [recent_post]),
            make_agent(2, "Metal works", "Industrial components and fabrication"),
        ],
    )

    assert results[0]["user_id"] == 1
    assert results[0]["score"] == 0.8
    assert results[0]["reasons"] == ["Embedding similarity found related business context."]
    assert model.query == "sustainable cultivation of edible crops"
    assert model.documents[0].startswith("Agent 1 Agriculture")
    assert "Regional harvest partnership" in model.documents[0]


def test_rank_agents_handles_missing_user_profile(monkeypatch):
    from app import semantic

    model = FakeEmbeddingModel()
    monkeypatch.setattr(semantic, "get_embedding_model", lambda: model)

    result = rank_agents_for_query(
        "farm logistics",
        [
            SimpleNamespace(
                id=1,
                user_id=1,
                user=None,
                posts=[],
                name="Independent Farm Scout",
                industry="Agriculture",
                summary="",
                business_goals="",
                interests="",
                location="Berlin",
            )
        ],
    )

    assert len(result) == 1
    assert result[0]["name"] == "Independent Farm Scout"
    assert result[0]["location"] == "Berlin"
    assert any("farm" in reason.lower() or "keyword" in reason.lower() for reason in result[0]["reasons"])
