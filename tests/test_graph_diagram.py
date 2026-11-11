from src.agents import graph as agent_graph


def test_graph_mermaid_exports_source_and_image(tmp_path, monkeypatch):
    class FakeDrawableGraph:
        def draw_mermaid(self):
            return "graph TD; supervisor --> finalize;"

        def draw_mermaid_png(self):
            return b"png-bytes"

    class FakeAgent:
        def get_graph(self):
            return FakeDrawableGraph()

    monkeypatch.setattr(
        agent_graph,
        "agent",
        FakeAgent(),
    )
    source_path = tmp_path / "agent.mmd"
    image_path = tmp_path / "agent.png"

    assert agent_graph.save_graph_mermaid_source(source_path) == source_path
    assert "supervisor --> finalize" in source_path.read_text(encoding="utf-8")
    assert agent_graph.save_graph_mermaid_image(image_path) == image_path
    assert image_path.read_bytes() == b"png-bytes"
