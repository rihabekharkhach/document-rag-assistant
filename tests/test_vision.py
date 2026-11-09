import pytest

from app.rag import vision


@pytest.fixture
def figure(tmp_path):
    image = tmp_path / "fig.png"
    image.write_bytes(b"fake image")
    return {
        "type": "figure", "source": "a.pdf", "page": 4, "image_path": str(image),
        "text": "[Figure, page 4] Figure 3: confusion matrices\nText detected in the figure:\n0.98",
    }


def text_item():
    return {"type": "text", "source": "a.pdf", "page": 4, "text": "some paragraph"}


def state(question, items, candidates=None, found=True):
    return {"query": question, "standalone_query": question, "items": items,
            "candidates": items if candidates is None else candidates, "found": found}


class FakeModel:
    def __init__(self, reply=None, error=None):
        self.reply, self.error = reply, error

    def invoke(self, messages):
        if self.error:
            raise self.error
        return type("Reply", (), {"content": self.reply})()


def test_visual_question_with_a_figure_triggers_vision(figure):
    assert vision.should_use_vision(state("Que montre la figure 3 ?", [text_item(), figure]))


def test_plain_text_question_does_not_trigger_vision(figure):
    assert not vision.should_use_vision(state("Quel est l'objectif ?", [text_item(), figure]))


def test_no_figure_no_vision():
    assert not vision.should_use_vision(state("Que montre la figure 3 ?", [text_item()]))


def test_vision_can_be_switched_off(figure, monkeypatch):
    monkeypatch.setattr(vision, "VISION_ENABLED", False)
    assert not vision.should_use_vision(state("Que montre la figure 3 ?", [figure]))


def test_named_figure_is_picked_from_the_candidates(figure, tmp_path):
    other_image = tmp_path / "other.png"
    other_image.write_bytes(b"x")
    other = {**figure, "image_path": str(other_image), "text": "[Figure, page 9] Figure 9 Test confusion matrix"}
    assert vision.pick_figures(state("Dans la Figure 9, quelle valeur ?", [text_item()],
                                     candidates=[figure, other])) == [other]


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "CACHE_FILE", tmp_path / "cache.json")


def test_ask_vision_returns_the_answer(figure, isolated_cache, monkeypatch):
    monkeypatch.setattr(vision, "_model", lambda: FakeModel("0,98"))
    assert vision.ask_vision("q", [figure]) == ("ok", "0,98")


def test_not_found_is_reported_and_then_served_from_the_cache(figure, isolated_cache, monkeypatch):
    monkeypatch.setattr(vision, "_model", lambda: FakeModel("NOT_FOUND"))
    assert vision.ask_vision("q", [figure]) == ("not_found", None)
    monkeypatch.setattr(vision, "_model", lambda: FakeModel(error=RuntimeError("must not be called")))
    assert vision.ask_vision("q", [figure]) == ("not_found", None)


def test_api_failure_is_reported_as_an_error(figure, isolated_cache, monkeypatch):
    monkeypatch.setattr(vision, "_model", lambda: FakeModel(error=RuntimeError("boom")))
    assert vision.ask_vision("q", [figure]) == ("error", None)