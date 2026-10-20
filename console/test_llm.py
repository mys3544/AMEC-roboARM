"""The language model's side of voice commands (llm.py, voice.understand), without a model: its
calls turned into commands, the schema they are held to, and what happens when it is not there."""

import llm
import voice


class Pretend:
    """A Brain that answers what it is told to, or fails."""

    def __init__(self, call=None, fails=False):
        self.call, self.fails, self.asked = call, fails, []

    def ask(self, text):
        self.asked.append(text)
        if self.fails:
            raise llm.Unavailable("the language model did not answer", "python console/llm.py serve")
        return llm.intent(llm.full(self.call), text), 42


def test_a_stop_never_waits_for_the_model():
    brain = Pretend({"action": "clean_up"})
    command, how = voice.understand("prosím zastav robota a uprac", brain)
    assert command == {"action": "stop", "robot": "all"} and how == {"by": "grammar"} and brain.asked == []


def test_the_models_call_is_the_command():
    brain = Pretend({"action": "pick_named", "robot": "m3", "name": "fish"})
    command, how = voice.understand("podaj mi rybu", brain)
    assert command == {"action": "pick_named", "robot": "m3", "name": "fish"}
    assert how == {"by": "llm", "llm_ms": 42}


def test_without_the_model_the_grammar_answers():
    command, how = voice.understand("X3, clear the table", Pretend(fails=True))
    assert command == {"action": "clean_up", "robot": "x3"} and how["by"] == "grammar" and "llm_error" in how
    assert voice.understand("X3, clear the table") == ({"action": "clean_up", "robot": "x3"}, {"by": "grammar"})


def test_calls_become_the_pages_commands():
    full = lambda **call: llm.intent(llm.full(call), "said")
    assert full(action="pick", robot="x3", kind="block", size_mm=40) == {
        "action": "pick", "robot": "x3", "kind": "block", "colour": None, "size_mm": None}   # a block has no size
    assert full(action="pick", kind="cube", colour="red", size_mm=30) == {
        "action": "pick", "robot": "m3", "kind": "cube", "colour": "red", "size_mm": 30}
    assert full(action="clean_up", robot="both", kind="cube") == {"action": "clean_up", "robot": "both", "kind": "cube"}
    assert full(action="sort", robot="m3") == {"action": "sort", "robot": "m3", "by": "kind"}
    assert full(action="find", robot="x3") == {"action": "find", "robot": "m3"}   # only the M3 was taught
    assert full(action="pick_named", robot="m3")["action"] == "unknown"   # no such name
    assert full(action="home", robot="x3") == {"action": "home", "robot": "x3"}
    assert full(action="stop", robot="x3") == {"action": "stop", "robot": "all"}
    assert full(action="unknown") == {"action": "unknown", "text": "said"}


def test_the_schema_holds_every_field_and_only_taught_names():
    s = llm.schema(["fish", "news"])
    assert s["required"] == llm.FIELDS and set(s["properties"]) == set(llm.FIELDS)
    assert s["properties"]["name"]["enum"] == ["fish", "news", None]
    assert llm.schema([])["properties"]["name"]["enum"] == [None]
    said = llm.messages("zober kocku", ["fish"])
    assert said[0]["role"] == "system" and "fish" in said[0]["content"] and said[-1]["content"] == "zober kocku"


def test_a_brain_without_its_files_says_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "SERVER", tmp_path / "bin" / "llama-server.exe")
    brain = llm.Brain()
    assert "no llama-server" in brain.missing()[0] and brain.status()["ready"] is False
    brain.start()   # nothing to start: no process
    assert brain.proc is None
