"""console/voice.py: the grammar (sentences as the recogniser writes them, in English, Slovak
and Czech), the WAV reader, and /api/voice through serve.py with a pretend recogniser -- no model
needed. With the model fetched, one more test hears the model's own English sample.

    python -m pytest console
"""

import io
import json
import pathlib
import sys
import urllib.error
import urllib.request
import wave
from http.server import ThreadingHTTPServer

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import serve
import voice
from test_serve import start


def pick(robot="m3", kind=None, colour=None, size_mm=None):
    return {"action": "pick", "robot": robot, "kind": kind, "colour": colour, "size_mm": size_mm}


STOP = {"action": "stop", "robot": "all"}


@pytest.mark.parametrize("said", [
    "stop", "Stop.", "STOP!", "Stop!!!", "please stop the robot", "Please stop the robot.",
    "Okay, stop, stop.", "halt", "Freeze!", "abort", "Don't stop.", "stopping", "Stop the X3.",
    "Pick up the red cube. No, stop!", "Go home, uh, stop", "clear the table stop",
    "Stoj!", "stoj", "Stůj!", "stuj", "Zastav!", "zastav robota", "Zastavte.", "Stůjte!",
    "Zastaň!", "M3, stop the arm right now",
])
def test_stop_wins_whenever_it_is_said(said):
    assert voice.parse(said) == STOP


@pytest.mark.parametrize("said, command", [
    ("Pick up the red cube.", pick(kind="cube", colour="red")),
    ("pick the cube", pick(kind="cube")),
    ("Pick up a cube", pick(kind="cube")),
    ("pick up the block", pick(kind="block")),
    ("Pick one.", pick()),
    ("Pick up the thirty millimetre cube", pick(kind="cube", size_mm=30)),
    ("pick up the forty millimeter blue cube", pick(kind="cube", colour="blue", size_mm=40)),
    ("Pick up the 40mm blue cube.", pick(kind="cube", colour="blue", size_mm=40)),
    ("Pick up the 40-millimeter cube.", pick(kind="cube", size_mm=40)),
    ("pick up the 30 mm green cube", pick(kind="cube", colour="green", size_mm=30)),
    ("pick up the 3 cm cube", pick(kind="cube", size_mm=30)),
    ("pick up the four centimetre cube", pick(kind="cube", size_mm=40)),
    ("Pick up the thirty cube", pick(kind="cube", size_mm=30)),
    ("grab the small yellow cube", pick(kind="cube", colour="yellow", size_mm=30)),
    ("Uh, can you grab the big red cube, please?", pick(kind="cube", colour="red", size_mm=40)),
    ("take the largest cube", pick(kind="cube", size_mm=40)),
    ("pick 2 cubes", pick(kind="cube")),   # a bare number that is not a size
    ("Hey robot, um, pick up the wooden block", pick(kind="block", colour="wood")),
    ("pick up the 30 mm block", pick(kind="block")),   # there is one block: no size
    ("Pick up the grey cube", pick(kind="cube", colour="grey")),
    ("pick up the gray cube", pick(kind="cube", colour="grey")),
    ("M3, pick up the wooden block.", pick(kind="block", colour="wood")),
    ("X3, pick up the red cube", pick("x3", "cube", "red")),
    ("x3 pick up the 4 cm blue cube", pick("x3", "cube", "blue", 40)),
    ("X-3, pick a cube", pick("x3", "cube")),
    ("X three, pick up the cube", pick("x3", "cube")),
    ("ex three pick the cube", pick("x3", "cube")),
    ("X3 Plus, pick the block", pick("x3", "block")),
    ("Plus, pick the cube", pick("x3", "cube")),
    ("M three pick the cube", pick(kind="cube")),
    ("M3 Pro, pick up the 30 millimetre red cube", pick(kind="cube", colour="red", size_mm=30)),
    ("Pick up the X3's red cube", pick("x3", "cube", "red")),
    # Slovak and Czech
    ("Zober červenú kocku", pick(kind="cube", colour="red")),
    ("Zober zelenú kocku", pick(kind="cube", colour="green")),
    ("Vezmi červenou kostku", pick(kind="cube", colour="red")),
    ("Vezmi malou modrou kostku", pick(kind="cube", colour="blue", size_mm=30)),
    ("Seber žlutou kostku", pick(kind="cube", colour="yellow")),
    ("Zober veľkú žltú kocku", pick(kind="cube", colour="yellow", size_mm=40)),
    ("Vezmi kostku pro X3", pick("x3", "cube")),   # Czech "pro" is "for", not the M3 Pro
    ("Zober drevený kváder", pick(kind="block", colour="wood")),
])
def test_pick_with_what_and_which_robot(said, command):
    assert voice.parse(said) == command


@pytest.mark.parametrize("said, command", [
    ("Clean up.", {"action": "clean_up", "robot": "m3"}),
    ("cleanup", {"action": "clean_up", "robot": "m3"}),
    ("Clear the table.", {"action": "clean_up", "robot": "m3"}),
    ("clear the arena please", {"action": "clean_up", "robot": "m3"}),
    ("X3, clear the table.", {"action": "clean_up", "robot": "x3"}),
    ("Tidy up, M3", {"action": "clean_up", "robot": "m3"}),
    ("pick them all", {"action": "clean_up", "robot": "m3"}),
    ("X3 pick up everything", {"action": "clean_up", "robot": "x3"}),
    ("pick up all the cubes", {"action": "clean_up", "robot": "m3"}),
    ("Uprac stôl", {"action": "clean_up", "robot": "m3"}),
    ("Ukliď stůl", {"action": "clean_up", "robot": "m3"}),
    ("Zober všetky kocky", {"action": "clean_up", "robot": "m3"}),
    ("Go home.", {"action": "home", "robot": "m3"}),
    ("home", {"action": "home", "robot": "m3"}),
    ("X3, go home", {"action": "home", "robot": "x3"}),
    ("M3 Pro, go to the home position", {"action": "home", "robot": "m3"}),
    ("Choď domov", {"action": "home", "robot": "m3"}),
    ("Jdi domů", {"action": "home", "robot": "m3"}),
])
def test_clean_up_and_home(said, command):
    assert voice.parse(said) == command


@pytest.mark.parametrize("said, why", [
    ("", None),
    ("   ", None),
    ("What time is it?", None),
    ("Alles hat ein Ende, nur die Wurst hat zwei.", None),
    ("the red cube", None),   # what, but no command
    ("X3", None),
    ("it's top", None),       # "stop" is looked for in words, not across them
    ("clean up and go home", "more than one command (clean_up and home)"),
    ("pick up the cube and go home", "more than one command (home and pick)"),
    ("X3 and M3, go home", "both robots named"),
    ("pick up the 50 mm cube", "no 50 mm cube in the kit"),
    ("pick up the 4 mm cube", "no 4 mm cube in the kit"),
])
def test_anything_else_is_unknown_and_says_why_when_it_can(said, why):
    command = voice.parse(said)
    assert command["action"] == "unknown" and command["text"] == said
    assert command.get("why") == why


def test_the_words_are_plain_however_they_were_written():
    assert voice.plain_words("Stůj! Pick up the X-3's 30mm cube.") == [
        "stuj", "pick", "up", "the", "x", "3", "s", "30", "mm", "cube"]
    assert voice.plain_words("ŽLUTOU, Veľkú; Červenú…") == ["zlutou", "velku", "cervenu"]


def make_wav(samples, rate=16000, channels=1, width=2):
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(b"".join(int(s).to_bytes(width, "little", signed=True) for s in samples))
    return out.getvalue()


def test_a_wav_is_read_as_floats():
    samples, rate = voice.read_wav(make_wav([0, 16384, -32768, 32767], rate=48000))
    assert rate == 48000 and samples == [0.0, 0.5, -1.0, 32767 / 32768]
    with pytest.raises(ValueError, match="16-bit PCM mono only"):
        voice.read_wav(make_wav([16384, 0, -16384, -16384], channels=2))
    with pytest.raises(ValueError, match="16-bit PCM mono only"):
        voice.read_wav(make_wav([0, 1, 2], width=1))
    with pytest.raises(ValueError, match="not a WAV"):
        voice.read_wav(b"RIFF nonsense")
    with pytest.raises(ValueError, match="not a WAV"):
        voice.read_wav(b"")


class Pretend:
    """A recogniser that hears whatever it is told to."""

    def __init__(self, says="Please pick up the red thirty millimetre cube."):
        self.says = says
        self.got = []

    def status(self):
        return {"ready": True, "load_s": 0}

    def text(self, samples, rate):
        self.got.append((len(samples), rate))
        return self.says


class Missing(Pretend):
    def status(self):
        return {"ready": False, "error": "no speech model", "fix": voice.FETCH}

    def text(self, samples, rate):
        raise voice.Unavailable("no speech model", voice.FETCH)


def post(url, body, kind):
    request = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": kind})
    try:
        with urllib.request.urlopen(request, timeout=10) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_the_console_hears_a_wav_and_answers_with_the_command():
    ears = Pretend()
    port = start(ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler([], recognizer=ears)))
    base = f"http://127.0.0.1:{port}"

    status, reply = post(f"{base}/api/voice", make_wav([0] * 24000), "audio/wav")
    assert status == 200 and ears.got == [(24000, 16000)]
    assert reply["text"] == "Please pick up the red thirty millimetre cube."
    assert reply["intent"] == pick(kind="cube", colour="red", size_mm=30)
    assert reply["audio_s"] == 1.5 and reply["asr_ms"] >= 0

    ears.says = "Stop! Stop!"
    assert post(f"{base}/api/voice", make_wav([0] * 1600), "audio/wav")[1]["intent"] == STOP
    status, reply = post(f"{base}/api/voice", b"X3, clear the table", "text/plain; charset=utf-8")
    assert status == 200 and reply == {"text": "X3, clear the table",
                                       "intent": {"action": "clean_up", "robot": "x3"}}
    assert len(ears.got) == 2   # typed text is not heard, only parsed
    status, reply = post(f"{base}/api/voice", "Stůj!".encode(), "text/plain; charset=utf-8")
    assert reply["intent"] == STOP
    status, reply = post(f"{base}/api/voice", b"not a wav", "audio/wav")
    assert status == 400 and "not a WAV" in reply["error"]
    with urllib.request.urlopen(f"{base}/api/voice", timeout=10) as answer:
        assert json.loads(answer.read())["ready"] is True
    assert post(f"{base}/api/nothing", b"{}", "application/json")[0] == 404


def test_without_a_model_the_console_says_how_to_get_one_and_goes_on():
    port = start(ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler([], recognizer=Missing())))
    base = f"http://127.0.0.1:{port}"
    status, reply = post(f"{base}/api/voice", make_wav([0] * 1600), "audio/wav")
    assert status == 503 and reply == {"error": "no speech model", "fix": voice.FETCH}
    status, reply = post(f"{base}/api/voice", b"go home", "text/plain")   # typed still works
    assert status == 200 and reply["intent"] == {"action": "home", "robot": "m3"}
    with urllib.request.urlopen(f"{base}/", timeout=10) as page:   # and the page
        assert b"voice.js" in page.read()


def test_the_real_recogniser_without_its_model_says_what_to_fetch(tmp_path):
    ears = voice.Recognizer(folder=tmp_path / "nothing here")
    status = ears.status()
    assert status["ready"] is False and status["fix"] in (voice.FETCH, voice.INSTALL)
    assert ears.loader is None   # nothing was started
    with pytest.raises(voice.Unavailable) as e:
        ears.text([0.0] * 1600, 16000)
    assert e.value.fix in (voice.FETCH, voice.INSTALL)


MODEL = voice.MODELS / voice.MODEL


@pytest.mark.skipif(voice.Recognizer().missing() is not None, reason="no sherpa-onnx or no model")
def test_the_real_recogniser_hears_the_models_own_english_sample():
    ears = voice.Recognizer()
    samples, rate = voice.read_wav((MODEL / "test_wavs" / "en.wav").read_bytes())
    text = ears.text(samples, rate)
    assert "what your country can do for you" in text.lower()
    assert ears.status()["ready"] is True and ears.load_s > 0
