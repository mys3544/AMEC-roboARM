"""Voice commands for the console: what was said (speech recognition on this laptop, no cloud),
and the command in it (a small grammar, no language model: that comes later, on a lab server).

    python console/voice.py fetch                  # the speech model, once (487 MB, GitHub)
    python console/voice.py parse pick up the red cube    # the command in a sentence

The recogniser is NVIDIA's Parakeet TDT 0.6B v3 (int8 ONNX; 25 European languages, English,
Slovak and Czech among them), run by sherpa-onnx on the CPU. serve.py keeps one, loaded on a
helper thread the first time it is asked for (the page asks as it opens: GET /api/voice), and
answers POST /api/voice:

    Content-Type: audio/wav    a WAV, 16-bit PCM mono, any rate (the page sends the
                               microphone's own; sherpa-onnx resamples to 16 kHz)
                               -> {"text", "intent", "audio_s", "asr_ms"}
    Content-Type: text/plain   a typed command: the grammar only, no recogniser needed
                               -> {"text", "intent"}

Without sherpa-onnx or the model, a WAV is answered 503 with how to get them; nothing else in the
console needs either. Standard library, and sherpa-onnx when it is there.
"""

import argparse
import array
import importlib.util
import io
import pathlib
import re
import tarfile
import threading
import time
import unicodedata
import urllib.request
import wave

HERE = pathlib.Path(__file__).resolve().parent
MODELS = HERE / "models"   # gitignored: models/ anywhere in the tree
MODEL = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{MODEL}.tar.bz2"
FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
THREADS = 4   # CPU threads for one utterance; see README.md for what 1, 2, 4 and 8 measured
FETCH = "python console/voice.py fetch"
INSTALL = "uv sync (from the repo root: sherpa-onnx is in console/pyproject.toml)"


# --- the grammar: a sentence -> one command --------------------------------------------------

STOP_STEMS = ("halt", "freez", "abort", "zastav", "zastan")   # and "stop" anywhere in a word
STOP_WORDS = {"stoj", "stojte", "stuj", "stujte"}   # Slovak stoj, Czech stůj (accents dropped)
THREE = {"3", "three", "tri"}
CUBE_STEMS = ("cube", "kock", "kostk")   # Slovak kocka/kocku, Czech kostka/kostku
BLOCK_STEMS = ("block", "brick", "cuboid", "kvader", "kvadr", "hranol")
COLOURS = {"red": "red", "green": "green", "blue": "blue", "yellow": "yellow", "orange": "orange",
           "purple": "purple", "violet": "purple", "pink": "pink", "white": "white",
           "black": "black", "grey": "grey", "gray": "grey", "brown": "brown", "wood": "wood",
           "wooden": "wood"}
COLOUR_STEMS = {   # Slovak and Czech, with any ending: červenú, červenou -> cerven
    "cerven": "red", "zelen": "green", "modr": "blue", "zlt": "yellow", "zlut": "yellow",
    "oranzov": "orange", "fialov": "purple", "biel": "white", "ciern": "black", "cern": "black",
    "dreven": "wood"}
NUMBERS = {"thirty": 30, "forty": 40, "fourty": 40, "tridsat": 30, "tricet": 30, "styridsat": 40,
           "ctyricet": 40, "three": 3, "four": 4, "tri": 3, "styri": 4, "ctyri": 4}
SIZES = (30, 40)   # mm: the kit's cubes (the block is 30 x 30 x 60, and there is one)
SMALL_STEMS = ("small", "little", "tiny", "mal")   # mal-: Slovak/Czech malú, malou
BIG_STEMS = ("big", "large", "velk")   # velk-: veľkú, velkou
CLEAN_STEMS = ("clean", "clear", "tidy", "uprac", "uprat", "uklid")
HOME_WORDS = {"home", "homing", "domov", "domu"}
PICK_STEMS = ("pick", "grab", "take", "fetch", "lift",
              "zober", "zdvihni", "vezmi", "seber", "zvedni")   # Slovak, Czech
ALL_STEMS = ("every", "vset", "vsech")   # and "all": pick them all, pick everything -> clean up


def plain_words(text):
    """'Stůj! Pick up the X-3's 30mm cube.' -> stuj pick up the x 3 s 30 mm cube: lower case,
    no accents, digits apart from letters, no punctuation -- however the recogniser wrote it."""
    text = unicodedata.normalize("NFKD", text.casefold())
    return re.findall(r"[a-z]+|[0-9]+", "".join(c for c in text if not unicodedata.combining(c)))


def find_robots(words):
    """The robots named, and the words left over. X3 / X-3 / X three / ex three are the X3 Plus,
    M3 / M three / em three the M3 Pro; "Plus" or "Pro" alone counts only when neither is named
    so (Czech "pro" is "for": "vezmi kostku pro X3" is the X3)."""
    named, hinted, rest, i = set(), set(), [], 0
    while i < len(words):
        word, after = words[i], words[i + 1] if i + 1 < len(words) else ""
        if word in ("x", "ex", "iks") and after in THREE:
            named.add("x3")
            i += 2
        elif word in ("m", "em") and after in THREE:
            named.add("m3")
            i += 2
        else:
            if word in ("plus", "pro"):
                hinted.add("x3" if word == "plus" else "m3")
            else:
                rest.append(word)
            i += 1
    return named or hinted, rest


def unit(word):
    """mm in one of these, or None: mm, millimetres, cm, centimetres (Slovak and Czech too)."""
    if word == "mm" or word.startswith(("millimet", "milimet")):
        return 1
    if word == "cm" or word.startswith("centimet"):
        return 10
    return None


def find_size(words):
    """The size said, in mm, or None: 30, thirty, 30 mm, 3 cm, three centimetres; small (30),
    big (40). A number with no unit is a size only if it is 30 or 40 ("pick 2 cubes" is not)."""
    for i, word in enumerate(words):
        value = int(word) if word.isdigit() else NUMBERS.get(word)
        if value is not None:
            per = unit(words[i + 1]) if i + 1 < len(words) else None
            if per:
                return value * per
            if value in SIZES:
                return value
        elif word.startswith(SMALL_STEMS):
            return SIZES[0]
        elif word.startswith(BIG_STEMS):
            return SIZES[1]
    return None


def first(words, pick):
    return next((found for found in map(pick, words) if found), None)


def parse(text):
    """The command in a sentence, for the page to carry out:

        {"action": "stop", "robot": "all"}       stop / halt / freeze, stoj, stůj, zastav
        {"action": "pick", "robot": "m3"|"x3", "kind": "cube"|"block"|None,
         "colour": "red"|...|None, "size_mm": 30|40|None}
        {"action": "clean_up", "robot": ...}     clean up, clear the table / arena
        {"action": "home", "robot": ...}         go home
        {"action": "unknown", "text": text}      anything else (with "why", when it is known)

    Stop wins whenever it is said, whatever else is ("please stop the robot", "don't stop"
    included: stopping is never the wrong thing to do), and stops both robots. The robot is
    the M3 unless the X3 is named. Words that are not part of a command are passed over."""
    words = plain_words(text)
    if any("stop" in w or w.startswith(STOP_STEMS) or w in STOP_WORDS for w in words):
        return {"action": "stop", "robot": "all"}
    robots, words = find_robots(words)
    actions = set()
    if any(w.startswith(CLEAN_STEMS) for w in words):
        actions.add("clean_up")
    if any(w in HOME_WORDS for w in words):
        actions.add("home")
    if any(w.startswith(PICK_STEMS) for w in words):
        everything = any(w == "all" or w.startswith(ALL_STEMS) for w in words)
        actions.add("clean_up" if everything else "pick")
    if not actions:
        return {"action": "unknown", "text": text}
    if len(actions) > 1:
        return {"action": "unknown", "text": text,
                "why": f"more than one command ({' and '.join(sorted(actions))})"}
    if len(robots) > 1:
        return {"action": "unknown", "text": text, "why": "both robots named"}
    action, robot = actions.pop(), robots.pop() if robots else "m3"
    if action != "pick":
        return {"action": action, "robot": robot}
    kind = first(words, lambda w: "cube" if w.startswith(CUBE_STEMS)
                 else "block" if w.startswith(BLOCK_STEMS) else None)
    size = None if kind == "block" else find_size(words)
    if size is not None and size not in SIZES:
        return {"action": "unknown", "text": text, "why": f"no {size} mm cube in the kit"}
    colour = first(words, lambda w: COLOURS.get(w) or next(
        (c for stem, c in COLOUR_STEMS.items() if w.startswith(stem)), None))
    return {"action": "pick", "robot": robot, "kind": kind, "colour": colour, "size_mm": size}


# --- the recogniser --------------------------------------------------------------------------

class Unavailable(Exception):
    """No recogniser to hear with: why, and the line that fixes it."""

    def __init__(self, why, fix):
        super().__init__(why)
        self.fix = fix


class Recognizer:
    """Parakeet through sherpa-onnx, loaded once, on a helper thread, the first time it is asked
    for. One utterance at a time; each waits for the load if it is not done yet."""

    def __init__(self, folder=MODELS / MODEL):
        self.folder = folder
        self.engine = None
        self.load_s = None
        self.failed = None   # (why, fix) if the load failed
        self.loader = None   # the helper thread, once started
        self.loaded = threading.Event()
        self.starting = threading.Lock()
        self.busy = threading.Lock()

    def missing(self):
        """(why, fix) if it cannot load; None if it can."""
        if importlib.util.find_spec("sherpa_onnx") is None:
            return "sherpa-onnx is not installed", INSTALL
        gone = [f for f in FILES if not (self.folder / f).is_file()]
        if gone:
            return f"no speech model in {self.folder} ({', '.join(gone)} missing)", FETCH
        return None

    def start(self):
        """Begin the load, unless it began already or cannot (then a later call tries again:
        the model may be fetched while the console runs)."""
        with self.starting:
            if self.loader is None and self.missing() is None:
                self.loader = threading.Thread(target=self.load, daemon=True)
                self.loader.start()

    def load(self):
        try:
            import sherpa_onnx
            t0 = time.perf_counter()
            self.engine = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(self.folder / FILES[0]), decoder=str(self.folder / FILES[1]),
                joiner=str(self.folder / FILES[2]), tokens=str(self.folder / FILES[3]),
                num_threads=THREADS, model_type="nemo_transducer",
                decoding_method="greedy_search")
            self.load_s = round(time.perf_counter() - t0, 1)
        except Exception as e:   # noqa: BLE001 -- a broken download, say: the page must hear it
            self.failed = (f"the speech model did not load: {e}",
                           f"delete {self.folder}, then {FETCH}")
        finally:
            self.loaded.set()

    def status(self):
        """For the page: ready, loading, or why not and the fix. Starts the load."""
        self.start()
        if self.engine is not None:
            return {"ready": True, "load_s": self.load_s}
        why = self.failed or (None if self.loader else self.missing())
        if why:
            return {"ready": False, "error": why[0], "fix": why[1]}
        return {"ready": False, "loading": True}

    def text(self, samples, rate):
        """What was said in these samples (mono floats in -1..1 at this rate)."""
        self.start()
        if self.loader is None:
            raise Unavailable(*(self.missing() or ("the speech model is not loaded", FETCH)))
        self.loaded.wait()
        if self.engine is None:
            raise Unavailable(*self.failed)
        with self.busy:
            stream = self.engine.create_stream()
            stream.accept_waveform(rate, samples)
            self.engine.decode_stream(stream)
            return stream.result.text.strip()


def read_wav(data):
    """A WAV's samples, floats in -1..1, and its rate. 16-bit PCM mono only: what the page sends
    (and what the model's test_wavs are)."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            width, channels, rate = w.getsampwidth(), w.getnchannels(), w.getframerate()
            frames = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as e:
        raise ValueError(f"not a WAV this reads: {e}") from None
    if (width, channels) != (2, 1):
        raise ValueError(f"{8 * width}-bit, {channels} channels: 16-bit PCM mono only")
    return [s / 32768 for s in array.array("h", frames)], rate


def answer(recognizer, body, content_type):
    """POST /api/voice -> (HTTP status, reply): a WAV is heard, then parsed; text is parsed."""
    if content_type.split(";")[0].strip().lower() == "text/plain":
        text = body.decode("utf-8", "replace").strip()
        return 200, {"text": text, "intent": parse(text)}
    try:
        samples, rate = read_wav(body)
    except ValueError as e:
        return 400, {"error": str(e)}
    t0 = time.perf_counter()
    try:
        text = recognizer.text(samples, rate)
    except Unavailable as e:
        return 503, {"error": str(e), "fix": e.fix}
    return 200, {"text": text, "intent": parse(text), "audio_s": round(len(samples) / rate, 2),
                 "asr_ms": round((time.perf_counter() - t0) * 1000)}


# --- from the command line -------------------------------------------------------------------

def fetch(folder=MODELS):
    """Download and unpack the model, once, from sherpa-onnx's own releases on GitHub."""
    target = folder / MODEL
    if all((target / f).is_file() for f in FILES):
        print(f"already here: {target}")
        return
    folder.mkdir(parents=True, exist_ok=True)
    print(f"from  {URL}\ninto  {folder}, unpacked as it comes (487 MB: minutes)", flush=True)
    with (urllib.request.urlopen(URL, timeout=60) as reply,
          tarfile.open(fileobj=reply, mode="r|bz2") as tar):
        tar.extractall(folder, filter="data")
    for f in FILES:
        print(f"      {f}  {(target / f).stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch", help="download the speech model into console/models/")
    said = commands.add_parser("parse", help="the command in a sentence")
    said.add_argument("words", nargs="+")
    args = parser.parse_args()
    if args.command == "fetch":
        fetch()
    else:
        print(parse(" ".join(args.words)))


if __name__ == "__main__":
    main()
