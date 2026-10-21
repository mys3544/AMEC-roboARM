"""Voice commands understood by a small language model on this laptop (2026-10-20, for the event
out of the lab on 2026-10-22: no internet there, no GPU server). The words the recogniser heard
(voice.py) go to llama.cpp's llama-server on the laptop's GPU; the model picks one of the console's
functions and fills in its arguments, as JSON held to a schema by llama.cpp's grammar (always
valid, only the functions and values there are). Stop never waits for it: voice.parse finds a stop
first, at once, whatever else was said; without the model the grammar does the rest, as before.

    python console/llm.py serve [--model M]      # llama-server alone (the console starts its own)
    python console/llm.py ask roztrieď kocky      # the command in a sentence
    python console/llm.py bench [--model M ...]  # each model on llm_cases.json: right, and how fast

The files live outside the repo, in ~/.roboarm/llm (ROBOARM_LLM to move it): bin/ the llama.cpp
release b11617 for Windows, CUDA 12.4 (llama-server.exe and the CUDA DLLs, both zips unpacked
there), models/ the GGUFs (MODELS). Standard library only.
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
HOME = pathlib.Path(os.environ.get("ROBOARM_LLM") or pathlib.Path.home() / ".roboarm" / "llm")
SERVER = HOME / "bin" / "llama-server.exe"
MODELS = {   # name -> GGUF in HOME/models (unsloth's quantisations, Hugging Face)
    "qwen3.5-2b": "Qwen3.5-2B-Q4_K_M.gguf",
    "qwen3.5-4b": "Qwen3.5-4B-Q4_K_M.gguf",
    "qwen3-4b": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "gemma4-e2b": "gemma-4-E2B-it-Q4_K_M.gguf",
    "gemma4-e4b": "gemma-4-E4B-it-Q4_K_M.gguf",
}
DEFAULT = "qwen3.5-4b"
PORT = 8081
TIMEOUT_S = 6   # a command later than this is the grammar's
WARM_S = 180    # the first answer after a start: 15 s on the RTX 2070 Super (CUDA readies itself), 0.4 s after
CASES = HERE / "llm_cases.json"

ACTIONS = ["pick", "pick_named", "fetch", "clean_up", "sort", "look", "find", "home", "stop", "unknown"]
COLOURS = ["red", "green", "blue", "yellow", "orange", "purple", "white", "black", "wood"]
PROMPT = """You turn one spoken command for two robot arms into one function call, as JSON.
The speech was written down by a recogniser: words may be misheard, accents missing. The speaker
talks Slovak, Czech or English. The recogniser often writes Slovak as Russian in Cyrillic
("Икс три убрать ствол" is "X3 uprac stôl") or in Polish-looking spellings ("kotku", "kostkę" is
"kocku"; "kwadrę" is "kváder"; "velkou" is "veľkú"): read them as the Slovak they sound like.

Robots: "m3" (the M3 Pro: also "M3", "em tri", "Pro"), "x3" (the X3 Plus: also "X3", "iks tri",
"Plus"), named anywhere in the sentence. A robot named is that robot, whatever it is asked.
"both" only when both robots are meant ("oba roboty", "obaja", "both robots"): "všetko",
"všetky", "all" are about the pieces, not the robots. No robot named: "m3".
On each robot's table: kit pieces -- cubes of 30 mm (small, "malá") and 40 mm (big, "veľká"),
and blocks of 30 x 30 x 60 mm ("kváder", "hranol", "block", "kvádr") -- and a box they go into.

Functions (action):
- pick: pick up ONE kit piece into the box. kind "cube"/"block", colour, size_mm 30/40 only when
  said ("nejaký kúsok", "something": all null).
- pick_named: the M3 picks a thing it was taught, by its name (only the M3 has any, whichever robot
  is named): anything asked for that is not a cube or a block (a fish, an apricot, newspapers, a
  can). name: the known name it means, said in any language and maybe by one of its words alone
  ("rybu" -> "fish", "marhuľu" -> "apricot", "plechovku" -> a "... can", "valec" -> a "...
  cylinder"). Known names: {names}.
  A cube or block word, however misheard ("kocka", "kotka", "kostka", "kváder", "cube"), is a
  pick, never pick_named. A thing that is none of the known names (scissors): unknown.
- fetch: the M3 goes for a piece or a taught thing, driving up to it when it is out of reach, picks
  it and brings it back into the box ("prines", "dones", "prineste", "bring", "fetch"; "zober",
  "podaj" are pick or pick_named): kind, colour, size_mm as pick, or name as pick_named.
- clean_up: every piece in reach into the box, one by one ("uprac", "ukliď", "clear the table",
  "zober všetko"). kind only when the speaker limits it to cubes or to blocks ("uprac kocky");
  "uprac stôl" has kind null.
- sort: sort the pieces ("roztrieď", "roztřiď", "sort"). by "kind" or "colour" (by kind when unsaid).
- look: look over the table, take nothing ("pozri sa", "podívej se", "look").
- find: the M3 names the taught things it sees ("nájdi", "čo vidíš", "what do you see").
- home: the arm back to rest ("domov", "domů", "go home").
- stop: stop.
- unknown: anything else -- not a command for the robots, or words you cannot read as one
  command for sure (unknown moves nothing; a wrong guess moves a robot).
Every field not said is null."""
EXAMPLES = [   # (said, the call): one of each, in the three languages
    ("zober červenú kocku", {"action": "pick", "robot": "m3", "kind": "cube", "colour": "red", "size_mm": None}),
    ("X3 pick up the big cube", {"action": "pick", "robot": "x3", "kind": "cube", "colour": None, "size_mm": 40}),
    ("uprac stôl, Plus", {"action": "clean_up", "robot": "x3", "kind": None}),
    ("oba roboty ukliďte kostky", {"action": "clean_up", "robot": "both", "kind": "cube"}),
    ("podaj mi marhuľu", {"action": "pick_named", "robot": "m3", "name": "apricot"}),
    ("roztrieď kúsky podľa farby", {"action": "sort", "robot": "m3", "by": "colour"}),
    ("čo vidíš na stole", {"action": "find", "robot": "m3"}),
    ("aké je dnes počasie", {"action": "unknown"}),
    ("podaj mi kľúče", {"action": "unknown"}),   # a thing not taught (2026-10-21: "nožnice" was taken for "news")
]
FIELDS = ["action", "robot", "kind", "colour", "size_mm", "name", "by"]
# The robots' few words as the recogniser mangles them, even into French or English (2026-10-21,
# Slovak spoken on the robots) -> the word meant, put back before the model reads the sentence: told
# these in its prompt, the 4B model began picking cubes for "valec".
SOUNDS_LIKE = [
    (re.compile(r"\b(?:v[aá]l+e[ct]s?|vallet|valence|bal+et)\b", re.IGNORECASE), "valec"),   # a cylinder
    (re.compile(r"\b(?:zuber|zobar)\b", re.IGNORECASE), "zober"),                             # pick
    (re.compile(r"\b(?:u ?prac\w*|[uo] ?prat\w*|prac)\b", re.IGNORECASE), "uprac"),           # clean up
    (re.compile(r"\b(?:у ?братств\w*|убратств\w*|убрать)", re.IGNORECASE), "uprac"),           # in Cyrillic
    (re.compile(r"\b(?:ствол|стол)\b", re.IGNORECASE), "stôl"),                                  # the table
]


def as_meant(text):
    """`text` with the mangled words SOUNDS_LIKE knows put back."""
    for pattern, word in SOUNDS_LIKE:
        text = pattern.sub(word, text)
    return text


class Unavailable(Exception):
    """No model to ask: why, and the line that fixes it."""

    def __init__(self, why, fix):
        super().__init__(why)
        self.fix = fix


def schema(names):
    """The JSON a reply must be (llama.cpp turns it into a grammar): an action and its arguments,
    every field present, null where not said; a name only one of those taught."""
    nullable = lambda values: {"enum": [*values, None]}
    return {"type": "object", "additionalProperties": False, "required": FIELDS, "properties": {
        "action": {"enum": ACTIONS}, "robot": {"enum": ["m3", "x3", "both"]},
        "kind": nullable(["cube", "block"]), "colour": nullable(COLOURS), "size_mm": nullable([30, 40]),
        "name": nullable(names), "by": nullable(["kind", "colour"])}}


def full(call):
    return {f: call.get(f) for f in FIELDS} | {"robot": call.get("robot", "m3")}


def messages(text, names):
    said = [{"role": "system", "content": PROMPT.replace("{names}", ", ".join(names) if names else "none yet")}]
    for example, call in EXAMPLES:
        said += [{"role": "user", "content": example}, {"role": "assistant", "content": json.dumps(full(call))}]
    return said + [{"role": "user", "content": text}]


def intent(call, text):
    """The model's call -> the command the page carries out (voice.parse's shape, and the new ones)."""
    action, robot = call["action"], call["robot"] or "m3"
    if action == "stop":
        return {"action": "stop", "robot": "all"}
    if action == "unknown":
        return {"action": "unknown", "text": text}
    if action == "pick":
        kind = call["kind"]
        return {"action": "pick", "robot": robot, "kind": kind, "colour": call["colour"],
                "size_mm": None if kind == "block" else call["size_mm"]}
    if action == "pick_named":
        if not call["name"]:
            return {"action": "unknown", "text": text, "why": "no thing of that name taught"}
        return {"action": "pick_named", "robot": "m3", "name": call["name"]}
    if action == "fetch":   # only the M3 drives: whichever robot was named
        kind = call["kind"]
        return {"action": "fetch", "robot": "m3", "name": call["name"], "kind": None if call["name"] else kind,
                "colour": call["colour"], "size_mm": None if kind == "block" else call["size_mm"]}
    if action == "clean_up":
        return {"action": "clean_up", "robot": robot, "kind": call["kind"]}
    if action == "sort":
        return {"action": "sort", "robot": robot, "by": call["by"] or "kind"}
    if action == "find":
        return {"action": "find", "robot": "m3"}
    return {"action": action, "robot": robot}   # look, home


class Brain:
    """llama-server on this laptop, started once (or found running: `llm.py serve`), asked one
    command at a time. `names()`: the things the M3 was taught, for pick_named (serve.py asks the
    robot)."""

    def __init__(self, model=DEFAULT, port=PORT, names=list):
        self.model, self.port, self.names = model, port, names
        self.url = f"http://127.0.0.1:{port}"
        self.proc = None
        self.started = None

    def missing(self):
        """(why, fix) if it cannot run; None if it can."""
        if not SERVER.is_file():
            return f"no llama-server in {SERVER.parent}", "unpack llama.cpp b11617 (win, CUDA 12.4) and its cudart there"
        if self.model not in MODELS or not (HOME / "models" / MODELS[self.model]).is_file():
            return f"no model {self.model} in {HOME / 'models'}", f"one of {', '.join(MODELS)}, as llm.py MODELS names"
        return None

    def up(self):
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=1) as reply:
                return reply.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def start(self):
        """llama-server with the model, on the GPU (not waited for: it loads in seconds); none if
        one answers already on the port, or it cannot run."""
        if self.proc is not None or self.missing() or self.up():
            return
        log = open(HOME / "server.log", "ab")   # noqa: SIM115 -- the child's, while it runs
        self.proc = subprocess.Popen(
            [str(SERVER), "-m", str(HOME / "models" / MODELS[self.model]), "-ngl", "99", "-c", "4096", "-np", "1",
             "--host", "127.0.0.1", "--port", str(self.port), "--jinja"],
            stdout=log, stderr=subprocess.STDOUT, cwd=SERVER.parent,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.started = time.time()

    def warm(self):
        """The first answer, on a helper thread: it takes ~15 s after a start, the next ~0.4 s."""
        def first():
            t0 = time.time()
            while not self.up() and time.time() - t0 < WARM_S:
                time.sleep(0.5)
            try:
                self.ask("domov", timeout=WARM_S)
            except Exception:   # noqa: BLE001, S110 -- not up: the grammar answers meanwhile
                pass
        threading.Thread(target=first, daemon=True).start()

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            self.proc = None

    def status(self):
        why = self.missing()
        if why:
            return {"ready": False, "error": why[0], "fix": why[1]}
        return {"ready": self.up(), "model": self.model}

    def ask(self, text, timeout=TIMEOUT_S):
        """(the command in `text`, ms): Unavailable when no model answers in time."""
        if self.missing():
            raise Unavailable(*self.missing())
        names = sorted(set(self.names() or []))
        body = {"messages": messages(as_meant(text), names), "temperature": 0, "max_tokens": 256,   # Gemma indents its JSON
                "response_format": {"type": "json_schema", "json_schema": {"name": "command", "schema": schema(names)}},
                "chat_template_kwargs": {"enable_thinking": False}}
        request = urllib.request.Request(f"{self.url}/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as reply:
                answer = json.loads(reply.read())
        except (OSError, urllib.error.URLError) as e:
            raise Unavailable(f"the language model did not answer: {e}", "python console/llm.py serve") from None
        ms = round((time.perf_counter() - t0) * 1000)
        try:
            call = json.loads(answer["choices"][0]["message"]["content"])
        except (ValueError, KeyError, IndexError) as e:   # cut short, say: the grammar's then
            raise Unavailable(f"the language model's answer is not a command: {e}", "a larger max_tokens") from None
        return intent(full(call), text), ms


# --- from the command line -------------------------------------------------------------------

def wait_up(brain, limit_s=120):
    t0 = time.time()
    while not brain.up():
        if brain.proc is not None and brain.proc.poll() is not None:
            raise SystemExit(f"llama-server ended (code {brain.proc.returncode}): see {HOME / 'server.log'}")
        if time.time() - t0 > limit_s:
            raise SystemExit(f"llama-server not up after {limit_s} s: see {HOME / 'server.log'}")
        time.sleep(0.3)
    return round(time.time() - t0, 1)


def matches(got, want):
    return all(got.get(k) == v for k, v in want.items())


def bench(models, names):
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    for model in models:
        brain = Brain(model, names=lambda: names)
        if brain.up():
            raise SystemExit(f"something answers on :{brain.port} already: stop it first")
        brain.start()
        try:
            load = wait_up(brain)
            t0 = time.time()
            brain.ask("domov", timeout=WARM_S)   # the first answer: CUDA readies itself, the prompt is cached
            load = f"{load} s + {time.time() - t0:.1f} s the first answer"
            right, times = 0, []
            for case in cases:
                try:
                    got, ms = brain.ask(case["say"])
                except Unavailable as e:
                    got, ms = {"error": str(e)}, TIMEOUT_S * 1000
                times.append(ms)
                ok = matches(got, case["want"])
                right += ok
                if not ok:
                    print(f"  {model}: {case['say']!r} -> {got} (want {case['want']})", flush=True)
            times.sort()
            print(f"{model:12s} {right}/{len(cases)} right, median {times[len(times) // 2]} ms, "
                  f"worst {times[-1]} ms, loaded in {load}", flush=True)
        finally:
            brain.stop()
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="llama-server with the model, until Ctrl-C")
    serve.add_argument("--model", default=DEFAULT, choices=MODELS)
    ask = commands.add_parser("ask", help="the command in a sentence (a server must run)")
    ask.add_argument("words", nargs="+")
    ask.add_argument("--names", default="apricot,fish,news,cosmetics", help="taught names, comma-separated")
    test = commands.add_parser("bench", help="each model on llm_cases.json")
    test.add_argument("--model", nargs="+", default=[m for m in MODELS if (HOME / "models" / MODELS[m]).is_file()])
    test.add_argument("--names", default="apricot,fish,news,cosmetics", help="taught names, comma-separated")
    args = parser.parse_args()
    if args.command == "serve":
        brain = Brain(args.model)
        if brain.missing():
            raise SystemExit(" -- ".join(brain.missing()))
        brain.start()
        print(f"{args.model} up in {wait_up(brain)} s on {brain.url} (Ctrl-C to stop)", flush=True)
        try:
            brain.proc.wait()
        except KeyboardInterrupt:
            brain.stop()
    elif args.command == "ask":
        print(*Brain(names=lambda: args.names.split(",")).ask(" ".join(args.words)))
    else:
        bench(args.model, args.names.split(","))


if __name__ == "__main__":
    sys.exit(main())
