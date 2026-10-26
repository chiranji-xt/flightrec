"""`flightrec show` and `flightrec diff`: inspect recordings from the terminal."""

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

from .store import read_events
from .viewer import write_viewer


def _short(value: Any, limit: int = 70) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def show(path: str, out=None) -> None:
    out = out or sys.stdout
    events = read_events(path)
    print(f"{path}: {len(events)} call(s)", file=out)
    for ev in events:
        status = "ERR" if ev["error"] else "ok "
        print(
            f"{ev['seq']:>3}  {status} {ev['kind']:<15} {ev['name']:<26} "
            f"{ev['duration'] * 1000:7.1f}ms  {_short(ev['request'])}",
            file=out,
        )
        if ev.get("stream"):
            tail = " (stopped early)" if ev.get("incomplete") else ""
            print(f"       -> stream of {len(ev['response'])} chunks{tail}" + (f"; {_short(ev['error'])}" if ev["error"] else ""), file=out)
            continue
        outcome = ev["error"] or ev["response"]
        print(f"       -> {_short(outcome)}", file=out)


def first_divergence(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Describe where two runs first differ, or None if they are identical."""
    for i, (ea, eb) in enumerate(zip(a, b)):
        if (ea["kind"], ea["name"]) != (eb["kind"], eb["name"]):
            return {"step": i, "why": "different call", "a": f"{ea['kind']} {ea['name']}", "b": f"{eb['kind']} {eb['name']}"}
        if ea["request_hash"] != eb["request_hash"]:
            return {"step": i, "why": "different request", "a": _short(ea["request"], 200), "b": _short(eb["request"], 200)}
        if ea["response"] != eb["response"] or ea["error"] != eb["error"]:
            return {
                "step": i,
                "why": "same request, different result",
                "a": _short(ea["error"] or ea["response"], 200),
                "b": _short(eb["error"] or eb["response"], 200),
            }
    if len(a) != len(b):
        longer = "A" if len(a) > len(b) else "B"
        return {"step": min(len(a), len(b)), "why": f"run {longer} continued after the other stopped", "a": "", "b": ""}
    return None


def diff(path_a: str, path_b: str, out=None) -> int:
    out = out or sys.stdout
    found = first_divergence(read_events(path_a), read_events(path_b))
    if found is None:
        print("runs are identical", file=out)
        return 0
    print(f"first divergence at step {found['step']}: {found['why']}", file=out)
    if found["a"] or found["b"]:
        print(f"  A: {found['a']}", file=out)
        print(f"  B: {found['b']}", file=out)
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="flightrec", description="Inspect agent recordings.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_show = sub.add_parser("show", help="print a timeline of a recording")
    p_show.add_argument("recording")
    p_diff = sub.add_parser("diff", help="find where two recordings first diverge")
    p_diff.add_argument("a")
    p_diff.add_argument("b")
    p_view = sub.add_parser("view", help="open a recording in the browser viewer")
    p_view.add_argument("recording")
    p_view.add_argument("-o", "--out", help="where to write the HTML (default: next to the recording)")
    p_view.add_argument("--no-open", action="store_true", help="write the file but do not open a browser")
    args = parser.parse_args(argv)
    if args.command == "show":
        show(args.recording)
        return 0
    if args.command == "view":
        recording = Path(args.recording)
        target = write_viewer(recording, Path(args.out) if args.out else recording.with_suffix(".html"))
        print(f"wrote {target}")
        if not args.no_open:
            webbrowser.open(target.resolve().as_uri())
        return 0
    return diff(args.a, args.b)


if __name__ == "__main__":
    sys.exit(main())
