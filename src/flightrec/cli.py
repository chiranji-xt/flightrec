"""`flightrec show` and `flightrec diff`: inspect recordings from the terminal."""

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from .store import read_events


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
    args = parser.parse_args(argv)
    if args.command == "show":
        show(args.recording)
        return 0
    return diff(args.a, args.b)


if __name__ == "__main__":
    sys.exit(main())
