"""Write a longer, messier recording (several turns, an error, odd text) for trying the viewer.

    python examples/make_rich_recording.py && flightrec view runs/rich.jsonl
"""

from types import SimpleNamespace

import flightrec


@flightrec.tool
def read_file(path: str) -> str:
    if path.endswith(".bak"):
        raise FileNotFoundError(f"{path}: no such file")
    return "def add(a, b):\n    return a - b  # <script>alert('not run')</script>\n" * 3


@flightrec.tool
def run_tests(target: str) -> dict:
    return {"passed": 3, "failed": 1, "log": "FAILED test_add: assert 1 == -1\n" * 4}


class Messages:
    def __init__(self):
        self.turn = 0

    def create(self, model, messages, **_):
        self.turn += 1
        plans = [
            ("tool_use", "read_file", {"path": "calc.py.bak"}),
            ("tool_use", "read_file", {"path": "calc.py"}),
            ("tool_use", "run_tests", {"target": "calc.py"}),
        ]
        if self.turn <= len(plans):
            _, name, args = plans[self.turn - 1]
            block = SimpleNamespace(type="tool_use", id=f"t{self.turn}", name=name, input=args)
            return SimpleNamespace(stop_reason="tool_use", content=[block])
        text = SimpleNamespace(type="text", text="Found it: `add` subtracts. Change `a - b` to `a + b`.")
        return SimpleNamespace(stop_reason="end_turn", content=[text])


def main() -> None:
    client = SimpleNamespace(messages=Messages())
    with flightrec.Recorder("runs/rich.jsonl") as rec:
        llm = rec.wrap(client)
        messages = [{"role": "user", "content": "Why does test_add fail in calc.py?"}]
        flightrec.now()
        while True:
            reply = llm.messages.create(model="fake-1", system="You are a careful debugging agent.", messages=messages)
            if reply.stop_reason == "end_turn":
                break
            block = reply.content[0]
            args = block.input if isinstance(block.input, dict) else block.input.model_dump()
            tools = {"read_file": read_file, "run_tests": run_tests}
            try:
                result = tools[block.name](**args)
            except FileNotFoundError as exc:
                result = f"error: {exc}"
            messages.append({"role": "assistant", "content": [{"type": "tool_use", "id": block.id, "name": block.name, "input": args}]})
            messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": block.id, "content": str(result)}]})
    print("wrote runs/rich.jsonl")


if __name__ == "__main__":
    main()
