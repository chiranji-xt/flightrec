"""A tiny tool-using agent with a scripted fake model, so the demo needs no API key.

    python examples/demo_agent.py record      # run for real, write runs/demo.jsonl
    python examples/demo_agent.py replay      # re-run offline from the recording
    python examples/demo_agent.py record-bad  # same task, but the search tool truncates its result
    flightrec diff runs/demo.jsonl runs/demo-bad.jsonl
"""

import sys
from types import SimpleNamespace

import flightrec

DOCS = {"flightrec": "flightrec records model and tool calls so agent runs can be replayed offline."}


@flightrec.tool
def search(query: str) -> str:
    return DOCS.get(query, "no results")


def search_truncated(query: str) -> str:
    return DOCS.get(query, "no results")[:12]


class FakeMessages:
    """Calls search first, then answers from whatever the tool returned."""

    def create(self, model, messages, max_tokens=256, **_):
        last = messages[-1]
        if last["role"] == "user" and isinstance(last["content"], str):
            block = SimpleNamespace(type="tool_use", id="t1", name="search", input={"query": "flightrec"})
            return SimpleNamespace(stop_reason="tool_use", content=[block])
        found = last["content"][0]["content"]
        block = SimpleNamespace(type="text", text=f"Answer: {found}")
        return SimpleNamespace(stop_reason="end_turn", content=[block])


class FakeClient:
    messages = FakeMessages()


def run_agent(client, task: str, search_tool=search) -> str:
    messages = [{"role": "user", "content": task}]
    while True:
        response = client.messages.create(model="fake-1", messages=messages)
        if response.stop_reason == "end_turn":
            return next(b.text for b in response.content if b.type == "text")
        for block in response.content:
            if block.type == "tool_use":
                result = search_tool(**block.input.model_dump() if hasattr(block.input, "model_dump") else block.input)
                messages.append({"role": "assistant", "content": "tool call"})
                messages.append({"role": "user", "content": [{"type": "tool_result", "content": result}]})


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "record"
    task = "What does flightrec do?"
    if mode == "replay":
        with flightrec.Recorder("runs/demo.jsonl", mode="replay") as rec:
            print(run_agent(rec.wrap(None), task))
    elif mode == "record-bad":
        with flightrec.Recorder("runs/demo-bad.jsonl") as rec:
            print(run_agent(rec.wrap(FakeClient()), task, search_tool=flightrec.tool(search_truncated, name="search")))
    else:
        with flightrec.Recorder("runs/demo.jsonl") as rec:
            print(run_agent(rec.wrap(FakeClient()), task))


if __name__ == "__main__":
    main()
