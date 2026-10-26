# flightrec

A flight recorder for AI agents. Record every model call, tool call, clock read and dice roll of a run, then **replay it offline** with no API calls, or **diff two runs** to see exactly where they split.

Agent bugs are hard to chase because the run that failed will not happen again: the model answers differently, the search returns new results. flightrec turns one bad run into a file you can replay as many times as you like.

## Use

```python
import flightrec

@flightrec.tool
def search(query: str) -> str: ...

# Record: calls run for real and are written to the file as they happen.
with flightrec.Recorder("runs/bad.jsonl") as rec:
    run_agent(rec.wrap(anthropic.Anthropic()), task)

# Replay: nothing runs. Results come from the file, in order. No network, no cost.
with flightrec.Recorder("runs/bad.jsonl", mode="replay") as rec:
    run_agent(rec.wrap(None), task)
```

`rec.wrap(client)` records `client.messages.create` by default. Pass `methods=("chat.completions.create",)` for other SDKs. Replayed responses support attribute access (`resp.content[0].text`) like the originals.

Commit a recording of a failure and replay it in CI as a regression test: it costs nothing and cannot flake.

### Divergence is an error, not a guess

During replay, if your code changes so that it makes a different call, or the same call with different arguments, or finishes early, flightrec raises `ReplayDivergence` and names the step. That is what makes a replayed run trustworthy.

### Nondeterminism

`flightrec.now()` and `flightrec.random()` replay to their recorded values. For anything else, `flightrec.capture("name", fn, **request)`.

## Inspect

```
$ flightrec show runs/demo.jsonl
runs/demo.jsonl: 3 call(s)
  0  ok  llm    llm.messages.create   0.0ms  {"model": "fake-1", ...
  1  ok  tool   search                0.0ms  {"query": "flightrec"}
  2  ok  llm    llm.messages.create   0.0ms  ...

$ flightrec diff runs/demo.jsonl runs/demo-bad.jsonl
first divergence at step 1: same request, different result
  A: "flightrec records model and tool calls so agent runs can be replayed offline."
  B: "flightrec re"
```

Try it without an API key: `python examples/demo_agent.py record`, then `replay`, then `record-bad`.

## Recording format

JSON Lines. A header line, then one object per call: `seq`, `kind` (`llm`, `tool`, `nondeterminism`), `name`, `request`, `request_hash`, `response`, `error`, `started`, `duration`. Each event is flushed as it happens, so a crashed run still leaves a usable recording.

## Limits (v0.1)

- Streaming responses are not recorded yet.
- Calls are matched by order, so concurrent calls (threads, `asyncio.gather`) replay in recorded order only if they are made in the same order.
- Async clients are not wrapped yet.

## Roadmap

- [ ] Time-travel viewer in the browser: scrub steps, inspect the context at each one
- [ ] Branching: edit a result at step N and re-run from there
- [ ] Streaming and async support
- [ ] `flightrec replay module:function` from the CLI

## Develop

```
pip install -e ".[dev]"
pytest
```
