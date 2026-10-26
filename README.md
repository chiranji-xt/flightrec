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

### Async and streaming

Nothing changes in how you use it. Async clients, `async def` tools and streams are recorded as you use them and come back the same way on replay.

```python
@flightrec.tool
async def search(query: str) -> str: ...

async with flightrec.Recorder("runs/a.jsonl") as rec:              # `with` works too
    llm = rec.wrap(anthropic.AsyncAnthropic())
    async with llm.messages.stream(**request) as stream:           # the messages.stream() helper
        async for text in stream.text_stream:
            ...
        final = await stream.get_final_message()
    events = await llm.messages.create(stream=True, **request)     # raw events
    async for event in events: ...
```

A streamed call is one step in the recording, holding its chunks. They are captured at the source, so it is recorded the same whether you iterate events, read `text_stream`, or only ask for the final message. In replay, chunks and the final message come back as the SDK's own types, with no network. In the viewer a stream shows its assembled text, its chunk list, and a note if it stopped early or failed part-way.

Use `stream_methods=` on `wrap()` for other SDKs' stream helpers (default: `("messages.stream",)`). Branching works across streams: earlier streams replay, and a stream step can be re-run live or made to fail. Editing one replaces its whole list of chunks.

### Works with the real Anthropic SDK

Replayed responses are rebuilt as the SDK's own types, so `isinstance(resp, anthropic.types.Message)`, `block.input` being a `dict`, and `**block.input` all behave as in the live run. This relies on the SDK being imported in your program (it is, if you use it); a recording can never cause an import. If the class is not available, you get an attribute-access stand-in instead.

The test suite runs record, replay and branch against the real SDK client with only the network faked (`tests/test_anthropic_sdk.py`, no API key needed). One extra test makes a real, tiny API request; it runs only with `FLIGHTREC_LIVE=1` and `ANTHROPIC_API_KEY` set.

### Branch: what if step N had gone differently?

Pick a step in a recording, change what it returned, and continue from there with a real agent. Steps before it replay from the file, the edited step answers with your value, and every later step runs live, so the agent reacts to the change for real.

```python
with flightrec.Recorder(
    "runs/bad.jsonl", mode="branch", at=31,
    response={"results": [{"title": "the right page"}]},   # or error=TimeoutError("search down")
    out="runs/bad.branch.jsonl",
) as rec:
    run_agent(rec.wrap(anthropic.Anthropic()), task)       # real client, real tools
```

Leave out `response`/`error` to re-run step 31 live and continue from there. The new run is saved to `out` as a complete recording: replay it, view it (the branch point is marked, with the original result kept), or `flightrec diff` it against the source to see what changed.

In the viewer, select a step and use **Branch from here**: edit the result, or choose "raise an error", and it writes the snippet for you.

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

### Browser viewer

```
flightrec view runs/demo.jsonl
```

Writes a single self-contained `demo.html` next to the recording and opens it. Scrub the timeline (drag, or `←` `→` `Home` `End`), watch the context grow call by call, and see which messages are new at each model call. Errors show in red. The page has no server and uploads nothing; you can also open any viewer page and drop a different `.jsonl` onto it.

`python examples/make_rich_recording.py` writes a longer sample (several turns, a tool error) to try it on.

Try it without an API key: `python examples/demo_agent.py record`, then `replay`, then `record-bad`.

## Recording format

JSON Lines. A header line, then one object per call: `seq`, `kind` (`llm`, `tool`, `nondeterminism`), `name`, `request`, `request_hash`, `response`, `error`, `started`, `duration`. Each event is flushed as it happens, so a crashed run still leaves a usable recording.

Extra fields appear when they apply: `async` (the call was awaited), `stream` (`"events"` or `"manager"`, with `response` as the list of chunks), `final` (a stream helper's final message), `incomplete`, `edited` and `original` (a branch point), `response_type` (the SDK class to rebuild on replay). Because a stream's event is written when it ends, lines can be out of order in the file; readers sort by `seq`.

## Limits (v0.1)

- Calls are matched by the order they start in. `asyncio.gather` works, because tasks start in a fixed order; threads do not (the call counter is not thread-safe).
- A stream is written to the file when it ends. If the process dies mid-stream, that stream's event is lost; an abandoned or closed stream is written as `"incomplete": true`.
- Replayed `messages.stream()` helpers support iteration, `text_stream`, `until_done()`, `get_final_message()`, `get_final_text()` and `current_message_snapshot`. Attributes that come from the live HTTP response (`.response`, `.request_id`) do not exist on replay.
- A call that is cancelled (`asyncio.CancelledError`) is not recorded.
- A tool that raises during recording raises `ReplayedError` on replay, not the original exception class (the message and type name are kept). The same goes for SDK errors such as `BadRequestError`.
- Tool results are stored as JSON: a tool that returns a custom object comes back as a plain dict on replay.

## Roadmap

- [x] Time-travel viewer in the browser: scrub steps, inspect the context at each one
- [x] Branching: edit a result at step N and re-run from there
- [x] Streaming and async support
- [ ] `flightrec replay module:function` from the CLI

## Develop

```
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"     # Windows; use .venv/bin/pip elsewhere
.venv/Scripts/python -m pytest
```
