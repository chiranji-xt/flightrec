"""Streaming and async against the real Anthropic SDK (network faked with server-sent events)."""

import asyncio
import json

import pytest

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

import flightrec  # noqa: E402

MODEL = "claude-sonnet-5-5"
WORDS = ["Hel", "lo ", "there"]


def _sse(events):
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


def _text_stream(words, fail_after=None):
    message = {
        "id": "msg_1", "type": "message", "role": "assistant", "model": MODEL, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 5, "output_tokens": 1},
    }
    events = [
        ("message_start", {"type": "message_start", "message": message}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    ]
    for i, word in enumerate(words):
        if fail_after is not None and i == fail_after:
            events.append(("error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}))
            return _sse(events)
        events.append(("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": word}}))
    events += [
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": len(words)}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return _sse(events)


class FakeApi:
    def __init__(self):
        self.requests = []

    def __call__(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        prompt = body["messages"][-1]["content"]
        fail_after = 1 if prompt == "overload" else None
        return httpx2.Response(200, content=_text_stream(WORDS, fail_after), headers={"content-type": "text/event-stream"})


@pytest.fixture
def api():
    return FakeApi()


@pytest.fixture
def client(api):
    return anthropic.Anthropic(api_key="k", http_client=httpx2.Client(transport=httpx2.MockTransport(api)), max_retries=0)


@pytest.fixture
def aclient(api):
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(api))
    return anthropic.AsyncAnthropic(api_key="k", http_client=http, max_retries=0)


def request(prompt="hi"):
    return dict(model=MODEL, max_tokens=64, messages=[{"role": "user", "content": prompt}])


# --- create(stream=True) -------------------------------------------------------------------------


def test_create_stream_records_and_replays_real_event_types(tmp_path, client, api):
    path = tmp_path / "run.jsonl"
    with flightrec.Recorder(path) as rec:
        live = list(rec.wrap(client).messages.create(stream=True, **request()))
    made = len(api.requests)

    event = flightrec.read_events(path)[0]
    assert event["stream"] == "events" and len(event["response"]) == len(live) == 8  # start, block start, 3 deltas, block stop, delta, stop

    with flightrec.Recorder(path, mode="replay") as rec:
        replayed = list(rec.wrap(None).messages.create(stream=True, **request()))
    assert len(api.requests) == made
    assert [type(e) for e in replayed] == [type(e) for e in live]
    assert isinstance(replayed[2], anthropic.types.RawContentBlockDeltaEvent)
    assert "".join(e.delta.text for e in replayed if e.type == "content_block_delta") == "Hello there"
    assert replayed == live


def test_async_create_stream(tmp_path, aclient, api):
    path = tmp_path / "run.jsonl"

    async def consume(llm):
        stream = await llm.messages.create(stream=True, **request())
        return [e async for e in stream]

    async def record():
        with flightrec.Recorder(path) as rec:
            return await consume(rec.wrap(aclient))

    async def replay():
        with flightrec.Recorder(path, mode="replay") as rec:
            return await consume(rec.wrap(None))

    live = asyncio.run(record())
    made = len(api.requests)
    event = flightrec.read_events(path)[0]
    assert event["async"] is True and event["async_stream"] is True
    replayed = asyncio.run(replay())
    assert len(api.requests) == made
    assert replayed == live


# --- messages.stream() helper ----------------------------------------------------------------------


def test_helper_text_stream_and_final_message(tmp_path, client, api):
    path = tmp_path / "run.jsonl"

    def run(llm):
        with llm.messages.stream(**request()) as stream:
            texts = list(stream.text_stream)
            return texts, stream.get_final_message(), stream.get_final_text()

    with flightrec.Recorder(path) as rec:
        live = run(rec.wrap(client))
    made = len(api.requests)
    event = flightrec.read_events(path)[0]
    assert event["stream"] == "manager" and event["final"]["content"][0]["text"] == "Hello there"

    with flightrec.Recorder(path, mode="replay") as rec:
        replayed = run(rec.wrap(None))
    assert len(api.requests) == made
    assert replayed[0] == live[0] == WORDS
    assert replayed[2] == "Hello there"
    assert isinstance(replayed[1], anthropic.types.Message)
    assert replayed[1] == live[1]


def test_helper_event_iteration_then_final_message(tmp_path, client):
    path = tmp_path / "run.jsonl"

    def run(llm):
        with llm.messages.stream(**request()) as stream:
            kinds = [e.type for e in stream]
            return kinds, stream.get_final_message().usage.output_tokens

    with flightrec.Recorder(path) as rec:
        live = run(rec.wrap(client))
    with flightrec.Recorder(path, mode="replay") as rec:
        assert run(rec.wrap(None)) == live
    assert "text" in live[0]


def test_helper_final_message_alone_still_records_the_events(tmp_path, client):
    """Code that never iterates, and only asks for the final message, is recorded just the same."""
    path = tmp_path / "run.jsonl"
    with flightrec.Recorder(path) as rec:
        with rec.wrap(client).messages.stream(**request()) as stream:
            stream.get_final_message()
    assert len(flightrec.read_events(path)[0]["response"]) > 3


def test_async_helper_stream(tmp_path, aclient, api):
    path = tmp_path / "run.jsonl"

    async def run(llm):
        async with llm.messages.stream(**request()) as stream:
            texts = [t async for t in stream.text_stream]
            final = await stream.get_final_message()
            return texts, final, await stream.get_final_text()

    async def record():
        with flightrec.Recorder(path) as rec:
            return await run(rec.wrap(aclient))

    async def replay():
        with flightrec.Recorder(path, mode="replay") as rec:
            return await run(rec.wrap(None))

    live = asyncio.run(record())
    made = len(api.requests)
    event = flightrec.read_events(path)[0]
    assert event["stream"] == "manager" and event["async_stream"] is True
    replayed = asyncio.run(replay())
    assert len(api.requests) == made
    assert replayed[0] == live[0] == WORDS and replayed[2] == "Hello there"
    assert isinstance(replayed[1], anthropic.types.Message)


# --- failures and partial streams -------------------------------------------------------------------


def test_error_event_mid_stream_replays_after_the_chunks(tmp_path, client):
    path = tmp_path / "run.jsonl"

    def run(llm, exc):
        seen = []
        with pytest.raises(exc) as info:
            with llm.messages.stream(**request("overload")) as stream:
                for text in stream.text_stream:
                    seen.append(text)
        return seen, info.value

    with flightrec.Recorder(path) as rec:
        live_seen, error = run(rec.wrap(client), anthropic.APIStatusError)
    event = flightrec.read_events(path)[0]
    assert event["error"]["type"] == type(error).__name__
    assert live_seen == [WORDS[0]]

    with flightrec.Recorder(path, mode="replay") as rec:
        replayed_seen, _ = run(rec.wrap(None), flightrec.ReplayedError)
    assert replayed_seen == live_seen


def test_leaving_the_with_block_early_records_a_partial_stream(tmp_path, client):
    path = tmp_path / "run.jsonl"
    with flightrec.Recorder(path) as rec:
        with rec.wrap(client).messages.stream(**request()) as stream:
            first = next(iter(stream.text_stream))
    event = flightrec.read_events(path)[0]
    assert event["incomplete"] is True and first == WORDS[0]

    with flightrec.Recorder(path, mode="replay") as rec:
        with rec.wrap(None).messages.stream(**request()) as stream:
            assert next(iter(stream.text_stream)) == first


# --- branching across streams -----------------------------------------------------------------------


@flightrec.tool
def shout(text):
    return text.upper()


def stream_then_tool(llm):
    with llm.messages.stream(**request()) as stream:
        text = stream.get_final_text()
    return shout(text)


def test_branch_replays_the_stream_and_edits_the_next_step(tmp_path, client, api):
    source, out = tmp_path / "source.jsonl", tmp_path / "branch.jsonl"
    with flightrec.Recorder(source) as rec:
        assert stream_then_tool(rec.wrap(client)) == "HELLO THERE"
    made = len(api.requests)

    with flightrec.Recorder(source, mode="branch", at=1, response="EDITED", out=out) as rec:
        assert stream_then_tool(rec.wrap(client)) == "EDITED"
    assert len(api.requests) == made, "the stream before the branch point must replay, not hit the API"

    with flightrec.Recorder(out, mode="replay") as rec:
        assert stream_then_tool(rec.wrap(None)) == "EDITED"


def test_branch_reruns_a_stream_live_after_the_branch_point(tmp_path, client, api):
    source, out = tmp_path / "source.jsonl", tmp_path / "branch.jsonl"
    with flightrec.Recorder(source) as rec:
        stream_then_tool(rec.wrap(client))
    made = len(api.requests)

    with flightrec.Recorder(source, mode="branch", at=0, out=out) as rec:  # no edit: run step 0 live
        stream_then_tool(rec.wrap(client))
    assert len(api.requests) == made + 1
    event = flightrec.read_events(out)[0]
    assert event["stream"] == "manager" and "edited" not in event
