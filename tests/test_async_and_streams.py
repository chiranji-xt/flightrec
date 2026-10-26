"""Async calls, async tools and streams, using plain Python fakes (no SDK needed)."""

import asyncio
import json
from types import SimpleNamespace

import pytest

import flightrec


class FakeMessages:
    """A model client: create() answers with a dict, or with a stream of chunks if stream=True."""

    def __init__(self):
        self.calls = 0

    def create(self, prompt, stream=False, fail_after=None):
        self.calls += 1
        if not stream:
            return {"text": f"answer to {prompt}"}
        return self._chunks(prompt, fail_after)

    def _chunks(self, prompt, fail_after):
        for i, word in enumerate(f"streamed answer to {prompt}".split()):
            if fail_after is not None and i == fail_after:
                raise ConnectionError("stream dropped")
            yield {"type": "delta", "text": word}


class FakeAsyncMessages:
    def __init__(self):
        self.calls = 0

    async def create(self, prompt, stream=False, fail_after=None):
        self.calls += 1
        await asyncio.sleep(0)
        if not stream:
            return {"text": f"answer to {prompt}"}
        return self._chunks(prompt, fail_after)

    async def _chunks(self, prompt, fail_after):
        for i, word in enumerate(f"streamed answer to {prompt}".split()):
            if fail_after is not None and i == fail_after:
                raise ConnectionError("stream dropped")
            await asyncio.sleep(0)
            yield {"type": "delta", "text": word}


def sync_client():
    return SimpleNamespace(messages=FakeMessages())


def async_client():
    return SimpleNamespace(messages=FakeAsyncMessages())


# --- async calls and tools ---------------------------------------------------------------------


def test_async_calls_and_tools_record_and_replay(tmp_path):
    path = tmp_path / "run.jsonl"
    client = async_client()

    @flightrec.tool
    async def lookup(topic):
        await asyncio.sleep(0)
        return f"notes on {topic}"

    async def agent(llm):
        reply = await llm.messages.create(prompt="hi")
        notes = await lookup("cats")
        return reply["text"], notes

    async def record():
        with flightrec.Recorder(path) as rec:
            return await agent(rec.wrap(client))

    async def replay():
        with flightrec.Recorder(path, mode="replay") as rec:
            return await agent(rec.wrap(None))

    recorded = asyncio.run(record())
    events = flightrec.read_events(path)
    assert [e.get("async") for e in events] == [True, True]
    assert asyncio.run(replay()) == recorded
    assert client.messages.calls == 1, "replay must not call the client"


def test_async_error_is_recorded_and_replayed_at_the_await(tmp_path):
    path = tmp_path / "run.jsonl"

    @flightrec.tool
    async def broken():
        raise TimeoutError("slow")

    async def record():
        with flightrec.Recorder(path):
            with pytest.raises(TimeoutError):
                await broken()

    async def replay():
        with flightrec.Recorder(path, mode="replay"):
            with pytest.raises(flightrec.ReplayedError, match="TimeoutError: slow"):
                await broken()

    asyncio.run(record())
    asyncio.run(replay())


def test_async_with_recorder(tmp_path):
    path = tmp_path / "run.jsonl"

    @flightrec.tool
    async def one():
        return 1

    async def run(mode):
        async with flightrec.Recorder(path, mode=mode):
            return await one()

    assert asyncio.run(run("record")) == asyncio.run(run("replay")) == 1


def test_concurrent_calls_replay_in_the_order_they_started(tmp_path):
    path = tmp_path / "run.jsonl"

    @flightrec.tool
    async def work(name, delay):
        await asyncio.sleep(delay)
        return name.upper()

    async def agent():
        # "slow" starts first but finishes last, so it is written to the file last.
        return await asyncio.gather(work("slow", 0.05), work("fast", 0))

    async def record():
        with flightrec.Recorder(path):
            return await agent()

    async def replay():
        with flightrec.Recorder(path, mode="replay"):
            return await agent()

    recorded = asyncio.run(record())
    assert [e["request"]["name"] for e in flightrec.read_events(path)] == ["slow", "fast"]
    assert recorded == ["SLOW", "FAST"]
    assert asyncio.run(replay()) == recorded


# --- streams -----------------------------------------------------------------------------------


def test_sync_stream_records_chunks_and_replays_them(tmp_path):
    path = tmp_path / "run.jsonl"
    client = sync_client()
    with flightrec.Recorder(path) as rec:
        live = [c["text"] for c in rec.wrap(client).messages.create(prompt="x", stream=True)]

    event = flightrec.read_events(path)[0]
    assert event["stream"] == "events" and event["async_stream"] is False
    assert [c["text"] for c in event["response"]] == live == ["streamed", "answer", "to", "x"]

    with flightrec.Recorder(path, mode="replay") as rec:
        replayed = [c.text for c in rec.wrap(None).messages.create(prompt="x", stream=True)]
    assert replayed == live
    assert client.messages.calls == 1


def test_async_stream_records_chunks_and_replays_them(tmp_path):
    path = tmp_path / "run.jsonl"
    client = async_client()

    async def consume(llm):
        stream = await llm.messages.create(prompt="x", stream=True)
        return [c["text"] async for c in stream]

    async def record():
        with flightrec.Recorder(path) as rec:
            return await consume(rec.wrap(client))

    async def replay():
        with flightrec.Recorder(path, mode="replay") as rec:
            return await consume(rec.wrap(None))

    live = asyncio.run(record())
    event = flightrec.read_events(path)[0]
    assert event["async"] is True and event["async_stream"] is True
    assert asyncio.run(replay()) == live == ["streamed", "answer", "to", "x"]
    assert client.messages.calls == 1


def test_stream_closed_early_is_marked_incomplete_and_replays_what_was_seen(tmp_path):
    path = tmp_path / "run.jsonl"
    with flightrec.Recorder(path) as rec:
        stream = rec.wrap(sync_client()).messages.create(prompt="x", stream=True)
        first = next(stream)
        stream.close()

    event = flightrec.read_events(path)[0]
    assert event["incomplete"] is True
    assert len(event["response"]) == 1
    with flightrec.Recorder(path, mode="replay") as rec:
        assert [c.text for c in rec.wrap(None).messages.create(prompt="x", stream=True)] == [first["text"]]


def test_stream_nobody_closed_is_still_written_when_the_recorder_exits(tmp_path):
    path = tmp_path / "run.jsonl"
    with flightrec.Recorder(path) as rec:
        stream = rec.wrap(sync_client()).messages.create(prompt="x", stream=True)
        next(stream)  # abandoned mid-way

    event = flightrec.read_events(path)[0]
    assert event["incomplete"] is True and len(event["response"]) == 1


def test_error_in_the_middle_of_a_stream_replays_after_the_chunks(tmp_path):
    path = tmp_path / "run.jsonl"
    seen = []
    with flightrec.Recorder(path) as rec:
        with pytest.raises(ConnectionError):
            for chunk in rec.wrap(sync_client()).messages.create(prompt="x", stream=True, fail_after=2):
                seen.append(chunk["text"])
    assert seen == ["streamed", "answer"]

    replayed = []
    with flightrec.Recorder(path, mode="replay") as rec:
        with pytest.raises(flightrec.ReplayedError, match="ConnectionError: stream dropped"):
            for chunk in rec.wrap(None).messages.create(prompt="x", stream=True, fail_after=2):
                replayed.append(chunk.text)
    assert replayed == seen


def test_calls_made_while_a_stream_is_open_keep_their_order(tmp_path):
    path = tmp_path / "run.jsonl"

    @flightrec.tool
    def note(text):
        return text

    def agent(llm):
        out = []
        for chunk in llm.messages.create(prompt="x", stream=True):
            out.append(note(chunk["text"] if isinstance(chunk, dict) else chunk.text))
        return out

    with flightrec.Recorder(path) as rec:
        live = agent(rec.wrap(sync_client()))
    # the stream's event is written last (it ends last) but belongs first
    in_file = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[1:]]
    assert [e["kind"] for e in in_file] == ["tool"] * 4 + ["llm"]
    assert [e["kind"] for e in flightrec.read_events(path)] == ["llm"] + ["tool"] * 4
    with flightrec.Recorder(path, mode="replay") as rec:
        assert agent(rec.wrap(None)) == live


def test_branch_replays_a_recorded_stream_then_goes_live(tmp_path):
    source, out = tmp_path / "source.jsonl", tmp_path / "branch.jsonl"
    client = sync_client()
    with flightrec.Recorder(source) as rec:
        llm = rec.wrap(client)
        list(llm.messages.create(prompt="x", stream=True))
        llm.messages.create(prompt="second")
    calls_after_record = client.messages.calls

    with flightrec.Recorder(source, mode="branch", at=1, response={"text": "edited"}, out=out) as rec:
        llm = rec.wrap(client)
        assert len(list(llm.messages.create(prompt="x", stream=True))) == 4  # replayed
        assert llm.messages.create(prompt="second")["text"] == "edited"
    assert client.messages.calls == calls_after_record, "nothing should have run live"
    assert [e.get("edited") for e in flightrec.read_events(out)] == [None, True]


def test_branch_can_replace_a_streams_chunks(tmp_path):
    source, out = tmp_path / "source.jsonl", tmp_path / "branch.jsonl"
    with flightrec.Recorder(source) as rec:
        list(rec.wrap(sync_client()).messages.create(prompt="x", stream=True))

    with pytest.raises(ValueError, match="list of chunks"):
        with flightrec.Recorder(source, mode="branch", at=0, response="nope", out=out) as rec:
            list(rec.wrap(sync_client()).messages.create(prompt="x", stream=True))

    with flightrec.Recorder(source, mode="branch", at=0, response=[{"text": "only"}], out=out) as rec:
        chunks = list(rec.wrap(sync_client()).messages.create(prompt="x", stream=True))
    assert [c.text for c in chunks] == ["only"]
    event = flightrec.read_events(out)[0]
    assert event["edited"] is True and len(event["original"]["response"]) == 4


def test_plain_dicts_and_lists_are_not_mistaken_for_streams(tmp_path):
    path = tmp_path / "run.jsonl"
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **_: [1, 2, 3]))
    with flightrec.Recorder(path) as rec:
        assert rec.wrap(client).messages.create(a=1) == [1, 2, 3]
    assert "stream" not in flightrec.read_events(path)[0]


def test_recording_file_is_valid_json_lines_with_streams(tmp_path):
    path = tmp_path / "run.jsonl"
    with flightrec.Recorder(path) as rec:
        list(rec.wrap(sync_client()).messages.create(prompt="x", stream=True))
    for line in path.read_text(encoding="utf-8").splitlines():
        json.loads(line)
