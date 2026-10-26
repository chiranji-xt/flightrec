import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import flightrec
from flightrec.cli import diff, first_divergence, show

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import demo_agent  # noqa: E402


def test_record_then_replay_offline(tmp_path):
    path = tmp_path / "run.jsonl"
    with flightrec.Recorder(path) as rec:
        recorded = demo_agent.run_agent(rec.wrap(demo_agent.FakeClient()), "What does flightrec do?")

    with flightrec.Recorder(path, mode="replay") as rec:
        replayed = demo_agent.run_agent(rec.wrap(None), "What does flightrec do?")

    assert replayed == recorded
    assert "records model and tool calls" in replayed


def test_replay_never_calls_the_real_client(tmp_path):
    path = tmp_path / "run.jsonl"
    with flightrec.Recorder(path) as rec:
        demo_agent.run_agent(rec.wrap(demo_agent.FakeClient()), "q")

    class Exploding:
        class messages:
            @staticmethod
            def create(**_):
                raise AssertionError("real client was called during replay")

    with flightrec.Recorder(path, mode="replay") as rec:
        demo_agent.run_agent(rec.wrap(Exploding()), "q")


def test_changed_request_raises_divergence(tmp_path):
    path = tmp_path / "run.jsonl"
    with flightrec.Recorder(path) as rec:
        demo_agent.run_agent(rec.wrap(demo_agent.FakeClient()), "task one")

    with pytest.raises(flightrec.ReplayDivergence, match="step 0"):
        with flightrec.Recorder(path, mode="replay") as rec:
            demo_agent.run_agent(rec.wrap(None), "a different task")


def test_unused_recorded_calls_are_reported(tmp_path):
    path = tmp_path / "run.jsonl"
    with flightrec.Recorder(path) as rec:
        client = rec.wrap(demo_agent.FakeClient())
        demo_agent.run_agent(client, "q")

    with pytest.raises(flightrec.ReplayDivergence, match="never made"):
        with flightrec.Recorder(path, mode="replay"):
            pass


def test_errors_are_recorded_and_replayed(tmp_path):
    path = tmp_path / "run.jsonl"

    @flightrec.tool
    def flaky(x):
        raise TimeoutError("upstream timed out")

    with flightrec.Recorder(path):
        with pytest.raises(TimeoutError):
            flaky(1)

    with flightrec.Recorder(path, mode="replay"):
        with pytest.raises(flightrec.ReplayedError, match="TimeoutError: upstream timed out"):
            flaky(1)


def test_clock_and_randomness_replay_to_recorded_values(tmp_path):
    path = tmp_path / "run.jsonl"
    with flightrec.Recorder(path):
        t, r = flightrec.now(), flightrec.random()
    with flightrec.Recorder(path, mode="replay"):
        assert (flightrec.now(), flightrec.random()) == (t, r)


def test_tool_is_a_noop_outside_a_recorder():
    assert demo_agent.search("flightrec").startswith("flightrec records")


def test_replayed_responses_support_attribute_access(tmp_path):
    path = tmp_path / "run.jsonl"

    class Messages:
        def create(self, **_):
            return SimpleNamespace(content=[SimpleNamespace(text="hi")], usage=SimpleNamespace(tokens=3))

    with flightrec.Recorder(path) as rec:
        rec.wrap(SimpleNamespace(messages=Messages())).messages.create(model="m")

    with flightrec.Recorder(path, mode="replay") as rec:
        resp = rec.wrap(None).messages.create(model="m")
    assert resp.content[0].text == "hi"
    assert resp.usage.tokens == 3


def test_diff_finds_first_divergence(tmp_path, capsys):
    good, bad = tmp_path / "good.jsonl", tmp_path / "bad.jsonl"
    with flightrec.Recorder(good) as rec:
        demo_agent.run_agent(rec.wrap(demo_agent.FakeClient()), "q")
    with flightrec.Recorder(bad) as rec:
        truncated = flightrec.tool(demo_agent.search_truncated, name="search")
        demo_agent.run_agent(rec.wrap(demo_agent.FakeClient()), "q", search_tool=truncated)

    assert diff(str(good), str(good)) == 0
    assert diff(str(good), str(bad)) == 1
    out = capsys.readouterr().out
    assert "step 1: same request, different result" in out

    events = flightrec.read_events(good)
    assert first_divergence(events, events) is None


def test_show_prints_timeline(tmp_path, capsys):
    path = tmp_path / "run.jsonl"
    with flightrec.Recorder(path) as rec:
        demo_agent.run_agent(rec.wrap(demo_agent.FakeClient()), "q")
    show(str(path))
    out = capsys.readouterr().out
    assert "3 call(s)" in out
    assert "tool" in out and "llm.messages.create" in out
