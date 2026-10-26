import sys
from pathlib import Path

import pytest

import flightrec
from flightrec.cli import diff
from flightrec.store import read_header

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import demo_agent  # noqa: E402

TASK = "What does flightrec do?"


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.jsonl"
    with flightrec.Recorder(path) as rec:
        demo_agent.run_agent(rec.wrap(demo_agent.FakeClient()), TASK)
    return path


def branch(source, tmp_path, **kwargs):
    out = tmp_path / "branch.jsonl"
    with flightrec.Recorder(source, mode="branch", out=out, **kwargs) as rec:
        answer = demo_agent.run_agent(rec.wrap(demo_agent.FakeClient()), TASK)
    return answer, out


def test_edited_result_flows_into_later_live_calls(source, tmp_path):
    answer, out = branch(source, tmp_path, at=1, response="the docs were rewritten")

    assert answer == "Answer: the docs were rewritten"
    events = flightrec.read_events(out)
    assert [e["seq"] for e in events] == [0, 1, 2]
    assert events[1]["edited"] is True
    assert events[1]["original"]["response"].startswith("flightrec records")
    assert "the docs were rewritten" in str(events[2]["request"]), "the later model call must see the edit"
    assert read_header(out)["branch"] == {"source": str(source), "at": 1}


def test_branch_file_replays_like_any_recording(source, tmp_path):
    answer, out = branch(source, tmp_path, at=1, response="edited")
    with flightrec.Recorder(out, mode="replay") as rec:
        assert demo_agent.run_agent(rec.wrap(None), TASK) == answer


def test_diff_points_at_the_branch_step(source, tmp_path, capsys):
    _, out = branch(source, tmp_path, at=1, response="edited")
    assert diff(str(source), str(out)) == 1
    assert "step 1: same request, different result" in capsys.readouterr().out


def test_branch_without_an_edit_reruns_that_step_live(source, tmp_path):
    answer, out = branch(source, tmp_path, at=1)
    original = flightrec.read_events(source)
    assert [e["response"] for e in flightrec.read_events(out)] == [e["response"] for e in original]
    assert "edited" not in flightrec.read_events(out)[1]
    assert answer.startswith("Answer: flightrec records")


def test_injected_error_reaches_the_agent(source, tmp_path):
    out = tmp_path / "branch.jsonl"
    with flightrec.Recorder(source, mode="branch", at=1, error=TimeoutError("search down"), out=out) as rec:
        with pytest.raises(TimeoutError, match="search down"):
            demo_agent.run_agent(rec.wrap(demo_agent.FakeClient()), TASK)
    event = flightrec.read_events(out)[1]
    assert event["error"] == {"type": "TimeoutError", "message": "search down"}
    assert event["edited"] is True


def test_prefix_must_still_match_the_recording(source, tmp_path):
    with pytest.raises(flightrec.ReplayDivergence, match="step 0"):
        with flightrec.Recorder(source, mode="branch", at=1, response="x", out=tmp_path / "b.jsonl") as rec:
            demo_agent.run_agent(rec.wrap(demo_agent.FakeClient()), "a different task")


def test_branching_a_branch(source, tmp_path):
    _, first = branch(source, tmp_path, at=1, response="first edit")
    out = tmp_path / "second.jsonl"
    with flightrec.Recorder(first, mode="branch", at=1, response="second edit", out=out) as rec:
        answer = demo_agent.run_agent(rec.wrap(demo_agent.FakeClient()), TASK)
    assert answer == "Answer: second edit"
    assert flightrec.read_events(out)[1]["original"]["response"] == "first edit"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"at": 1}, "needs `at`"),
        ({"out": "x.jsonl"}, "needs `at`"),
        ({"at": 1, "out": "x.jsonl", "response": 1, "error": ValueError()}, "not both"),
    ],
)
def test_bad_branch_arguments(source, kwargs, message):
    with pytest.raises(ValueError, match=message):
        flightrec.Recorder(source, mode="branch", **kwargs)


def test_branch_options_rejected_in_other_modes(source):
    with pytest.raises(ValueError, match="only for mode='branch'"):
        flightrec.Recorder(source, mode="replay", at=1)


def test_step_out_of_range(source, tmp_path):
    with pytest.raises(ValueError, match="outside the recording"):
        with flightrec.Recorder(source, mode="branch", at=99, out=tmp_path / "b.jsonl"):
            pass


def test_out_may_not_overwrite_the_source(source):
    with pytest.raises(ValueError, match="must differ"):
        flightrec.Recorder(source, mode="branch", at=1, out=source)
