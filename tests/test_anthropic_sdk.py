"""flightrec against the real Anthropic SDK.

The SDK client is real: requests are built and responses parsed by the SDK into its own pydantic
types. Only the network is replaced, by an httpx2 MockTransport that plays a tiny scripted API, so
these tests need no API key and cost nothing. A live test at the bottom runs only when asked for.
"""

import json
import os

import pytest

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

import flightrec  # noqa: E402
from flightrec.cli import diff  # noqa: E402

MODEL = "claude-sonnet-5-5"
TOOLS = [
    {
        "name": "get_weather",
        "description": "Current weather for a city",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }
]


def _message(content, stop_reason):
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 7},
    }


class FakeApi:
    """A scripted Messages API: asks for the weather tool first, then answers from its result."""

    def __init__(self):
        self.requests = []

    def __call__(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        if body["messages"][0]["content"] == "bad request":
            return httpx2.Response(
                400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "nope"}}
            )
        last = body["messages"][-1]
        if isinstance(last["content"], list) and last["content"][0]["type"] == "tool_result":
            text = "The weather is: " + last["content"][0]["content"]
            return httpx2.Response(200, json=_message([{"type": "text", "text": text}], "end_turn"))
        block = {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "Paris"}}
        return httpx2.Response(200, json=_message([{"type": "text", "text": "Let me check."}, block], "tool_use"))


@pytest.fixture
def api():
    return FakeApi()


@pytest.fixture
def client(api):
    http = httpx2.Client(transport=httpx2.MockTransport(api))
    return anthropic.Anthropic(api_key="test-key", http_client=http, max_retries=0)


@flightrec.tool
def get_weather(city: str) -> str:
    return f"sunny in {city}"


def run_agent(client, task, tools=None):
    """The usual agent loop, written the way SDK docs write it: it hands response objects back."""
    messages = [{"role": "user", "content": task}]
    while True:
        response = client.messages.create(model=MODEL, max_tokens=256, tools=TOOLS, messages=messages)
        if response.stop_reason != "tool_use":
            return "".join(block.text for block in response.content if block.type == "text")
        messages.append({"role": "assistant", "content": response.content})
        results = [
            {"type": "tool_result", "tool_use_id": block.id, "content": (tools or get_weather)(**block.input)}
            for block in response.content
            if block.type == "tool_use"
        ]
        messages.append({"role": "user", "content": results})


@pytest.fixture
def recording(tmp_path, client):
    path = tmp_path / "run.jsonl"
    with flightrec.Recorder(path) as rec:
        answer = run_agent(rec.wrap(client), "Weather in Paris?")
    assert answer == "The weather is: sunny in Paris"
    return path


def test_records_real_sdk_calls(recording):
    events = flightrec.read_events(recording)
    assert [(e["kind"], e["name"]) for e in events] == [
        ("llm", "llm.messages.create"),
        ("tool", "get_weather"),
        ("llm", "llm.messages.create"),
    ]
    assert events[0]["request"]["tools"] == TOOLS
    assert events[0]["response"]["stop_reason"] == "tool_use"
    assert events[0]["response"]["content"][1]["input"] == {"city": "Paris"}
    assert events[2]["response"]["usage"]["output_tokens"] == 7


def test_replay_makes_no_http_requests(recording, api, client):
    made = len(api.requests)
    with flightrec.Recorder(recording, mode="replay") as rec:
        answer = run_agent(rec.wrap(None), "Weather in Paris?")
    assert answer == "The weather is: sunny in Paris"
    assert len(api.requests) == made


def test_replayed_responses_are_real_sdk_objects(recording):
    with flightrec.Recorder(recording, mode="replay", strict=False) as rec:
        llm = rec.wrap(None)
        messages = [{"role": "user", "content": "Weather in Paris?"}]
        response = llm.messages.create(model=MODEL, max_tokens=256, tools=TOOLS, messages=messages)
    assert isinstance(response, anthropic.types.Message)
    assert isinstance(response.content[1], anthropic.types.ToolUseBlock)
    assert isinstance(response.content[1].input, dict)
    assert response.usage.input_tokens == 12


def test_branch_edit_flows_through_the_real_client(recording, api, client, tmp_path):
    out = tmp_path / "branch.jsonl"
    before = len(api.requests)
    with flightrec.Recorder(recording, mode="branch", at=1, response="a hailstorm in Paris", out=out) as rec:
        answer = run_agent(rec.wrap(client), "Weather in Paris?")

    assert answer == "The weather is: a hailstorm in Paris"
    assert len(api.requests) == before + 1, "only the step after the edit should hit the API"
    sent = api.requests[-1]["messages"][-1]["content"][0]
    assert sent["content"] == "a hailstorm in Paris"
    assert diff(str(recording), str(out)) == 1

    with flightrec.Recorder(out, mode="replay") as rec:
        assert run_agent(rec.wrap(None), "Weather in Paris?") == answer


def test_branch_error_injection(recording, client, tmp_path):
    @flightrec.tool(name="get_weather")
    def flaky(city):
        raise AssertionError("must not run: the branch point replaces this call")

    out = tmp_path / "branch.jsonl"
    with flightrec.Recorder(recording, mode="branch", at=1, error=TimeoutError("weather api down"), out=out) as rec:
        with pytest.raises(TimeoutError, match="weather api down"):
            run_agent(rec.wrap(client), "Weather in Paris?", tools=flaky)


def test_sdk_errors_are_recorded_and_replayed(tmp_path, client):
    path = tmp_path / "err.jsonl"
    request = dict(model=MODEL, max_tokens=16, messages=[{"role": "user", "content": "bad request"}])
    with flightrec.Recorder(path) as rec:
        with pytest.raises(anthropic.BadRequestError):
            rec.wrap(client).messages.create(**request)

    event = flightrec.read_events(path)[0]
    assert event["error"]["type"] == "BadRequestError"
    with flightrec.Recorder(path, mode="replay") as rec:
        with pytest.raises(flightrec.ReplayedError, match="BadRequestError"):
            rec.wrap(None).messages.create(**request)


def test_other_client_attributes_pass_through(client):
    with flightrec.Recorder("unused.jsonl", mode="record") as rec:
        wrapped = rec.wrap(client)
        assert wrapped.base_url == client.base_url
        assert wrapped.with_options(timeout=5) is not None


@pytest.mark.skipif(
    not (os.environ.get("FLIGHTREC_LIVE") == "1" and os.environ.get("ANTHROPIC_API_KEY")),
    reason="live API test: set FLIGHTREC_LIVE=1 and ANTHROPIC_API_KEY (makes one tiny paid request)",
)
def test_live_api_record_then_replay(tmp_path):
    path = tmp_path / "live.jsonl"
    request = dict(model="claude-haiku-5-5", max_tokens=16, messages=[{"role": "user", "content": "Say ok."}])
    with flightrec.Recorder(path) as rec:
        live = rec.wrap(anthropic.Anthropic()).messages.create(**request)
    with flightrec.Recorder(path, mode="replay") as rec:
        replayed = rec.wrap(None).messages.create(**request)
    assert isinstance(replayed, anthropic.types.Message)
    assert replayed.content[0].text == live.content[0].text
    assert replayed.usage.output_tokens == live.usage.output_tokens
