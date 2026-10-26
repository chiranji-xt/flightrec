import json
import re

import flightrec
from flightrec.cli import main
from flightrec.viewer import build_html


def _embedded(page: str):
    match = re.search(r'<script id="flightrec-data" type="application/json">(.*?)</script>', page, re.S)
    return json.loads(match.group(1))


def _record(path, text="hello"):
    @flightrec.tool
    def echo(value):
        return value

    with flightrec.Recorder(path):
        echo(text)
    return flightrec.read_events(path)


def test_events_survive_embedding(tmp_path):
    events = _record(tmp_path / "r.jsonl")
    data = _embedded(build_html(events, "r.jsonl"))
    assert data["name"] == "r.jsonl"
    assert data["events"] == events


def test_hostile_text_cannot_break_out_of_the_data_tag(tmp_path):
    nasty = '</script><script>window.pwned=1</script><!--  '
    events = _record(tmp_path / "r.jsonl", nasty)
    page = build_html(events, "r.jsonl")
    assert page.count("</script>") == page.count("<script"), "recorded text closed a script tag early"
    assert _embedded(page)["events"][0]["response"] == nasty


def test_view_command_writes_a_page(tmp_path, capsys):
    rec = tmp_path / "run.jsonl"
    _record(rec)
    assert main(["view", str(rec), "--no-open"]) == 0
    out = tmp_path / "run.html"
    assert out.exists()
    assert _embedded(out.read_text(encoding="utf-8"))["events"][0]["name"] == "echo"
    assert "run.html" in capsys.readouterr().out


def test_version_flag(capsys):
    import pytest

    with pytest.raises(SystemExit) as stop:
        main(["--version"])
    assert stop.value.code == 0
    assert capsys.readouterr().out.strip() == f"flightrec {flightrec.__version__}"
