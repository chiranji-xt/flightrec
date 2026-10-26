"""Build a standalone HTML page that shows a recording."""

import json
from pathlib import Path
from typing import Any, Dict, List

from .store import read_events

_TEMPLATE = Path(__file__).with_name("viewer.html")
_SLOT = '<script id="flightrec-data" type="application/json">null</script>'


def build_html(events: List[Dict[str, Any]], name: str) -> str:
    """The viewer page with `events` embedded, so it opens with no server and no upload step."""
    page = _TEMPLATE.read_text(encoding="utf-8")
    if _SLOT not in page:
        raise RuntimeError("viewer.html is missing its data slot")
    payload = json.dumps({"name": name, "events": events}, ensure_ascii=False)
    # Keep recorded text from closing the script tag or starting an HTML comment.
    payload = payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return page.replace(_SLOT, f'<script id="flightrec-data" type="application/json">{payload}</script>')


def write_viewer(recording: Path, out: Path) -> Path:
    out.write_text(build_html(read_events(recording), recording.name), encoding="utf-8")
    return out
