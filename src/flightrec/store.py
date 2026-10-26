"""Recording file format: one JSON object per line, flushed per event so crashes keep the trace."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List

FORMAT_VERSION = 1


class Writer:
    def __init__(self, path: Path, **header: Any):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
        self.write({"type": "header", "version": FORMAT_VERSION, "created": time.time(), **header})

    def write(self, record: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def read_header(path: Path) -> Dict[str, Any]:
    """The first line of a recording (empty dict if the file is empty)."""
    with open(path, "r", encoding="utf-8") as fh:
        first = fh.readline().strip()
    return json.loads(first) if first else {}


def read_events(path: Path) -> List[Dict[str, Any]]:
    """All call events from a recording, in order (the header is skipped)."""
    events = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") == "header":
                if record.get("version") != FORMAT_VERSION:
                    raise ValueError(f"unsupported recording version {record.get('version')}")
                continue
            events.append(record)
    return events
