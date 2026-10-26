"""Record and replay the nondeterministic edges of an agent: model calls, tool calls, clocks, dice."""

import contextvars
import functools
import inspect
import random as _random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .serialize import Box, digest, rehydrate, to_jsonable, type_path
from .store import Writer, read_events

_UNSET: Any = object()

_current: "contextvars.ContextVar[Optional[Recorder]]" = contextvars.ContextVar(
    "flightrec_current", default=None
)


class ReplayDivergence(Exception):
    """The agent made a call that does not match the recording."""


class ReplayedError(Exception):
    """A call that raised during recording raises this during replay."""

    def __init__(self, type_name: str, message: str):
        super().__init__(f"{type_name}: {message}")
        self.type_name = type_name
        self.message = message


class Recorder:
    """Context manager that records calls to a file, or serves them back from one.

    mode="record": calls run for real and each result is appended to `path`.
    mode="replay": nothing runs; results come from `path` in order, and any call that
    differs from the recording raises ReplayDivergence.
    mode="branch": "what if step `at` had returned something else?" Steps before `at` replay
    from `path`; step `at` returns `response` (or raises `error`); every later step runs live.
    With neither given, step `at` itself runs live. The new run is written to `out`, so it can
    be replayed, viewed and diffed like any other recording. Use a real client and real tools.
    """

    def __init__(
        self,
        path: Union[str, Path],
        mode: str = "record",
        strict: bool = True,
        *,
        at: Optional[int] = None,
        response: Any = _UNSET,
        error: Optional[BaseException] = None,
        out: Union[str, Path, None] = None,
    ):
        if mode not in ("record", "replay", "branch"):
            raise ValueError("mode must be 'record', 'replay' or 'branch'")
        if mode == "branch":
            if at is None or out is None:
                raise ValueError("branch mode needs `at` (step number) and `out` (where to write the new run)")
            if response is not _UNSET and error is not None:
                raise ValueError("give `response` or `error`, not both")
            if Path(out).resolve() == Path(path).resolve():
                raise ValueError("`out` must differ from the recording being branched")
        elif at is not None or response is not _UNSET or error is not None or out is not None:
            raise ValueError("`at`, `response`, `error` and `out` are only for mode='branch'")
        self.path = Path(path)
        self.mode = mode
        self.strict = strict
        self.at = at
        self._override = response
        self._override_error = error
        self.out = Path(out) if out is not None else None
        self._writer: Optional[Writer] = None
        self._events: List[Dict[str, Any]] = []
        self._cursor = 0
        self._token = None

    @property
    def recording(self) -> bool:
        return self.mode == "record"

    def __enter__(self) -> "Recorder":
        if self.recording:
            self._writer = Writer(self.path)
        else:
            self._events = read_events(self.path)
            self._cursor = 0
            if self.mode == "branch":
                if not 0 <= self.at < len(self._events):
                    raise ValueError(f"step {self.at} is outside the recording (it has {len(self._events)} steps)")
                self._writer = Writer(self.out, branch={"source": str(self.path), "at": self.at})
        self._token = _current.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _current.reset(self._token)
        if self._writer:
            self._writer.close()
            self._writer = None
        elif exc_type is None and self.strict and self._cursor < len(self._events):
            left = len(self._events) - self._cursor
            raise ReplayDivergence(f"run ended with {left} recorded call(s) never made")

    def wrap(self, client: Any, methods: Sequence[str] = ("messages.create",), name: str = "llm") -> Any:
        """Proxy a model client so the listed dotted methods are recorded. In replay `client` may be None."""
        return _Proxy(self, client, (), tuple(methods), name)

    def call(self, kind: str, name: str, fn: Callable[[], Any], request: Dict[str, Any], box: bool = False) -> Any:
        request = to_jsonable(request)
        if self.recording:
            return self._record(kind, name, fn, request)
        if self.mode == "branch":
            if self._cursor < self.at:
                event = self._next_event(kind, name, request)
                self._writer.write(event)
                return self._result(event, box)
            if self._cursor == self.at and (self._override is not _UNSET or self._override_error is not None):
                return self._edited(kind, name, request, box)
            return self._record(kind, name, fn, request)
        return self._result(self._next_event(kind, name, request), box)

    def _edited(self, kind: str, name: str, request: Dict[str, Any], box: bool) -> Any:
        """The branch point: check we got here the same way, then answer with the edited result."""
        original = self._next_event(kind, name, request)
        event = dict(original, edited=True, original={"response": original["response"], "error": original["error"]})
        if self._override_error is not None:
            exc = self._override_error
            event.update(response=None, error={"type": type(exc).__name__, "message": str(exc)})
            self._writer.write(event)
            raise exc
        event.update(response=to_jsonable(self._override), error=None)
        self._writer.write(event)
        return self._result(event, box)

    def _record(self, kind: str, name: str, fn: Callable[[], Any], request: Dict[str, Any]) -> Any:
        started = time.time()
        event = {
            "type": "call",
            "seq": self._cursor,
            "kind": kind,
            "name": name,
            "request": request,
            "request_hash": digest(request),
            "started": started,
        }
        self._cursor += 1
        try:
            result = fn()
        except Exception as exc:
            event.update(response=None, error={"type": type(exc).__name__, "message": str(exc)})
            event["duration"] = time.time() - started
            self._writer.write(event)
            raise
        event.update(response=to_jsonable(result), error=None)
        kind_of_object = type_path(result)
        if kind_of_object:
            event["response_type"] = kind_of_object
        event["duration"] = time.time() - started
        self._writer.write(event)
        return result

    def _next_event(self, kind: str, name: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """The recorded event this call must correspond to; advances the cursor."""
        if self._cursor >= len(self._events):
            raise ReplayDivergence(f"step {self._cursor}: {kind} '{name}' called but the recording has ended")
        event = self._events[self._cursor]
        step = self._cursor
        if (event["kind"], event["name"]) != (kind, name):
            raise ReplayDivergence(
                f"step {step}: expected {event['kind']} '{event['name']}', got {kind} '{name}'"
            )
        if event["request_hash"] != digest(request):
            raise ReplayDivergence(f"step {step}: {kind} '{name}' was called with different arguments")
        self._cursor += 1
        return event

    @staticmethod
    def _result(event: Dict[str, Any], box: bool) -> Any:
        if event["error"]:
            raise ReplayedError(event["error"]["type"], event["error"]["message"])
        response = event["response"]
        if box and event.get("response_type"):
            rebuilt = rehydrate(event["response_type"], response)
            if rebuilt is not None:
                return rebuilt
        return Box(response) if box and isinstance(response, (dict, list)) else response


class _Proxy:
    """Transparent proxy that intercepts chosen dotted methods and passes everything else through."""

    def __init__(self, recorder: Recorder, target: Any, path: Tuple[str, ...], methods: Tuple[str, ...], name: str):
        self._recorder = recorder
        self._target = target
        self._path = path
        self._methods = methods
        self._name = name

    def __getattr__(self, attr: str) -> Any:
        path = self._path + (attr,)
        dotted = ".".join(path)
        child = getattr(self._target, attr) if self._target is not None else None
        if dotted in self._methods:
            return self._intercept(dotted, child)
        if any(m.startswith(dotted + ".") for m in self._methods):
            return _Proxy(self._recorder, child, path, self._methods, self._name)
        return child

    def _intercept(self, dotted: str, method: Optional[Callable[..., Any]]) -> Callable[..., Any]:
        def intercepted(*args: Any, **kwargs: Any) -> Any:
            request = dict(kwargs)
            if args:
                request["_args"] = list(args)
            return self._recorder.call(
                "llm", f"{self._name}.{dotted}", lambda: method(*args, **kwargs), request, box=True
            )

        return intercepted


def tool(fn: Optional[Callable[..., Any]] = None, *, name: Optional[str] = None) -> Any:
    """Decorator: record this tool's arguments and result. Outside a Recorder it is a no-op."""

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or func.__name__
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            rec = _current.get()
            if rec is None:
                return func(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            return rec.call("tool", tool_name, lambda: func(*args, **kwargs), dict(bound.arguments))

        return wrapper

    return decorate(fn) if fn is not None else decorate


def capture(name: str, fn: Callable[[], Any], **request: Any) -> Any:
    """Record the result of any nondeterministic call. Outside a Recorder it just calls `fn`."""
    rec = _current.get()
    if rec is None:
        return fn()
    return rec.call("nondeterminism", name, fn, request)


def now() -> float:
    """Wall-clock time that replays to the recorded value."""
    return capture("now", time.time)


def random() -> float:
    """A random float that replays to the recorded value."""
    return capture("random", _random.random)
