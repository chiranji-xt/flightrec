"""Streaming: record chunks as they flow through a live stream, and serve them back on replay.

A streamed call is still one event in the recording. Its `response` is the list of chunks, written
when the stream ends (or is closed early, or the Recorder exits). Chunks are captured at the
source, so a run is recorded the same however the code consumed it: iterating events, reading
`text_stream`, or asking for the final message.
"""

import inspect
import time
from typing import Any, Callable, Dict, List, Optional

from .serialize import Box, rehydrate, to_jsonable, type_path


class Capture:
    """The chunks of one streamed call, and the event they will be written into."""

    def __init__(self, recorder: Any, event: Dict[str, Any], started: float, final_fn: Optional[Callable[[], Any]] = None):
        self.recorder = recorder
        self.event = event
        self.started = started
        self.final_fn = final_fn
        self.chunks: List[Any] = []
        self.ids: List[int] = []
        self.types: List[str] = []
        self.exhausted = False
        self.done = False
        event["error"] = None
        recorder._open_streams.append(self)

    def add(self, chunk: Any) -> None:
        self.chunks.append(to_jsonable(chunk))
        path = type_path(chunk)
        if path is None:
            self.ids.append(-1)
            return
        if path not in self.types:
            self.types.append(path)
        self.ids.append(self.types.index(path))

    def finish(self, error: Optional[BaseException] = None, incomplete: bool = False) -> None:
        if self.done:
            return
        self.done = True
        event = self.event
        event["response"] = self.chunks
        if self.types:
            event["types"] = self.types
            event["response_type"] = self.ids
        if error is not None:
            event["error"] = {"type": type(error).__name__, "message": str(error)}
        if incomplete and not self.exhausted:
            event["incomplete"] = True
        final = self._final()
        if final is not None:
            event["final"] = to_jsonable(final)
            final_type = type_path(final)
            if final_type:
                event["final_type"] = final_type
        event["duration"] = time.time() - self.started
        self.recorder._stream_finished(self)

    def _final(self) -> Any:
        if self.final_fn is None:
            return None
        try:
            return self.final_fn()
        except Exception:
            return None

    def abandon(self) -> None:
        """Called when a wrapper is garbage collected without the stream having been closed."""
        try:
            self.finish(incomplete=True)
        except Exception:
            pass


class _Recording:
    """Shared behaviour of the recording wrappers: delegate everything we do not intercept."""

    def __init__(self, inner: Any, capture: Capture):
        self._inner = inner
        self._capture = capture

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def __del__(self) -> None:
        capture = self.__dict__.get("_capture")
        if capture is not None:
            capture.abandon()


class RecordingStream(_Recording):
    def __init__(self, inner: Any, capture: Capture):
        super().__init__(inner, capture)
        self._it = iter(inner)

    def __iter__(self) -> "RecordingStream":
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._it)
        except StopIteration:
            self._capture.exhausted = True
            self._capture.finish()
            raise
        except Exception as exc:
            self._capture.finish(error=exc)
            raise
        self._capture.add(chunk)
        return chunk

    def close(self) -> None:
        try:
            close = getattr(self._inner, "close", None)
            if close is not None:
                close()
        finally:
            self._capture.finish(incomplete=True)

    def __enter__(self) -> "RecordingStream":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class RecordingMessageStream(RecordingStream):
    """The helper object `with client.messages.stream(...) as stream` gives you."""

    @property
    def text_stream(self) -> Any:
        return (event.text for event in self if getattr(event, "type", None) == "text")

    def until_done(self) -> "RecordingMessageStream":
        for _ in self:
            pass
        return self

    def get_final_message(self) -> Any:
        self.until_done()
        return self._inner.get_final_message()

    def get_final_text(self) -> str:
        self.until_done()
        return self._inner.get_final_text()


class RecordingManager(_Recording):
    def __enter__(self) -> RecordingMessageStream:
        inner = self._inner.__enter__()
        self._capture.final_fn = lambda: inner.current_message_snapshot
        self._stream = RecordingMessageStream(inner, self._capture)
        return self._stream

    def __exit__(self, *exc_info: Any) -> Any:
        try:
            return self._inner.__exit__(*exc_info)
        finally:
            self._capture.finish(incomplete=True)


class RecordingAsyncStream(_Recording):
    def __init__(self, inner: Any, capture: Capture):
        super().__init__(inner, capture)
        self._it = inner.__aiter__()

    def __aiter__(self) -> "RecordingAsyncStream":
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._it.__anext__()
        except StopAsyncIteration:
            self._capture.exhausted = True
            self._capture.finish()
            raise
        except Exception as exc:
            self._capture.finish(error=exc)
            raise
        self._capture.add(chunk)
        return chunk

    async def close(self) -> None:
        try:
            close = getattr(self._inner, "close", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result
        finally:
            self._capture.finish(incomplete=True)

    async def __aenter__(self) -> "RecordingAsyncStream":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()


class RecordingAsyncMessageStream(RecordingAsyncStream):
    @property
    def text_stream(self) -> Any:
        async def texts() -> Any:
            async for event in self:
                if getattr(event, "type", None) == "text":
                    yield event.text

        return texts()

    async def until_done(self) -> "RecordingAsyncMessageStream":
        async for _ in self:
            pass
        return self

    async def get_final_message(self) -> Any:
        await self.until_done()
        return await self._inner.get_final_message()

    async def get_final_text(self) -> str:
        await self.until_done()
        return await self._inner.get_final_text()


class RecordingAsyncManager(_Recording):
    async def __aenter__(self) -> RecordingAsyncMessageStream:
        inner = await self._inner.__aenter__()
        self._capture.final_fn = lambda: inner.current_message_snapshot
        self._stream = RecordingAsyncMessageStream(inner, self._capture)
        return self._stream

    async def __aexit__(self, *exc_info: Any) -> Any:
        try:
            return await self._inner.__aexit__(*exc_info)
        finally:
            self._capture.finish(incomplete=True)


# --- replay -----------------------------------------------------------------------------------


def _plain(data: Any) -> Any:
    return Box(data) if isinstance(data, (dict, list)) else data


def _chunks(event: Dict[str, Any]) -> List[Any]:
    types = event.get("types") or []
    ids = event.get("response_type")
    out = []
    for i, data in enumerate(event["response"]):
        rebuilt = rehydrate(types[ids[i]], data) if ids is not None and ids[i] >= 0 else None
        out.append(rebuilt if rebuilt is not None else _plain(data))
    return out


def _final(event: Dict[str, Any]) -> Any:
    if "final" not in event:
        return None
    rebuilt = rehydrate(event["final_type"], event["final"]) if event.get("final_type") else None
    return rebuilt if rebuilt is not None else _plain(event["final"])


def _error(event: Dict[str, Any]) -> Optional[BaseException]:
    from .recorder import ReplayedError

    err = event.get("error")
    return ReplayedError(err["type"], err["message"]) if err else None


class ReplayStream:
    def __init__(self, chunks: List[Any], error: Optional[BaseException], final: Any = None):
        self._chunks = chunks
        self._i = 0
        self._error = error
        self._final = final

    def __iter__(self) -> "ReplayStream":
        return self

    def __next__(self) -> Any:
        if self._i < len(self._chunks):
            self._i += 1
            return self._chunks[self._i - 1]
        error, self._error = self._error, None
        if error is not None:
            raise error
        raise StopIteration

    def close(self) -> None:
        self._i = len(self._chunks)
        self._error = None

    def __enter__(self) -> "ReplayStream":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class ReplayMessageStream(ReplayStream):
    @property
    def text_stream(self) -> Any:
        return (event.text for event in self if getattr(event, "type", None) == "text")

    @property
    def current_message_snapshot(self) -> Any:
        return self._final

    def until_done(self) -> "ReplayMessageStream":
        for _ in self:
            pass
        return self

    def get_final_message(self) -> Any:
        self.until_done()
        if self._final is None:
            raise RuntimeError("this recording has no final message for the stream (it was not recorded as a helper stream)")
        return self._final

    def get_final_text(self) -> str:
        return "".join(block.text for block in self.get_final_message().content if block.type == "text")


class ReplayAsyncStream:
    def __init__(self, chunks: List[Any], error: Optional[BaseException], final: Any = None):
        self._sync = ReplayStream(chunks, error, final)

    def __aiter__(self) -> "ReplayAsyncStream":
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._sync)
        except StopIteration:
            raise StopAsyncIteration from None

    async def close(self) -> None:
        self._sync.close()

    async def __aenter__(self) -> "ReplayAsyncStream":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()


class ReplayAsyncMessageStream(ReplayAsyncStream):
    @property
    def text_stream(self) -> Any:
        async def texts() -> Any:
            async for event in self:
                if getattr(event, "type", None) == "text":
                    yield event.text

        return texts()

    @property
    def current_message_snapshot(self) -> Any:
        return self._sync._final

    async def until_done(self) -> "ReplayAsyncMessageStream":
        async for _ in self:
            pass
        return self

    async def get_final_message(self) -> Any:
        await self.until_done()
        if self._sync._final is None:
            raise RuntimeError("this recording has no final message for the stream (it was not recorded as a helper stream)")
        return self._sync._final

    async def get_final_text(self) -> str:
        message = await self.get_final_message()
        return "".join(block.text for block in message.content if block.type == "text")


class ReplayManager:
    def __init__(self, make: Callable[[], ReplayMessageStream]):
        self._make = make

    def __enter__(self) -> ReplayMessageStream:
        self._stream = self._make()
        return self._stream

    def __exit__(self, *exc_info: Any) -> None:
        self._stream.close()


class ReplayAsyncManager:
    def __init__(self, make: Callable[[], ReplayAsyncMessageStream]):
        self._make = make

    async def __aenter__(self) -> ReplayAsyncMessageStream:
        self._stream = self._make()
        return self._stream

    async def __aexit__(self, *exc_info: Any) -> None:
        await self._stream.close()


def replay_stream(event: Dict[str, Any]) -> Any:
    """The object a recorded streamed call hands back: a stream, or a `messages.stream()`-style manager."""
    chunks, error, final = _chunks(event), _error(event), _final(event)
    is_async = bool(event.get("async_stream"))
    if event["stream"] == "manager":
        if is_async:
            return ReplayAsyncManager(lambda: ReplayAsyncMessageStream(chunks, error, final))
        return ReplayManager(lambda: ReplayMessageStream(chunks, error, final))
    return ReplayAsyncStream(chunks, error, final) if is_async else ReplayStream(chunks, error, final)
