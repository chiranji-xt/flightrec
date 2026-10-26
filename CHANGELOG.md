# Changelog

## 0.1.0 (unreleased)

First version.

- Record and replay model calls, tool calls, clock reads and random numbers to a JSON Lines file. Replay raises `ReplayDivergence` when the code stops matching the recording.
- Branch mode: change one step's result (or make it raise) and continue live from there.
- Browser viewer (`flightrec view`): scrubbable timeline, the context sent at each model call, errors, streamed responses, and a "Branch from here" panel that writes the snippet.
- `flightrec show`, `flightrec diff`, `flightrec --version`.
- Anthropic SDK support, including async clients, `async def` tools, `create(stream=True)` and `messages.stream()`. Replay rebuilds the SDK's own response types.
- Known limits: threads are not supported, a stream's event is lost if the process dies mid-stream, and only the Anthropic SDK has been tested.
