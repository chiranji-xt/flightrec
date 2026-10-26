# Demo GIF plan

One GIF at the top of the README, under the one-line description. It has to answer "what is this and why do I want it" with no sound and no reading beyond a few words.

**The story, in one sentence:** an agent gives a wrong answer, you scrub its recording to the step that went wrong, branch from there with the right data, and the agent answers correctly.

Everything below runs offline with the fake model in `examples/demo_agent.py`, so the take is deterministic and needs no API key.

## Storyboard (about 16 s)

Capture 1280×720, alternating full-screen between terminal and browser with a hard cut (no transitions). The viewer switches to a stacked layout below 830 px wide, so a split screen would show the cramped version.

| Time | Screen | What happens | What the viewer sees |
|---|---|---|---|
| 0:00 to 0:03 | Terminal | Type `# the agent gave a wrong answer`, then `python examples/demo_agent.py record-bad` | `Answer: flightrec re` (a truncated answer) |
| 0:03 to 0:07 | Browser | Run `flightrec view runs/demo-bad.jsonl` first (off camera, or cut to it). Drag across the timeline bars, then stop on step 1, the green tool bar | Detail pane shows tool call `search` returning `"flightrec re"` |
| 0:07 to 0:11 | Browser | Scroll to **Branch from here**. Keep "Return a different result", replace the text with the full sentence | The generated snippet updates as you type: `mode="branch", at=1, response=json.loads(...)` |
| 0:11 to 0:14 | Terminal | Run `python examples/demo_agent.py branch` | `Answer: flightrec records model and tool calls so agent runs can be replayed offline.` |
| 0:14 to 0:16 | Terminal | Run `flightrec diff runs/demo-bad.jsonl runs/demo-fixed.jsonl` | `first divergence at step 1: same request, different result` with the two values |

These outputs were produced by running exactly these commands (see the end of this file for the command list).

A caveat to keep honest: the snippet in the viewer ends with placeholders (`run_agent(rec.wrap(client), task)`), and the terminal beat runs a prepared script that does the same thing for this demo agent. In the GIF, cut from the snippet straight to the terminal. Do not zoom into the snippet's last lines.

## Capture settings

- **Tool:** ScreenToGif on Windows (free, records a region and lets you trim and cut frames). On macOS or Linux, any screen recorder to MP4, then `ffmpeg` to GIF.
- **Region:** 1280×720, 12 fps. 10 fps also looks fine for this kind of content.
- **Look:** dark theme in both the terminal and the browser (the viewer follows the system theme). Terminal font 18 pt or larger. Browser zoom 125%. The text must be readable at the README width of roughly 830 px.
- **Timing:** hold each result for at least 1.5 s so it can be read. No scrolling except the one short scroll to **Branch from here**.
- **Size:** aim for under 4 MB. Optimise with `gifsicle -O3 --lossy=80 --colors 128 demo.gif -o docs/demo.gif`. If it is still too big, drop to 10 fps or shrink to 1000 px wide before cutting content.
- **Loop:** loop forever, with a 2 s pause on the last frame.

## Before recording

1. Fresh checkout, `pip install -e .`, and delete any old `runs/` folder.
2. Terminal prompt set to a plain `$ ` so no user name or path shows. In PowerShell: `function prompt { "$ " }`.
3. Close every other browser tab, hide the bookmarks bar, turn off notifications.
4. Do a dry run of the whole sequence. Recordings are tiny, so retakes cost nothing.
5. Make sure `runs/demo-bad.jsonl` exists before the browser beat starts (the first terminal beat creates it).

## Acceptance check

- Someone who has never seen the project can say what happened after one viewing.
- The wrong answer and the right answer are both readable without pausing.
- The step where it went wrong (step 1) is visibly the one you land on.
- File size under 4 MB, and it loops cleanly.

## Where it goes

- Save as `docs/demo.gif`.
- In `README.md`, directly under the first paragraph:
  `![Recording an agent run, scrubbing to the bad step, branching from it, and getting the right answer](docs/demo.gif)`
- Also worth making: a static 1280×640 image of the viewer timeline with a streamed run, for the GitHub repository social preview (Settings, Social preview).

## Exact commands, in order

```
python examples/demo_agent.py record-bad
flightrec view runs/demo-bad.jsonl
python examples/demo_agent.py branch
flightrec diff runs/demo-bad.jsonl runs/demo-fixed.jsonl
```
