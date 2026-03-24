<p align="center">
  <img src="wordmark.svg" alt="Magnitude" width="100%" />
</p>

<p align="center">
  <a href="https://docs.magnitude.dev" target="_blank"><img src="https://img.shields.io/badge/📕-Docs-232f41?style=flat-square&labelColor=232f41&color=gray" alt="Documentation" /></a> <img src="https://img.shields.io/badge/License-Apache%202.0-232f41?style=flat-square&labelColor=232f41&color=gray" alt="License" /> <a href="https://discord.gg/VcdpMh9tTy" target="_blank"><img src="https://img.shields.io/discord/1305570963206836295?style=flat-square&logo=discord&logoColor=white&label=Discord&labelColor=5865F2&color=gray" alt="Discord" /></a> <a href="https://x.com/usemagnitude" target="_blank"><img src="https://img.shields.io/badge/-Follow%20Magnitude-000000?style=flat-square&labelColor=000000&color=gray&logo=x&logoColor=white" alt="Follow Magnitude" /></a>
</p>

Magnitude is an **open source coding agent** with subagent-native architecture. You can work for hours on the same task with no performance degradation. 

<p align="center">
  <img src="interface.png" alt="Magnitude Interface" width="100%" />
</p>

## Installation

```bash
npm install -g @magnitudedev/cli
```

Then navigate to the directory you want to work in and launch the TUI:

```bash
magnitude
```

This will launch Magnitude with a setup wizard for configuring providers and models.

### Providers

Magnitude works with most major model providers out of the box, including open source and local models.

You can use your **ChatGPT Plus/Pro** or **GitHub Copilot** subscription.

See the [provider docs](https://docs.magnitude.dev/configuration/providers) for full provider support.

## How it works

Magnitude has a team lead that coordinates subagent usage on your behalf. Its primary objective is to listen to you and turn your intent into running subagents. It can message, stop, resume, or redirect subagents and run many in parallel. You can also message subagents directly.

Magnitude comes out of the box with the following specialized subagents:
- **Explorer**: for codebase or web research, both broad and narrow
- **Planner**: for evaluating various implementation strategies
- **Builder**: for implementing code changes directly in your files
- **Reviewer**: for strict, independent review of code changes
- **Debugger**: for root causing bugs and fixing them
- **Browser**: for verifying UI changes with a built-in browser agent

Magnitude may use none or all of these in a given session. For a very quick fix in a single file, it may edit it directly. For a very in-depth change, it may use all six. For most tasks, it will use some combination of explorer, planner, builder, and reviewer. 

## Why we built this

We became fully subagent-pilled the first time we saw Claude Code use an explore agent. We expected it to continue getting better and better. **But it didn't.**

The community clearly feels the same — projects like Superpowers have hit 100k+ GitHub stars augmenting coding agents with custom skills and subagents. But these approaches are limited by the underlying agent they plug into. They provide great skills but rely on the host agent to actually use them consistently and effectively. And often you're stuck manually orchestrating them with slash commands.

By controlling the underlying agent primitive, you can go much further. Fine-grained control over lead-subagent communication, lifecycle, and behavior. Mechanical consistency, not just prompts. That's what Magnitude is.

## Additional Info

### Documentation

Full documentation is available at [docs.magnitude.dev](https://docs.magnitude.dev).

### Contributing

See the [contributing guide](https://docs.magnitude.dev/contributing) to get started.

### Acknowledgements

Built on top of [BAML](https://boundaryml.com), [Effect](https://effect.website), and [OpenTUI](https://github.com/anomalyco/opentui).

Inspired by other open-source coding agents, including [OpenCode](https://github.com/anomalyco/opencode) and [Codex](https://github.com/openai/codex).
