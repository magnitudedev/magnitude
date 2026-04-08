<p align="center">
  <img src="wordmark.svg" alt="Magnitude" width="100%" />
</p>

<p align="center">
  <a href="https://docs.magnitude.dev" target="_blank"><img src="https://img.shields.io/badge/📕-Docs-232f41?style=flat-square&labelColor=0369a1&color=gray" alt="Documentation" /></a> <img src="https://img.shields.io/badge/License-Apache%202.0-232f41?style=flat-square&labelColor=0369a1&color=gray" alt="License" /> <a href="https://discord.gg/VcdpMh9tTy" target="_blank"><img src="https://img.shields.io/discord/1305570963206836295?style=flat-square&logo=discord&logoColor=white&label=Discord&labelColor=5865F2&color=gray" alt="Discord" /></a> <a href="https://x.com/usemagnitude" target="_blank"><img src="https://img.shields.io/badge/-Follow%20Magnitude-000000?style=flat-square&labelColor=000000&color=gray&logo=x&logoColor=white" alt="Follow Magnitude" /></a>
</p>

Magnitude is an **open source coding agent** with a built-in engineering process. A lead agent understands your intent, maps it into engineering tasks, and delegates to specialized workers.

- **Better code**. Real engineering workflows with verification throughout.
- **Less babysitting**. The lead drives the process so you don't have to.
- **Holds up on long tasks**. The lead doesn't code, so its context stays focused.
<p align="center">
  <img src="interface.png" alt="Magnitude Interface" width="100%" />
</p>

## Get started

```bash
npm install -g @magnitudedev/cli
```

Then navigate to the directory you want to work in and launch the TUI:

```bash
magnitude
```

This will launch Magnitude with a setup wizard for configuring providers and models.

Magnitude works with most major model providers out of the box, including open source and local models. You can use your **ChatGPT Plus/Pro** or **GitHub Copilot** subscription.

See the [provider docs](https://docs.magnitude.dev/configuration/providers) for full provider support.

## How it works

A lead agent chats with you to understand your intent. It groups the task into feature, bug, or refactor, each with its own workflow. It builds a task tree, breaks the work into subtasks, and matches each one to a more specific workflow for things like research or implementation. It then delegates each task to specialized workers with focused context and toolsets.

The lead coordinates those workers through full plan / build / verify loops. The task tree updates as work unfolds and adapts as you steer it. You can jump in anytime by chatting with the lead. Otherwise, it keeps going autonomously until clarification is actually needed.

### Workers

The lead agent manages all workers on your behalf. It can message, stop, resume, or redirect them and run many in parallel.

Magnitude comes out of the box with the following workers:
- **Explorer**: for doing codebase or web research, both broad and narrow
- **Planner**: for evaluating various implementation strategies
- **Builder**: for implementing code changes directly in your files
- **Reviewer**: for strict, independent review of code changes
- **Debugger**: for root causing bugs and fixing them
- **Browser**: for verifying UI changes with a built-in browser agent

Magnitude may use none or all of these in a given session. For a quick fix in a single file, it may edit it directly. For a very in-depth change, it may use the whole team. For most tasks, it will use some combination of explorer, planner, builder, and reviewer.

<p align="center">
  <img src="architecture-dark.png" alt="Magnitude Architecture" width="100%" />
</p>

## Why we built this

We love AI coding and were early power users of Claude Code. But left unsupervised, it quickly turns into *slop*. We expected Claude Code to keep evolving to fix this, but it hasn't. It leans too far into "just let the model figure it out". Codex is similar.

Projects like Superpowers (which gives CC more skills) have gained a lot of momentum because people want agents to follow better software development practices. But in practice, the results are mixed. You often have to keep prompting the agent to actually use the skills, especially as tasks get longer.

This is because these skills are layered on top of an underlying agent primitive that wasn't built to use them consistently. So instead, **we built the engineering process into the primitive itself**.

## Documentation

Full documentation is available at [docs.magnitude.dev](https://docs.magnitude.dev).

## Contributing

See the [contributing guide](https://docs.magnitude.dev/contributing) to get started.

## Acknowledgements

Built on top of [BAML](https://boundaryml.com), [Effect](https://effect.website), and [OpenTUI](https://github.com/anomalyco/opentui).

Inspired by other open-source coding agents, including [OpenCode](https://github.com/anomalyco/opencode) and [Codex](https://github.com/openai/codex).
