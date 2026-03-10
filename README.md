<p align="center">
  <img src="wordmark.svg" alt="Magnitude" width="100%" />
</p>

<p align="center"><b>Open source AI coding agent</b></p>

<p align="center">
  <a href="https://docs.magnitude.dev" target="_blank"><img src="https://img.shields.io/badge/📕-Docs-232f41?style=flat-square&labelColor=232f41&color=gray" alt="Documentation" /></a> <img src="https://img.shields.io/badge/License-Apache%202.0-232f41?style=flat-square&labelColor=232f41&color=gray" alt="License" /> <a href="https://discord.gg/VcdpMh9tTy" target="_blank"><img src="https://img.shields.io/discord/1305570963206836295?style=flat-square&logo=discord&logoColor=white&label=Discord&labelColor=5865F2&color=gray" alt="Discord" /></a> <a href="https://x.com/usemagnitude" target="_blank"><img src="https://img.shields.io/badge/-Follow%20Magnitude-000000?style=flat-square&labelColor=000000&color=gray&logo=x&logoColor=white" alt="Follow Magnitude" /></a>
</p>

Most coding agents are one long conversation that compacts and loses context. Magnitude is a multi-agent system that stays sharp over long horizon tasks.

- **Advanced orchestration** — Orchestrator delegates to subagents with two-way communication
- **Context stays clean** — Conversation stays focused on your intent. Less compaction, better UX
- **Parallel by default** — Spins up multiple subagents working concurrently, not one thing at a time
- **Native browser agent** — Vision-based browser agent verifies UI changes and behavior
- **Any model** — Frontier, open source, local, whatever you want

<p align="center">
  <img src="interface.png" alt="Magnitude Interface" width="100%" />
</p>

## Installation

```bash
npm install -g @magnitudedev/magnitude
bun add -g @magnitudedev/magnitude
pnpm add -g @magnitudedev/magnitude
yarn global add @magnitudedev/magnitude
```

After installation, just run `magnitude` in your terminal to launch the TUI:

```bash
magnitude
```

This will launch Magnitude with a setup wizard for configuring providers and models.

Optionally, run `/init` to automatically set up an [AGENTS.md](https://agents.md) for your project.

### Skills

Magnitude supports the [Agent Skills](https://agentskills.io) standard. Skills are automatically discovered on launch from the following locations in order of priority:

1. `<project>/.magnitude/skills/`
2. `<project>/.agents/skills/`
3. `~/.magnitude/skills/`
4. `~/.agents/skills/`

## Features

### Orchestrator → subagent architecture

The orchestrator manages the conversation and delegates to specialized subagents (builder, debugger, explorer, planner, reviewer, browser), each with its own context window, toolset, and permissions. Subagents do focused work and report back, keeping the orchestrator's context clean.

### Context sharing via artifacts

Named documents that carry context between agents with scoped visibility. An explorer writes findings, a planner reads them and writes a plan, a builder reads the plan and implements. Each agent sees exactly what it needs without inheriting everything.

### Two-way agent communication

The orchestrator and subagents have full bidirectional messaging. The orchestrator can steer, redirect, or interrupt subagents mid-work. Not fire-and-forget delegation.

### Event-sourced runtime

Every action is an immutable event. Sessions are fully replayable and resumable.

### Built-in browser agent

A vision-based browser agent built on [browser-agent](https://github.com/magnitudedev/browser-agent) runs natively in the same runtime. Used to verify UI changes and behavior.

### Steerable and hackable

Skills, AGENTS.md, persistent memory, agent policies. The system adapts to how you work, not the other way around. Magnitude learns your preferences and codebase conventions over time, storing them in `.magnitude/memory.md` and applying them automatically to future sessions.

## Additional Info

### Provider and model support

See the [providers](https://docs.magnitude.dev/configuration/providers) and [models](https://docs.magnitude.dev/configuration/models) pages in the docs. If you would like another provider to be supported, please feel free to create an issue or raise a PR yourself.

### Contributing

See the [contributing guide](https://docs.magnitude.dev/contributing) to get started.

### Documentation

Full documentation is available at [docs.magnitude.dev](https://docs.magnitude.dev).

### Acknowledgements

Built on top of [BAML](https://boundaryml.com), [Effect](https://effect.website), and [OpenTUI](https://github.com/anomalyco/opentui).

Inspired by other open-source coding agents, including [OpenCode](https://github.com/anomalyco/opencode) and [Codex](https://github.com/openai/codex).

The future is open!
