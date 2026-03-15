<p align="center">
  <img src="wordmark.svg" alt="Magnitude" width="100%" />
</p>

<p align="center"><b>Open source coding agent</b></p>

<p align="center">
  <a href="https://docs.magnitude.dev" target="_blank"><img src="https://img.shields.io/badge/📕-Docs-232f41?style=flat-square&labelColor=232f41&color=gray" alt="Documentation" /></a> <img src="https://img.shields.io/badge/License-Apache%202.0-232f41?style=flat-square&labelColor=232f41&color=gray" alt="License" /> <a href="https://discord.gg/VcdpMh9tTy" target="_blank"><img src="https://img.shields.io/discord/1305570963206836295?style=flat-square&logo=discord&logoColor=white&label=Discord&labelColor=5865F2&color=gray" alt="Discord" /></a> <a href="https://x.com/usemagnitude" target="_blank"><img src="https://img.shields.io/badge/-Follow%20Magnitude-000000?style=flat-square&labelColor=000000&color=gray&logo=x&logoColor=white" alt="Follow Magnitude" /></a>
</p>

Open source coding agent where everything happens in specialized subagents. We call this *subagent driven development*. The main agent stays focused on your intent, leading to stronger performance on longer tasks and higher quality code.

<p align="center">
  <img src="interface.png" alt="Magnitude Interface" width="100%" />
</p>

## Installation

```bash
npm install -g @magnitudedev/cli
```

After installation, navigate to the directory you want to work in and launch the TUI:

```bash
magnitude
```

This will launch Magnitude with a setup wizard for configuring providers and models.

### Providers

Magnitude works with most major model providers out of the box.

**Subscription-based** — Use models you already pay for, no API key needed.
- **ChatGPT Plus/Pro** — Sign in with your OpenAI account
- **GitHub Copilot** — Sign in with your GitHub account

**API key** — Bring your own key from any of these providers.
- Anthropic, OpenAI, Google Gemini, OpenRouter, Vercel AI Gateway, Google Vertex AI, Amazon Bedrock, MiniMax, Z.AI, Cerebras

**Local models** — Connect to Ollama or any OpenAI-compatible endpoint. No auth required.

See the [providers docs](https://docs.magnitude.dev/configuration/providers) for setup instructions. If you would like another provider to be supported, please feel free to create an issue or raise a PR yourself.

## Why Magnitude?

- **Avoids the dumb zone.** Since building happens in subagents, the main agent stays focused on your intent and effective even on longer, messier tasks.
- **Steerable subagents.** Subagents can surface blockers, ask clarifying questions, and be paused, resumed, or redirected mid-task.
- **Throw out the skills files.** Plan, review, debug, and browse are built-in subagents. No configuration needed, and the main agent actually uses them consistently.
- **Mix your models.** Chat with Claude, build with Codex, or use whatever combination fits your preferences for intelligence, speed, and cost.
- **Scoped toolsets, not modes.** Real coding jumps between planning and building too often for explicit mode toggles. Each subagent gets tools scoped to its role instead.
- **Talk to it while it works.** The main agent isn't blocked by implementation. Ask questions, clarify mid-build, or start planning what's next without interrupting anything.

## Features

### Main agent → subagent architecture
The main agent manages the conversation and delegates to specialized subagents (explorer, planner, builder, reviewer, browser, debugger), each with its own context window, toolset, and permissions. Subagents do focused work and report back, keeping the main agent's context clean.

### Two-way agent communication
The main agent and subagents have full bidirectional messaging. The main agent can steer, redirect, or interrupt subagents mid-work. Subagents can message back when they hit blockers or need clarification. Not fire-and-forget delegation.

### Context sharing via artifacts
Shared markdown documents pass context between agents directly. An explorer writes findings, a planner reads them and produces a plan, a builder reads the plan and implements. No context is lost to summarization, and the main agent doesn't burn output tokens on handoff.

### Parallel by default
The main agent spins up multiple subagents concurrently and keeps working while they run. Exploring separate areas of the codebase, implementing unrelated features, or debugging multiple issues all happen in parallel.

### Progressive disclosure of tool output
Tool output does not enter the context window by default. Agents inspect results explicitly and pull in only what they need, keeping noisy output out of the conversation.

### Sensible default permissions
Commands are classified by risk. Read-only commands run freely. Normal commands are allowed in your project directory but blocked outside it. Dangerous commands are always blocked. No `--dangerously-skip-permissions`, no stuck permission prompts.

### Built-in browser agent
A vision-based browser agent built on [browser-agent](https://github.com/magnitudedev/browser-agent) runs natively in the same runtime for verifying UI changes and behavior as part of the workflow.

### Persistent memory
Magnitude learns your preferences and codebase conventions over time, storing them in `.magnitude/memory.md` and applying them automatically to future sessions. Skills and AGENTS.md are also supported.

## Additional Info

### Philosophy

The coding agent primitive is not solved. Most failures come from two recurring patterns: **context degradation** over long sessions, and **local maximum traps** where the agent settles for the nearest plausible fix. Magnitude is built around those failure modes. Read more in our [manifesto](https://magnitude.dev/manifesto).

### Contributing

See the [contributing guide](https://docs.magnitude.dev/contributing) to get started.

### Documentation

Full documentation is available at [docs.magnitude.dev](https://docs.magnitude.dev).

### Acknowledgements

Built on top of [BAML](https://boundaryml.com), [Effect](https://effect.website), and [OpenTUI](https://github.com/anomalyco/opentui).

Inspired by other open-source coding agents, including [OpenCode](https://github.com/anomalyco/opencode) and [Codex](https://github.com/openai/codex).
