<p align="center">
  <img src="wordmark.svg" alt="Magnitude" width="100%" />
</p>

<p align="center">
  <a href="https://docs.magnitude.dev" target="_blank"><img src="https://img.shields.io/badge/📕-Docs-232f41?style=flat-square&labelColor=0369a1&color=gray" alt="Documentation" /></a> <a href="https://app.magnitude.dev" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%96%A5-Provider-232f41?style=flat-square&labelColor=0369a1&color=gray" alt="Provider" /></a> <img src="https://img.shields.io/badge/License-Apache%202.0-232f41?style=flat-square&labelColor=0369a1&color=gray" alt="License" /> <a href="https://discord.gg/VcdpMh9tTy" target="_blank"><img src="https://img.shields.io/discord/1305570963206836295?style=flat-square&logo=discord&logoColor=white&label=Discord&labelColor=5865F2&color=gray" alt="Discord" /></a> <a href="https://x.com/usemagnitude" target="_blank"><img src="https://img.shields.io/badge/-Follow%20Magnitude-000000?style=flat-square&labelColor=000000&color=gray&logo=x&logoColor=white" alt="Follow Magnitude" /></a>
</p>

<p align="center">
  <strong>Open source coding agent for open source models</strong>
</p>

Magnitude is ruthlessly optimized to achieve frontier level (or greater) performance on coding tasks using open source models. For way cheaper than frontier APIs and with no lock in.

How we do that:
- Model agnostic response syntax + parser that is continuously tested
- Custom [GBNF](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) grammars that enforce our response syntax
- Built-in software development skills (that are hackable)
- Task tree primitive that keeps agent on track

Everything is open source and can be set up to run locally. However, we do offer the [Magnitude Provider](https://app.magnitude.dev) that has generous rate limits and native support for our GBNF grammars.

<p align="center">
  <img src="interface.png" alt="Magnitude interface" width="100%" />
</p>

## Get started

Download Magnitude from your favorite package manager:
```
npm i -g @magnitudedev/cli
```

Start Magnitude in your favorite terminal emulator:
```
magnitude
```

> If you are on Windows, you will need to use `wsl`. We do not have native Windows support yet.

This will start a setup wizard for configuring your provider + choosing models. We recommend the [Magnitude Provider](https://app.magnitude.dev). However, we have support for most major providers, including OpenRouter, Vercel AI Gateway, and Fireworks. You can also run models locally with LMStudio, Ollama, llama.cpp, or any OpenAI-compatible provider.

## Customizing skills

Magnitude ships with 14 built-in skills that activate automatically when relevant. On first run, built-in skills are copied to  `~/.magnitude/skills/` . You can edit these files directly to customize them.

- **bug** - Fixing unexpected behavior, errors, or test failures
- **feature** - Building new functionality end-to-end
- **refactor** - Restructuring code without changing behavior
- **scan** - Quick, targeted information gathering
- **explore-codebase** - Mapping how systems work
- **explore-docs** - Researching external APIs and libraries
- **research** - Investigation requiring evidence-backed findings
- **plan** - Designing before implementing complex work
- **ideate** - Exploring options and tradeoffs
- **implement** - Executing when objectives are already clear
- **review** - Independent verification of work
- **debug** - Hypothesis-driven root cause isolation
- **approve** - User decision points
- **other** - Catch-all for uncategorized work

You can also provide project-specific skills in `(cwd)/.magnitude/skills/` which will override the global ones. Skills follow the standard [Agent Skills](https://agentskills.io/home) format.

## Applying grammars

Magnitude uses custom [GBNF](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) grammars to enforce our response syntax. This can meaningfully improves adherence to the syntax and prevents malformed outputs. 

Grammars are generated automatically from the tool definitions. They're sent to providers that support them (Magnitude Provider, Fireworks, and llama.cpp) and silently skipped for providers that don't. Set `MAGNITUDE_ENABLE_GRAMMAR=false`  to disable grammar constraints entirely.

## Magnitude Provider

We offer a $20/mo coding plan. This includes:
- The newest open source models like GLM 5.1 and Kimi K2.5
- Native support for Magnitude GBNF grammars with no setup
- Generous usage limits that reset every 5 hours
- Built-in web search support using Exa
- Very fast models with global infrastructure
- Zero data retention for all models

Sign up at [app.magnitude.dev](https://app.magnitude.dev).

## Why we built this

Open source models have caught up to the frontier. But no one is building a harness for them.

Even open source harnesses seem to be geared towards Claude and Codex models. It's understandable. Open source models were not good enough for a long time. And the subsidized subscriptions from the frontier labs are compelling.

But the time is now for open source. You can get the same (or even better) performance in the right harness. For way cheaper.

Not to mention you won't be locked in to one ecosystem. We all know how that ends up. 

## Acknowledgments

Built on top of [BAML](https://boundaryml.com), [Effect](https://effect.website), and [OpenTUI](https://github.com/anomalyco/opentui).

Inspired by other open source coding agents, including [OpenCode](https://github.com/anomalyco/opencode) and [Codex](https://github.com/openai/codex).