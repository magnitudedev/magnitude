<p align="center">
  <img src="wordmark.svg" alt="Magnitude" width="100%" />
</p>

<p align="center">
  <a href="https://docs.magnitude.dev" target="_blank"><img src="https://img.shields.io/badge/📕-Docs-232f41?style=flat-square&labelColor=0369a1&color=gray" alt="Documentation" /></a> <a href="https://app.magnitude.dev" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%96%A5-Provider-232f41?style=flat-square&labelColor=0369a1&color=gray" alt="Provider" /></a> <img src="https://img.shields.io/badge/License-Apache%202.0-232f41?style=flat-square&labelColor=0369a1&color=gray" alt="License" /> <a href="https://discord.gg/VcdpMh9tTy" target="_blank"><img src="https://img.shields.io/discord/1305570963206836295?style=flat-square&logo=discord&logoColor=white&label=Discord&labelColor=5865F2&color=gray" alt="Discord" /></a> <a href="https://x.com/usemagnitude" target="_blank"><img src="https://img.shields.io/badge/-Follow%20Magnitude-000000?style=flat-square&labelColor=000000&color=gray&logo=x&logoColor=white" alt="Follow Magnitude" /></a>
</p>

<p align="center">
  <strong>Open source coding agent for open source models</strong>
</p>

Magnitude is ruthlessly optimized for coding with open source models. How we do that:
- **Custom response handling** — Native thinking and tool calls require the model template, provider, and harness to line up perfectly. Often they don't, and things break. We solve this with a custom response format, grammar, and streaming parser which work consistently across open models.
- **Efficient subagents** — Our subagents work async and can be configured with a different model. Plan with a strong model while it delegates to faster/cheaper ones.
- **Flexible behavior** — Instead of baking behavior into a system prompt meant for a specific model, Magnitude uses skills as a core part of its workflow. Use the robust defaults or tweak them to fit your workflow.
- **Purpose-built provider** — Our [provider](https://app.magnitude.dev) enables grammar-constrained decoding with our response format for maximum reliability, on a flat-rate plan with generous limits. We also support major open model providers (OpenRouter, Vercel, Z.AI, etc.) and first-class local inference via llama.cpp, Ollama, and LMStudio.

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

Magnitude uses a formal [GBNF](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) grammar, which constrains the model's vocabulary during generation to only our response format. This helps further reduce edge cases.

The grammar is generated automatically from the tool definitions. It's sent to providers that support it (Magnitude Provider, Fireworks, and llama.cpp) and silently skipped for providers that don't. Set `MAGNITUDE_ENABLE_GRAMMAR=false`  to disable grammar constraints entirely.

## Magnitude Provider

We offer a $20/mo coding plan. This includes:
- The newest open source models like GLM 5.1 and Kimi K2.6
- Native support for Magnitude GBNF grammar with no setup
- Generous usage limits that reset every 5 hours
- Built-in web search support using Exa
- Very fast models with global infrastructure
- Zero data retention for all models

Sign up at [app.magnitude.dev](https://app.magnitude.dev). There's a 3 day free trial with no card required.

## Why we built this

Right now there are two categories of harnesses:
- Harnesses that optimize for a single model family (Claude Code, Codex)
- Harnesses that support a wide variety of models (Opencode, Cursor, etc)

The problem is, open source models have their own challenges and quirks that need be to addressed in the harness. Without careful attention, you get: broken thinking/tool calls, model behavior failures (doom loops, randomly stopping), and generally subpar performance.

For the teams building Claude Code and Codex: they have consistent thinking, tool calling, provider, and model behaviors, which enables them to build a reliable agent experience. Current model-agnostic harnesses need to support a broad range of closed and open models, which means open models don't get the necessary attention.

We're giving them the necessary attention to make their performance equivalent to closed source models at coding tasks. For way cheaper.

## Acknowledgments

Built on top of [BAML](https://boundaryml.com), [Effect](https://effect.website), and [OpenTUI](https://github.com/anomalyco/opentui).

Inspired by other open source coding agents, including [OpenCode](https://github.com/anomalyco/opencode) and [Codex](https://github.com/openai/codex).