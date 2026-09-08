<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/icon-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/brand/icon-light.svg">
    <img alt="Magnitude icon" src="assets/brand/icon-light.svg" width="120">
  </picture>
</p>

<h1 align="center">Magnitude</h1>

<p align="center"><strong>Local models, tuned for your Mac.</strong></p>

<p align="center">
  <a href="https://docs.magnitude.dev"><img src="https://img.shields.io/badge/%F0%9F%93%95-Docs-0369a1?style=flat-square&labelColor=0369a1&color=gray" alt="Documentation"></a>
  <a href="https://discord.gg/EHt48pPWdC"><img src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square&logo=discord&logoColor=white&labelColor=5865F2&color=gray" alt="Discord"></a>
  <a href="https://x.com/usemagnitude"><img src="https://img.shields.io/badge/Twitter-Follow-000000?style=flat-square&logo=x&logoColor=white&labelColor=000000&color=gray" alt="Follow Magnitude on Twitter"></a>
  <a href="https://github.com/magnitudedev/magnitude/stargazers"><img src="https://img.shields.io/github/stars/magnitudedev/magnitude" alt="GitHub Repo stars"></a>
  <a href="https://www.npmjs.com/package/@magnitudedev/cli"><img src="https://img.shields.io/npm/v/%40magnitudedev%2Fcli" alt="npm version"></a>
</p>

Magnitude is an open source inference server for Apple silicon. It profiles your Mac, recommends the best models for it, then downloads, tunes, and runs them. Plugs into Pi, OpenCode, Hermes, OpenClaw, Codex, Claude Code, Oh My Pi, and Cline, or use the built-in harness.

⭐ Help us reach more developers and grow the Magnitude community. Star this repo!

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/readme/ecosystem-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="assets/readme/ecosystem-light.png">
  <img alt="Pi, OpenCode, Hermes, Codex, Claude Code, and OpenClaw connected to Magnitude, which runs local models for your Mac." src="assets/readme/ecosystem-light.png">
</picture>

## Get started

See what your Mac can run:

```sh
npm i -g @magnitudedev/cli
magnitude setup
```

Setup profiles your hardware, ranks models by speed, accuracy, intelligence, and memory, and connects your harness to the one you pick.

Or let your agent handle it. Send this to Pi, Claude Code, OpenCode, or whatever you use:

```text
Set up local models for me with the Magnitude CLI. Install it with `npm i -g @magnitudedev/cli` (or my package manager), then run `magnitude docs onboarding` and follow the instructions.
```

Your agent will profile your hardware, walk you through the best local models for it, download the ones you pick, and switch itself over to them.

## Why Magnitude?

- **Knows your Mac:** profiles your hardware to assess fit and estimate tok/s per model
- **Recommends the best models:** ranked by speed, accuracy, intelligence, and memory
- **Tuned end to end:** speculative decoding and more, all set for your Mac
- **Easy setup:** one command and your agent is running local models
- **Free to run:** no token costs, API keys, or rate limits
- **Fully private and offline:** models, prompts, and files stay on your Mac
- **Models on demand:** loaded on request, unloaded when idle or memory fills
- **Open source:** Apache 2.0, yours to modify

## FAQ

### What is Magnitude?

An open source inference server for Apple silicon. It profiles your Mac, recommends the best models for it, then downloads, tunes, and runs them. Plug it into the agent you already use.

### Why only Mac?

Apple silicon is the best consumer hardware for local models, and building for one platform lets us optimize the whole stack for it, from model selection down to the kernels.

### How does it know what my Mac can run?

Magnitude profiles your chip, memory, and bandwidth, then estimates fit and tok/s for every model in the catalog. It ranks them by speed, accuracy, intelligence, and memory so you can pick.

### What Mac do I need?

Any Apple silicon Mac (M1 or later, 2020 onward) running macOS 15 or newer. Intel Macs aren't supported. There's no fixed minimum beyond that. More memory lets you run larger models.

### Which harnesses work with it?

Pi, OpenCode, Hermes, OpenClaw, Codex, Claude Code, Oh My Pi, and Cline. During setup, your agent connects your harness to the model you pick. Or use Magnitude's built-in harness.

### Do I need to manage it after setup?

No. It runs in the background, loads models when your agent needs them, and unloads them when idle or memory gets tight. Your agent can install or switch models through the CLI anytime.

### Is it private?

Yes. Prompts, files, and models stay on your Mac. Once a model is downloaded, no internet connection is needed.

## Learn more

- [Documentation](https://docs.magnitude.dev)
- [CLI reference](https://docs.magnitude.dev/reference)
- [Discord](https://discord.gg/EHt48pPWdC)
- [Report an issue](https://github.com/magnitudedev/magnitude/issues)

## License

Magnitude is licensed under the [Apache License 2.0](https://github.com/magnitudedev/magnitude/blob/main/LICENSE).
