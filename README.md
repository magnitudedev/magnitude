<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/icon-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/brand/icon-light.svg">
    <img alt="Magnitude icon" src="assets/brand/icon-light.svg" width="120">
  </picture>
</p>

<h1 align="center">Magnitude</h1>

<p align="center"><strong>Run the best local models for your machine</strong></p>

<p align="center">
  <a href="https://docs.magnitude.dev"><img src="https://img.shields.io/badge/%F0%9F%93%95-Docs-0369a1?style=flat-square&labelColor=0369a1&color=gray" alt="Documentation"></a>
  <a href="https://discord.gg/EHt48pPWdC"><img src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square&logo=discord&logoColor=white&labelColor=5865F2&color=gray" alt="Discord"></a>
  <a href="https://x.com/usemagnitude"><img src="https://img.shields.io/badge/Twitter-Follow-000000?style=flat-square&logo=x&logoColor=white&labelColor=000000&color=gray" alt="Follow Magnitude on Twitter"></a>
  <a href="https://github.com/magnitudedev/magnitude/stargazers"><img src="https://img.shields.io/github/stars/magnitudedev/magnitude" alt="GitHub Repo stars"></a>
  <a href="https://www.npmjs.com/package/@magnitudedev/cli"><img src="https://img.shields.io/npm/v/%40magnitudedev%2Fcli" alt="npm version"></a>
</p>

Magnitude is an open source local inference engine. It runs on the hardware you already have, whether that's a Mac, an NVIDIA or AMD GPU, or just a CPU. It profiles your machine, recommends the right models for it, then downloads, tunes, and runs them. Plug it into Pi, OpenCode, Hermes, OpenClaw, Codex, Claude Code, Oh My Pi, and Cline.

⭐ Help us reach more developers and grow the Magnitude community. Star this repo!

## Get started

Install the Magnitude desktop application from the [release downloads](https://github.com/magnitudedev/magnitude/releases), using the DMG on macOS or an installer available for your platform. On macOS, open the DMG, drag Magnitude into Applications, and launch it from there. Open the app to browse **Discover**, compare curated recommendations for your machine, and download a model.

Use **My Models** to load or stop models, **Connections** to configure an external harness, **Status** to inspect service and model state, and **Settings** for appearance and launch at login. Connect writes the harness configuration and supported integration files; open the harness yourself when it is ready.

Closing the window keeps Magnitude running in the background. **Quit Magnitude** stops the service and its inference processes. On Linux desktops without a supported tray host, reopen the app from your applications menu.

### Moving from the previous application

Quit the old application and stop and disable its standalone service before installing the new
desktop. Download the new installer directly; there is no automatic migration or shell installer.
Keep your Magnitude data directory to retain downloaded models and settings. Update your CLI if you
use it, then reconnect your external harnesses from **Connections**. Set **Launch at login** in the
new app if desired. If Status reports an occupied port, stop the process using it and select
**Retry service**.

### Headless CLI

Install the CLI for agents, scripts, and terminal use after installing the desktop application:

```sh
npm i -g @magnitudedev/cli
magnitude docs cli
magnitude service status
```

The CLI controls the same desktop-owned service. Background startup does not open a window. Use `magnitude app open` when you explicitly want to show the app. Onboarding lives in the desktop application.

## Why Magnitude?

- **Knows your machine:** profiles your hardware to assess fit and estimate tok/s
- **Recommends the best models:** ranked by speed, accuracy, intelligence, and memory
- **Tuned end to end:** speculative decoding and more, all set for your hardware
- **Easy setup:** discover a model and connect your harness in the desktop app
- **Free to run:** no token costs, API keys, or rate limits
- **Fully private and offline:** models, prompts, and files stay on your machine
- **Models on demand:** loaded on request, unloaded when idle or memory fills
- **Open source:** Apache 2.0, yours to modify

## FAQ

### What is Magnitude?

An open source inference server for the hardware you already have. It profiles your machine, recommends the right models for it, then downloads, tunes, and runs them. Plug it into the agent you already use.

### How does it know what my machine can run?

Magnitude profiles your chip, memory, and bandwidth, then estimates fit and tok/s for every model in the catalog. It ranks them by speed, accuracy, intelligence, and memory so you can pick.

### What hardware do I need?

There's no fixed minimum. Magnitude profiles your machine and recommends what runs well on it. More memory lets you run larger models.

### What systems does Magnitude support?

Installer availability is listed with each release. Hardware support depends on the inference backend available for that platform. Magnitude profiles your machine and recommends compatible catalog models; Linux tray support also depends on your desktop environment.

### Which harnesses work with it?

Pi, OpenCode, Hermes, OpenClaw, Codex, Claude Code, Oh My Pi, and Cline. Use Connections in the desktop app to configure your harness without launching it.

### Do I need to manage it after setup?

No. It runs in the background, loads models when your agent needs them, and unloads them when idle or memory gets tight. Your agent can install or switch models through the CLI anytime.

### Is it private?

Yes. Prompts, files, and models stay on your machine. Once a model is downloaded, no internet connection is needed.

## Learn more

- [Documentation](https://docs.magnitude.dev)
- [CLI reference](https://docs.magnitude.dev/reference)
- [Discord](https://discord.gg/EHt48pPWdC)
- [Report an issue](https://github.com/magnitudedev/magnitude/issues)

## License

Magnitude is licensed under the [Apache License 2.0](https://github.com/magnitudedev/magnitude/blob/main/LICENSE).
