<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/icon-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/brand/icon-light.svg">
    <img alt="Magnitude icon" src="assets/brand/icon-light.svg" width="120">
  </picture>
</p>

<h1 align="center">Magnitude</h1>

<p align="center"><strong>Run open models as fast as your hardware allows</strong></p>

<p align="center">
  <a href="https://magnitude.dev/download"><img src="https://img.shields.io/badge/-Download-gray?style=flat-square&labelColor=0369a1&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZmZmZmZmIiBzdHJva2Utd2lkdGg9IjIuMjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI%2BPHBhdGggZD0iTTEyIDN2MTIiLz48cGF0aCBkPSJtNyAxMCA1IDUgNS01Ii8%2BPHBhdGggZD0iTTQgMTd2MmEyIDIgMCAwIDAgMiAyaDEyYTIgMiAwIDAgMCAyLTJ2LTIiLz48L3N2Zz4%3D" alt="Download Magnitude"></a>
  <a href="https://docs.magnitude.dev"><img src="https://img.shields.io/badge/%F0%9F%93%95-Docs-0369a1?style=flat-square&labelColor=0369a1&color=gray" alt="Documentation"></a>
  <a href="https://discord.gg/EHt48pPWdC"><img src="https://img.shields.io/badge/-Discord-gray?style=flat-square&logo=discord&logoColor=white&labelColor=5865F2" alt="Discord"></a>
  <a href="https://x.com/usemagnitude"><img src="https://img.shields.io/badge/-Twitter-gray?style=flat-square&logo=x&logoColor=white&labelColor=000000" alt="Follow Magnitude on Twitter"></a>
  <a href="https://github.com/magnitudedev/magnitude/stargazers"><img src="https://img.shields.io/github/stars/magnitudedev/magnitude" alt="GitHub Repo stars"></a>
</p>

Magnitude is an open source inference engine for agents that optimizes itself for your exact hardware. It compiles and tunes its kernels on your device, so open models run up to 2x faster than llama.cpp. One click connects the agent you already use (Pi, OpenCode, Hermes, Codex, and more). Works on Apple Silicon, NVIDIA, AMD, or nothing but a CPU.

**[Download Magnitude for macOS, Windows, or Linux](https://magnitude.dev/download)**

⭐ Help us reach more developers and grow the Magnitude community. Star this repo!

https://github.com/user-attachments/assets/983328c8-93e8-4360-bfef-11e93ff76035

## Get started

1. [Download Magnitude](https://magnitude.dev/download), install it, and open the app.
2. Choose a recommended model in **Discover** and download it.
3. Connect your agent in **Connections** and start using it.

Or install from a terminal:

```sh
curl -fsSL https://magnitude.dev/install.sh | sh     # macOS and Linux
irm https://magnitude.dev/install.ps1 | iex          # Windows PowerShell
```

The desktop app includes the `magnitude` CLI. No separate installation is needed.

### Run it on a server

On Linux and macOS, `magnitude server setup` runs Magnitude as a service that starts at boot, keeps running after logout, updates itself, and serves browsers and agents on other computers. On Linux, the install command offers to set this up for you. See [Remote server](https://docs.magnitude.dev/remote-server).

## Why Magnitude?

- **Up to 2x faster than llama.cpp:** 92% faster decode on Metal, 19% on CUDA
- **Tuned on your device:** kernels are tuned on your hardware before a model runs
- **Built for the best models:** hand-optimized kernels for popular open-weight families
- **Memory that flexes:** 27% less memory per agent, freed when agents stop
- **Fast concurrent sessions:** sessions share prefix caches to prevent slowdown
- **Works with your agent:** one click to connect Pi, OpenCode, Hermes, Codex, and more
- **Free, private, open source:** no token costs, nothing leaves your machine, Apache 2.0

## Up to 2x faster than llama.cpp

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/benchmarks/llama-cpp-dark.svg">
  <img alt="Magnitude vs llama.cpp: 9% faster prefill and 92% faster decode on Metal, 23% faster prefill and 19% faster decode on CUDA" src="assets/benchmarks/llama-cpp-light.svg" width="800">
</picture>

## FAQ

### What is Magnitude?

An open source inference engine that optimizes itself for your hardware. It ships as a desktop app that runs open models and connects them to the agent you already use.

### How is it faster than llama.cpp, Ollama, or LM Studio?

They ship kernels precompiled for broad classes of hardware. Magnitude compiles and tunes its kernels on your actual device before a model runs, so they fit your exact chip. [See the benchmarks against llama.cpp.](#up-to-2x-faster-than-llamacpp)

### What hardware do I need?

Any Apple Silicon, NVIDIA, or AMD GPU, or nothing but a CPU. There is no fixed minimum. Smaller machines run smaller models, and more memory lets you run larger ones.

### What operating systems does it support?

macOS, Linux, and Windows.

### Which models does it support?

See the full list at [magnitude.dev/models](https://magnitude.dev/models). We write optimized kernels for the most popular open-weight families, which is how we beat generalist engines.

### Which agents work with it?

One click connects Pi, OpenCode, Hermes, OpenClaw, Codex, Claude Code, Oh My Pi, and Cline. Anything else works through the OpenAI-compatible API.

### Is it private?

Yes. Prompts, files, and models stay on your machine. No internet needed once a model is downloaded.
