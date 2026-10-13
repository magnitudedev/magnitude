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

Magnitude is an open source inference engine that optimizes each model for your exact hardware. When you download a model, Magnitude spends up to a minute tuning its kernels to find the fastest configuration for your hardware. On the same hardware, it's 2x faster than Ollama. It works on Apple Silicon, NVIDIA, AMD, or just a CPU. One click connects it to the agent you already use (Pi, OpenCode, Hermes, Codex, and more).

**[Download Magnitude for macOS, Windows, or Linux](https://magnitude.dev/download)**

⭐ Help us reach more developers and grow the Magnitude community. Star this repo!

https://github.com/user-attachments/assets/983328c8-93e8-4360-bfef-11e93ff76035

## Get started

1. [Download Magnitude](https://magnitude.dev/download), install it, and open the app.
2. Choose a recommended model in **Discover** and download it.
3. Connect your agent in **Connections** and start using it.

The desktop app includes the `magnitude` CLI. No separate installation is needed.

## Why Magnitude?

- **2x faster than Ollama:** on average across popular local models, on Mac and NVIDIA
- **Tuned on your hardware:** kernels are tuned for each model when you download it
- **Best drafter built in:** each model ships with the fastest speculative decoding drafter
- **Memory that flexes:** quantized KV halves memory per agent, freed when agents stop
- **Fast concurrent sessions:** sessions share prefix caches to prevent slowdown
- **Works with your agent:** one click to connect Pi, OpenCode, Hermes, Codex, and more
- **Free, private, open source:** no token costs, nothing leaves your machine, Apache 2.0

## 2x faster than Ollama

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/benchmarks/ollama-dark.svg">
  <img alt="Magnitude vs Ollama, average of 11 popular local models: 34% faster prefill and 100% faster decode on Mac, 27% faster prefill and 100% faster decode on NVIDIA" src="assets/benchmarks/ollama-light.svg" width="800">
</picture>

## FAQ

### What is Magnitude?

An open source inference engine that optimizes each model for your exact hardware. It ships as a desktop app that runs open models and connects them to the agent you already use.

### How is it faster than llama.cpp, Ollama, or LM Studio?

They ship one-size-fits-all kernels, because tuning each model for each chip by hand would take forever. Magnitude tunes them automatically. When you download a model, it spends up to a minute on your actual chip trying different ways to split up the math (how many GPU threads team up, how big a chunk each one takes, how far ahead to load data) and keeps the fastest combination that doesn't change the model's output. [See the benchmarks against Ollama.](#2x-faster-than-ollama)

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
