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
  <a href="https://magnitude.dev/download"><img src="https://img.shields.io/badge/-Download-gray?style=flat-square&labelColor=0369a1&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZmZmZmZmIiBzdHJva2Utd2lkdGg9IjIuMjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI%2BPHBhdGggZD0iTTEyIDN2MTIiLz48cGF0aCBkPSJtNyAxMCA1IDUgNS01Ii8%2BPHBhdGggZD0iTTQgMTd2MmEyIDIgMCAwIDAgMiAyaDEyYTIgMiAwIDAgMCAyLTJ2LTIiLz48L3N2Zz4%3D" alt="Download Magnitude"></a>
  <a href="https://docs.magnitude.dev"><img src="https://img.shields.io/badge/%F0%9F%93%95-Docs-0369a1?style=flat-square&labelColor=0369a1&color=gray" alt="Documentation"></a>
  <a href="https://discord.gg/EHt48pPWdC"><img src="https://img.shields.io/badge/-Discord-gray?style=flat-square&logo=discord&logoColor=white&labelColor=5865F2" alt="Discord"></a>
  <a href="https://x.com/usemagnitude"><img src="https://img.shields.io/badge/-Twitter-gray?style=flat-square&logo=x&logoColor=white&labelColor=000000" alt="Follow Magnitude on Twitter"></a>
  <a href="https://github.com/magnitudedev/magnitude/stargazers"><img src="https://img.shields.io/github/stars/magnitudedev/magnitude" alt="GitHub Repo stars"></a>
</p>

Magnitude is an open source inference engine optimized for consumer hardware. It profiles your machine, recommends the best models for it, then downloads, tunes, and runs them. One click connects the agent you already use. Works on Apple Silicon, NVIDIA, AMD, or nothing but a CPU.

**[Download Magnitude for macOS, Windows, or Linux](https://magnitude.dev/download)**

⭐ Help us reach more developers and grow the Magnitude community. Star this repo!

https://github.com/user-attachments/assets/8317d05b-8a6e-40e0-b45d-81011ecbc329

## Get started

1. [Download Magnitude](https://magnitude.dev/download), install it, and open the app.
2. Choose a recommended model in **Discover** and download it.
3. Connect your agent in **Connections** and start using it.

The desktop app includes the `magnitude` CLI. No separate installation is needed.

## Why Magnitude?

- **Knows your machine:** profiles your hardware and estimates tok/s before you download
- **Recommends the best models:** ranked by speed, accuracy, intelligence, and memory
- **Tuned end to end:** speculative decoding and more, all set for your hardware
- **Works with your agent:** one click to connect Pi, OpenCode, Hermes, and more
- **Free to run:** no token costs, API keys, or rate limits
- **Fully private and offline:** models, prompts, and files stay on your machine
- **Models on demand:** loaded on request, unloaded when idle or memory fills
- **Open source:** Apache 2.0, yours to modify

## FAQ

### What is Magnitude?

An open source inference engine optimized for consumer hardware. The desktop app profiles your machine, recommends the best models for it, then downloads, tunes, and runs them. One click connects the agent you already use.

### How does it know what my machine can run?

Magnitude profiles your hardware and estimates tok/s for every model in the catalog before you download anything. It ranks them by speed, accuracy, intelligence, and memory so you can pick.

### How is this different from Ollama or LM Studio?

They run whatever model you pick. Magnitude helps you pick. It estimates how every model and quant will perform on your machine before you download, then tunes the one you choose for your exact hardware, from context size to speculative decoding.

### What hardware do I need?

There's no fixed minimum. Magnitude profiles your machine and recommends what runs well on it. More memory lets you run larger models.

### What systems does Magnitude support?

The desktop app is native on macOS, Linux, and Windows. It runs on Apple Silicon, NVIDIA and AMD GPUs, and CPU-only machines, including unified-memory boxes like DGX Spark and Strix Halo.

### Which harnesses work with it?

Pi, OpenCode, Hermes, OpenClaw, Codex, Claude Code, Oh My Pi, and Cline. Pick a model and connect your harness in one click.

### Do I need to manage it after setup?

No. It runs in the background, loads models when your agent needs them, and unloads them when idle or memory gets tight.

### Is it private?

Yes. Prompts, files, and models stay on your machine. Once a model is downloaded, no internet connection is needed.

## Learn more

- [Documentation](https://docs.magnitude.dev)
- [CLI reference](https://docs.magnitude.dev/reference)
- [Discord](https://discord.gg/EHt48pPWdC)
- [Report an issue](https://github.com/magnitudedev/magnitude/issues)

## License

Magnitude is licensed under the [Apache License 2.0](https://github.com/magnitudedev/magnitude/blob/main/LICENSE).
