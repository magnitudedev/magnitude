# Magnitude

Everything you need to code with local models.

<a href="https://docs.magnitude.dev" target="_blank"><img src="https://img.shields.io/badge/📕-Docs-0369a1?style=flat-square&labelColor=0369a1&color=gray" alt="Documentation" /></a> <a href="https://app.magnitude.dev" target="_blank"><img src="https://img.shields.io/badge/☁️-Cloud-0369a1?style=flat-square&labelColor=0369a1&color=gray" alt="Magnitude Cloud" /></a> <a href="https://discord.gg/EHt48pPWdC" target="_blank"><img src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square&logo=discord&logoColor=white&labelColor=5865F2&color=gray" alt="Discord" /></a> <a href="https://x.com/usemagnitude" target="_blank"><img src="https://img.shields.io/badge/Twitter-Follow-000000?style=flat-square&logo=x&logoColor=white&labelColor=000000&color=gray" alt="Follow Magnitude on Twitter" /></a>

## Get started

```sh
npm install -g @magnitudedev/cli
cd your-project
magnitude
```

On first launch, Magnitude detects your hardware and recommends local models and quantizations that fit your machine. Choose one and Magnitude handles the download and configuration.

Magnitude supports macOS and Linux. Windows is supported through WSL.

<!-- Add the 15–20 second demo video here. -->

## How it works

### Automatic model setup

Magnitude detects available memory and acceleration, then recommends model and quantization combinations based on your hardware and how many local sessions you want to run. Choose a recommendation to install and configure it.

### A purpose-built inference engine

Magnitude runs local models on an inference engine built for coding agent workloads. Built in Rust on llama.cpp, it configures model fit, context, parallelism, and runtime settings for your hardware.

The inference engine gives the coding agent direct control over KV cache, concurrency, and subagent limits, allowing both parts of Magnitude to adapt to the work being done.

### A coding agent

Once your model is running, Magnitude is ready to work in your codebase. It can inspect and edit files, run commands, manage long sessions, work with images, and delegate tasks to parallel agents.

## Magnitude Cloud

Magnitude Cloud gives you access to open models that are too large to run on your hardware. It costs $10 for the first month, then $20 per month.

- GLM 5.2, Kimi K3, DeepSeek V4, and more
- Generous usage limits for long coding sessions
- Fast, reliable inference with native weights
- Exa web search for external research

<a href="https://app.magnitude.dev"><u>Get Magnitude Cloud →</u></a>

## Learn more

- [Documentation](https://docs.magnitude.dev)
- [CLI reference](https://docs.magnitude.dev/reference)
- [Discord](https://discord.gg/EHt48pPWdC)
- [Cloud](https://app.magnitude.dev)
- [Report an issue](https://github.com/magnitudedev/magnitude/issues)

## License

Magnitude is licensed under the [Apache License 2.0](https://github.com/magnitudedev/magnitude/blob/main/LICENSE).
