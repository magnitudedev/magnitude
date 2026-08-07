# Magnitude

<a href="https://docs.magnitude.dev" target="_blank"><img src="https://img.shields.io/badge/📕-Docs-0369a1?style=flat-square&labelColor=0369a1&color=gray" alt="Documentation" /></a>
<a href="https://discord.gg/EHt48pPWdC" target="_blank"><img src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square&logo=discord&logoColor=white&labelColor=5865F2&color=gray" alt="Discord" /></a> <a href="https://x.com/usemagnitude" target="_blank"><img src="https://img.shields.io/badge/Twitter-Follow-000000?style=flat-square&logo=x&logoColor=white&labelColor=000000&color=gray" alt="Follow Magnitude on Twitter" /></a>

An open source coding agent with its own local inference engine.

![Magnitude coding with a local model](docs/maglocaldemo.gif)

## Get started

```sh
npm install -g @magnitudedev/cli
cd your-project
magnitude
```

Magnitude supports macOS and Linux. Windows is supported through WSL.

## How it works

### Automatic model setup

Magnitude profiles your hardware and recommends the best models your machine can run. Choose Balanced, Best Quality, Fastest, or Lightweight, and Magnitude handles the download and configuration.

### An inference engine built for agent work

Magnitude includes a custom inference engine written in Rust on top of llama.cpp. It offers verified model configurations, calculates memory requirements before loading, and tunes acceleration, placement, and batching for your hardware. Parallel agents retain full context windows, model switching preserves consistent tool use, and new requests remain responsive while other work is running.

### A coding agent built around local inference

Magnitude can inspect and edit files, run commands, work with images, manage long-running tasks, and delegate work to parallel agents. Because local inference is built in, it also manages model loading and switching and surfaces native prefill, cache reuse, and generation performance directly in the agent UI.

## Learn more

- [Documentation](https://docs.magnitude.dev)
- [CLI reference](https://docs.magnitude.dev/reference)
- [Discord](https://discord.gg/EHt48pPWdC)
- [Report an issue](https://github.com/magnitudedev/magnitude/issues)

## License

Magnitude is licensed under the [Apache License 2.0](https://github.com/magnitudedev/magnitude/blob/main/LICENSE).
