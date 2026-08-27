<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/icon-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/brand/icon-light.svg">
    <img alt="Magnitude icon" src="assets/brand/icon-light.svg" width="120">
  </picture>
</p>

<h1 align="center">Magnitude</h1>

<p align="center"><strong>Easy local inference for agents</strong></p>

<p align="center">
  <a href="https://docs.magnitude.dev"><img src="https://img.shields.io/badge/%F0%9F%93%95-Docs-0369a1?style=flat-square&labelColor=0369a1&color=gray" alt="Documentation"></a>
  <a href="https://discord.gg/EHt48pPWdC"><img src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square&logo=discord&logoColor=white&labelColor=5865F2&color=gray" alt="Discord"></a>
  <a href="https://x.com/usemagnitude"><img src="https://img.shields.io/badge/Twitter-Follow-000000?style=flat-square&logo=x&logoColor=white&labelColor=000000&color=gray" alt="Follow Magnitude on Twitter"></a>
  <a href="https://github.com/magnitudedev/magnitude/stargazers"><img src="https://img.shields.io/github/stars/magnitudedev/magnitude" alt="GitHub Repo stars"></a>
  <a href="https://www.npmjs.com/package/@magnitudedev/cli"><img src="https://img.shields.io/npm/v/%40magnitudedev%2Fcli" alt="npm version"></a>
</p>

Open source inference server that profiles your hardware, recommends the best models for it, then downloads, tunes, and runs them. Use the built-in harness or connect Pi, Hermes, or whatever you already use.

⭐ Help us reach more developers and grow the Magnitude community. Star this repo!

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/readme/ecosystem-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="assets/readme/ecosystem-light.png">
  <img alt="Magnitude works with Pi, OpenCode, Hermes, OpenClaw, Codex, Claude Code, Oh My Pi, and Cline, and leading local models" src="assets/readme/ecosystem-light.png">
</picture>

## Get started

```
npm i -g @magnitudedev/cli && magnitude setup
```

Magnitude profiles your machine and shows you the best models for it. Choose one and Magnitude downloads it. Then continue with the built-in harness or launch Pi, OpenCode, Hermes, or another supported harness with the model already loaded.

Magnitude supports macOS and Linux. Windows is supported through WSL.

## Why Magnitude?

- **Knows your hardware:** profiles your chip, memory, and bandwidth
- **Recommends what fits:** the best models for your machine, with estimated tok/s
- **No separate server:** our own inference engine, optimized for agent workloads
- **Manages models for you:** loaded on request, unloaded when idle or memory fills
- **Tuned end to end:** speculative decoding, concurrency, all set for your machine
- **Set up in one command:** pick a recommended model, then launch your harness
- **Fully private and offline:** models, prompts, and files stay on your machine
- **Free to run:** no token costs, API keys, or rate limits
- **Open source:** Apache 2.0, yours to modify

## FAQ

### What is Magnitude?

An open source inference server that profiles your hardware, recommends the best models for it, and runs them, tuned to your machine. Use the built-in harness or connect Pi, Hermes, or whatever you already use.

### What hardware do I need?

There's no fixed minimum. Magnitude profiles your hardware and recommends the best models for your machine. More memory lets you run larger models.

### How is it different from Ollama or LM Studio?

Ollama and LM Studio run the model you tell them to. Figuring out which model, which quant, whether it fits, and how fast it'll be is on you. Magnitude removes the guesswork. It profiles your machine, shows you exactly what will run well and how fast, then runs it with everything tuned, from speculative decoding to concurrency. It's also built for agent workloads. Models load just in time as requests come in, and unload when idle or memory gets tight.

### Which harnesses work with it?

Pi, OpenCode, Hermes, OpenClaw, Codex, Claude Code, Oh My Pi, and Cline. After downloading a model, continue with the built-in harness or launch another supported harness with that model already loaded in Magnitude.

### Does my data go to the cloud?

No. Prompts, files, and models stay on your machine.

### Can it run completely offline?

Yes. Once Magnitude and a model are downloaded, you can use it without an internet connection.

### Can I use models outside the catalog?

Yes. You can [download compatible GGUF models from Hugging Face](https://docs.magnitude.dev/models#download-a-model-outside-the-catalog) and use them in Magnitude.

## Learn more

- [Documentation](https://docs.magnitude.dev)
- [CLI reference](https://docs.magnitude.dev/reference)
- [Discord](https://discord.gg/EHt48pPWdC)
- [Report an issue](https://github.com/magnitudedev/magnitude/issues)

## License

Magnitude is licensed under the [Apache License 2.0](https://github.com/magnitudedev/magnitude/blob/main/LICENSE).
