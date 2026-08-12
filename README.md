# Magnitude

<a href="https://docs.magnitude.dev" target="_blank"><img src="https://img.shields.io/badge/📕-Docs-0369a1?style=flat-square&labelColor=0369a1&color=gray" alt="Documentation" /></a>
<a href="https://discord.gg/EHt48pPWdC" target="_blank"><img src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square&logo=discord&logoColor=white&labelColor=5865F2&color=gray" alt="Discord" /></a> <a href="https://x.com/usemagnitude" target="_blank"><img src="https://img.shields.io/badge/Twitter-Follow-000000?style=flat-square&logo=x&logoColor=white&labelColor=000000&color=gray" alt="Follow Magnitude on Twitter" /></a>

Your fully local, private agent. Runs models on your machine with its built-in inference engine. Works out of the box, on any hardware. 

![Magnitude running a local model](docs/maglocaldemo.gif)

## Get started

```sh
npm install -g @magnitudedev/cli
cd your-project
magnitude
```

Magnitude supports macOS and Linux. Windows is supported through WSL.

## What you can use it for

- Analyze sensitive data
- Manage private notes
- Review code and logs
- Search and organize files
- Build docs or slides
- Create automation scripts

Out of the box it can use your shell, edit files, and run scripts. Add skills and it can work with Excel, PowerPoint, PDFs or Chrome.

## Add skills

Skills are reusable capabilities for your agent. A good way to get them is
[skills.sh](https://www.skills.sh), a skills directory from Vercel.

Skills we recommend:

```sh
npx skills add vercel-labs/agent-browser   # drive your logged-in Chrome browser
npx skills add anthropics/skills/xlsx      # read and build Excel spreadsheets
npx skills add anthropics/skills/pptx      # build PowerPoint decks
npx skills add anthropics/skills/docx      # read and write Word documents
npx skills add anthropics/skills/pdf       # read, fill, and create PDFs
```

## How it works

### Automatic model setup

Magnitude profiles your hardware and recommends the best models your machine can run. Choose Balanced, Best Quality, Fastest, or Lightweight, and Magnitude handles the download and configuration.

### An inference engine built for agent work

Magnitude includes a custom inference engine written in Rust on top of llama.cpp. It offers verified model configurations, calculates memory requirements before loading, and tunes acceleration, placement, and batching for your hardware. Parallel agents retain full context windows, model switching preserves consistent tool use, and new requests remain responsive while other work is running.

### An agent built around local models

Magnitude can inspect and edit files, run commands, work with images, and manage long sessions. Because local inference is built in, it also manages model loading and switching and surfaces native prefill, cache reuse, and generation performance directly in the agent UI. Nothing leaves your machine.

## Learn more

- [Documentation](https://docs.magnitude.dev)
- [CLI reference](https://docs.magnitude.dev/reference)
- [Discord](https://discord.gg/EHt48pPWdC)
- [Report an issue](https://github.com/magnitudedev/magnitude/issues)

## License

Magnitude is licensed under the [Apache License 2.0](https://github.com/magnitudedev/magnitude/blob/main/LICENSE).
