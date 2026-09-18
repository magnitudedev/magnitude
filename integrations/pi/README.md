# Magnitude for Pi

Run Pi on local models. Free, private, and offline.

Magnitude runs the best local models for your hardware, plugged into the agent you already use.
It profiles your machine, recommends the models that fit, then downloads, tunes, and runs them.

## Get started

Install the Magnitude desktop app. Download a model in Discover, then connect Pi from Connections.
Restart Pi and choose a Magnitude model with `/model`. Normal connections configure models and
install the Magnitude skill; they do not install or require this extension.

## Local extension development

This directory contains the optional Pi extension source. The package is private and is not
published to npm. From a configured Magnitude development checkout, run `bun dev:pi` to develop
the extension. The extension features and slash commands below apply to that development setup.

Requires Pi 0.83.0 or newer. Use a supported Magnitude desktop installation in the same graphical user session as Pi.
An internet connection is needed for installation and model downloads; after that, you can work offline.

## Why Magnitude?

- **Free to run:** no token costs, API keys, or rate limits
- **Private and offline:** local model requests stay on your machine
- **Recommends what fits:** the best models for your hardware, with estimated tok/s
- **Tuned end to end:** inference settings chosen for your machine
- **Models on demand:** loaded when Pi needs them, unloaded when idle or memory gets tight
- **Live performance stats:** see model loading, prompt processing, and time spent working

After each completed run, a summary shows the model, time worked, time to first token, and tokens per
second. It stays in your chat history so you can refer back to it.

## Using your models

Choose an installed Magnitude model from Pi's `/model` selector. To discover or install more models,
ask your agent:

```text
Use Magnitude to recommend the best local models for my hardware and help me install one.
```

Use `/magnitude-setup` to open the desktop again, or `/stop-model` to unload the active model and
free its memory.

### Already using Magnitude?

Connect your installed models to Pi:

```sh
magnitude connections add pi
```

Then restart Pi or run `/reload`.

## Learn more

- [Documentation](https://docs.magnitude.dev)
- [GitHub](https://github.com/magnitudedev/magnitude)
- [Discord](https://discord.gg/EHt48pPWdC)
- [Report an issue](https://github.com/magnitudedev/magnitude/issues)
