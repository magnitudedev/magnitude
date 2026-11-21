---
name: magnitude
description: Operate Magnitude local inference through its headless CLI, recommend hardware-fit local models, monitor acquisition and loading, and connect an agent harness. Use for Magnitude service, model, setup, or harness requests.
---

# Magnitude

Magnitude profiles the local machine, assesses compatible model configurations, acquires and runs
selected models, and connects them to supported agent harnesses.

Magnitude is installed either as the desktop app, which includes the CLI, or as the CLI only. Both
have the same app for first-time setup and choosing models: the desktop app shows it in a window, and
with the CLI only it is in a browser while `magnitude serve` runs. Direct first-time setup to the app;
do not recreate an onboarding wizard or launch a terminal harness. For requested model operations,
use the headless commands and observe their reported progress.

## Running the service

Model and connection commands use a running service and never start one.

- When the user asks to open Magnitude or start guided setup, run `magnitude app open`. It shows the
  desktop app's window, which starts the service, or with the CLI only opens the app in a browser.
- Without a display (over SSH, or on a server), or when `app open` can't open anything, run
  `magnitude serve`. It works with either install and serves until stopped, so start it in its own
  terminal, or, after asking the user, in the background as described in `magnitude docs headless`.
  While the desktop app is open it serves instead, and `serve` waits.
- In a Docker container, the image already runs `magnitude serve`. The app and API are at the port
  the container publishes on its host, not at the container's own `127.0.0.1`.

## Models and agents

- `magnitude models recommend` lists the models that run well on this computer, and
  `magnitude models pull <model-id>` downloads one. See `magnitude docs recommendations` and
  `magnitude docs custom-models` for any other model from Hugging Face.
- `magnitude connections connect <harness>` connects an agent on this computer, with
  `--set-model <model-id>` to choose its model. The desktop app's Connect does the same; in a
  browser, the app gives setup instructions instead.
- For other computers, read `magnitude docs remote-access`.

For the general non-interactive command contract, read `magnitude docs cli`. Command output is
designed to be read directly by both agents and people.
