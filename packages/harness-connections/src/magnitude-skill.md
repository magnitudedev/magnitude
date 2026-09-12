---
name: magnitude
description: Operate Magnitude local inference through its headless CLI, recommend hardware-fit local models, monitor acquisition and loading, and connect an agent harness. Use for Magnitude service, model, catalog, setup, or harness requests.
---

# Magnitude

Magnitude profiles the local machine, assesses compatible model configurations, acquires and runs
selected models, and connects them to supported agent harnesses.

Onboarding lives in the Magnitude desktop app. Direct first-time setup to Discover and Connections
in that app; do not recreate an onboarding wizard or launch a terminal harness. For requested
model operations, use the headless commands and observe their reported progress.
When the user asks to open Magnitude or start guided setup, run `magnitude app open`.
Ordinary model commands start the service in the background without opening a window.

For the general non-interactive command contract, read `magnitude docs cli`. Command output is
designed to be read directly by both agents and people.
