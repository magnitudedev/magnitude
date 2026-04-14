# Acknowledgements

## Origin

This skillset is adapted from [GStack](https://github.com/garrytan/gstack), created by [Garry Tan](https://github.com/garrytan). GStack is a collection of opinionated skills that serve as CEO, Designer, Eng Manager, Release Manager, Doc Engineer, QA Lead, and more — packaging deep domain expertise into structured, repeatable workflows for AI coding agents.

## License

GStack is released under the MIT License. A copy of the original license is included in this directory as [`LICENSE`](./LICENSE).

## What was adapted

GStack was built as a CLI tool for Claude Code, so each skill file was wrapped in several hundred lines of session management, telemetry, cross-session learning, tool-specific formats, and CLI binary calls. This adaptation strips all of that infrastructure out, redirects file paths to the Magnitude agent workspace, replaces tool-specific commands with plain prose, and makes all language model-agnostic — preserving the core workflows, scoring rubrics, checklists, and domain knowledge as faithfully as possible.
