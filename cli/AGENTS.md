# CLI Architecture

The CLI is entirely headless. It exposes finite commands for agents and terminal users; onboarding and visual interaction belong to the desktop application. Do not add a TUI, chat harness, terminal renderer, or interactive onboarding.

## Client-State Guidance

Before changing CLI state, RPC usage, subscriptions, or async lifecycle code, read [`packages/client-common/AGENTS.md`](../packages/client-common/AGENTS.md). Preserve its state ownership and Effect-scoped resource rules. Component hooks and React mounting patterns apply to renderers, not this headless package.

## CLI Boundaries

- Import product APIs and wire types only from `@magnitudedev/client-common` and `@magnitudedev/sdk`. Never import ACN, protocol, agent, AI, provider, storage, or inference-engine packages directly.
- Privileged bootstrap and service-command composition may import private `daemon-management`; ordinary product commands do not own process coordination. Privileged connection composition may import the private harness-connections package.
- Reuse client-common's product derivations, recommendation policy, and shared capabilities. Keep CLI modules focused on command registration, finite output, and headless orchestration.
- Acquire the shared first-party connection in an Effect scope. Finite SDK reads and commands consume their results directly; do not construct a renderer, React hooks, a second request cache, or a separate onboarding workflow to run them.
- Compose independent observations in pure output functions. Do not add a combined server RPC or retained state merely because one command prints several domains.
- A command acknowledgement is distinct from download or load completion. Print the relevant observation command when admitted work continues; progress and terminal state come from the authoritative product resource.
- Validate command syntax before acquiring runtime capabilities. Help, version, documentation, and rejected syntax must not start the application or mutate user configuration.
- Keep runtime imports behind command execution so passive commands remain usable without native application adapters or an installed desktop.
- External connection commands configure and inspect supported harnesses; they never launch one. Keep reusable connector implementation in the host package rather than adding a CLI-specific path.
