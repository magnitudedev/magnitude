# CLI Architecture

The CLI is a React client rendered with OpenTUI (`@opentui/core` and `@opentui/react`). It handles terminal input and owns CLI-only presentation state. Shared server state and domain behavior belong in [`packages/client-common`](../packages/client-common/).

## Required Client-State Guidance

Before changing CLI state, hooks, RPC usage, subscriptions, or async lifecycle code, read [`packages/client-common/AGENTS.md`](../packages/client-common/AGENTS.md). Its rules govern:

- choosing server, shared-client, or presentation state ownership;
- AtomRpc queries, mutations, `Result`, and `reactivityKeys`;
- Effect Atom derivation and event-source actions;
- `useAtomMount` lifecycle and stream-to-query invalidation bridges;
- prohibited React state mirroring and imperative synchronization patterns.

## CLI Boundaries

- Import product APIs and wire types only from `@magnitudedev/client-common` and `@magnitudedev/sdk`. Never import ACN, protocol, agent, AI, provider, storage, or inference-engine packages directly.
- Use OpenTUI components and hooks for terminal rendering and interaction. Renderer and event types come from `@opentui/core`; React bindings such as `createRoot`, `useRenderer`, and `useKeyboard` come from `@opentui/react`.
- Put reusable domain atoms, hooks, actions, and RPC subscription bridges in client-common. Keep CLI modules focused on OpenTUI rendering, terminal interaction, CLI routes, and genuinely CLI-only presentation atoms.
- Compose independent domains in a pure CLI view model when a screen needs them together. Do not merge their state systems or request a screen-shaped RPC.
- Use `useAgentClient` AtomRpc query and mutation atoms for ordinary RPC work. Do not add promise wrappers plus `useState` for loading, errors, progress, or server snapshots.
- Long-running operation UI reads progress from an authoritative query. A mounted stream may invalidate that query; it must not become a second state store.
- `useEffect`, ref-diff synchronization, async IIFEs for server state, and callback-ref dependency effects are not accepted. Follow the declarative/event-source/`useAtomMount` decision in the client-common guidance.
