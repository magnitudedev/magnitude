---
applies_to:
  - packages/agent/src/ambient/tool-availability-ambient.ts
  - packages/agent/src/tools/toolkits.ts
  - packages/agent/src/projections/agent-toolkit.ts
  - packages/agent/src/coding-agent.ts
  - packages/acn/src/shared-client.ts
  - packages/acn/src/agent-factory.ts
  - packages/acn/src/provider-credentials.ts
  - packages/sdk/src/provider-client.ts
  - packages/providers/src/exa/**
  - packages/providers/src/magnitude/provider.ts
  - packages/providers/src/web-search-error.ts
---

# Runtime tool availability

The agent distinguishes tools the runtime understands from tools an agent may use now.

## Stable universe and effective toolkit

The tool universe contains every executable tool and state model understood by the runtime. It is
stable for the event-engine lifetime and is the authority for replaying historical tool events.
Credentials, provider health, model configuration, and role policy never remove entries from that
universe.

`AgentToolkitProjection` is the sole authority for the effective ordered tool keys exposed to each
fork. It combines:

- static role eligibility;
- session policy;
- active-model configuration; and
- dynamic runtime tool availability.

Consumers materialize toolkits from those projected keys. Replay uses the stable universe rather
than current availability.

## Availability ownership

`ToolAvailabilityAmbient` contains non-model runtime facts that gate tools already
present in the universe. It is separate from:

- model configuration, which describes active model slots;
- session options, which are launch policy; and
- the tool universe, which is stable replay authority.

ACN owns provider-backed availability because it owns credential storage, concrete provider-client
construction, and resident-session client refresh. ACN publishes one process-wide authoritative
value and a change stream that emits the current observation when subscribed, followed by semantic
changes. Each coding-agent session treats an observation as a reason to read the authoritative
value under its synchronization lock and updates its ambient only when the semantic value differs.
The initial stream observation closes the gap between taking a startup snapshot and subscribing.

Event-core commits ambient changes and their dependent projection updates as one serialized
transaction. An invocation reads the model configuration and ordered tool keys from the same
projected toolkit state, then captures an immutable effective toolkit. Later ambient changes alter
tool membership on the next invocation, not the active Harness or its retries.

Provider-client refresh is independent of that membership snapshot. Each web-search execution
captures one concrete route when the request begins. A refresh may therefore affect a later search
started by an already-active Harness, but it never changes the route of an in-flight request.

## Web-search routing

Cloud is disabled. Web search has exactly one configured route:

1. direct Exa when `EXA_API_KEY` is nonblank; or
2. unavailable when Exa is not configured.

The disabled Cloud route remains implemented but does not participate in provider registration or
web-search routing.

When unavailable, `webSearch` remains in the tool universe but is absent from every effective
toolkit that would otherwise include it. `webFetch` is independent and remains eligible.

Direct Exa calls use Effect HTTP from the ACN process and preserve the common web-search result
contract. Provider failures use explicit schema-backed variants for configuration, transport,
request encoding, timeout, rejected response, body read, and invalid response. Raw unknown failures
are normalized at the provider boundary and are never retained in an untyped cause field.

## Credential changes

Provider-client refresh and availability publication derive from the same resolved concrete
clients.

- Enabling a route installs executable clients before publishing availability.
- Disabling the last route publishes unavailability before installing unavailable clients.
- Switching available routes installs replacement clients before publishing the new source.

Availability is published only when the semantic source changes. Rebuilding clients with the same
source does not churn projected tool definitions.

Keys never enter ambients, events, projections, display state, introspection, or logs.

## Acceptance criteria

1. Exa selects direct Exa regardless of stored or environment Magnitude credentials.
2. Without Exa, `web_search` is removed from model tool definitions.
3. Current unavailability never prevents historical web-search events from replaying.
4. Resident sessions reflect availability changes in their next effective toolkit, while searches
   started after a provider-client refresh use the refreshed route.
5. Effective toolkit selection is identical across normal turns, compaction, and exported tool
   definitions.
6. Provider failures retain typed semantic detail without unknown cause bags.
