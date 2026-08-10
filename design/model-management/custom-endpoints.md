---
applies_to:
  - packages/storage/src/types/custom-endpoints.ts
  - packages/storage/src/types/config.ts
  - packages/ai/src/codec/native-chat-completions/**
  - packages/ai/src/protocol/native-chat-completions.ts
  - packages/providers/src/custom-endpoint/**
  - packages/providers/src/registry.ts
  - packages/acn/src/custom-endpoint-**
  - packages/acn/src/custom-endpoints.ts
  - packages/acn/src/shared-client.ts
  - packages/acn/src/provider-model-catalog.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - cli/src/features/model-menus/**
---

# Custom endpoints

Custom endpoints let a user declare OpenAI-compatible Chat Completions providers in the global
Magnitude configuration. The declaration is authoritative for connection metadata and the models
Magnitude presents. Magnitude does not discover models from the endpoint.

## Identity

Each endpoint has an authored key and maps deterministically to a provider ID in the reserved
`custom:` namespace. Each model key is both its provider-model ID and the exact model value sent to
the endpoint. Display-name, connection, credential, and capability edits at the same keys preserve
identity. Changing a key removes one identity and adds another.

## Declaration

A declaration contains a display name, HTTP or HTTPS API root, explicit authentication strategy,
optional non-secret headers, and at least one model. Models declare their display name, context
window, maximum output, and optional vision or reasoning behavior. Tools are supported by the
protocol contract. Structured output is not declared. Pricing is unknown rather than zero.

Secrets are never stored in configuration. Authentication may be absent by explicit choice or may
reference an environment variable for bearer or named-header authentication. A missing credential
keeps the provider and its models declared and selected, but projects the provider as not configured.

## Observation

ACN loads the declaration at startup and polls the global configuration once per second. Each poll
performs one file read and one schema decode. A valid semantic change replaces the resident custom
endpoint declarations. A missing file means no declarations. A read failure, malformed JSON, or
schema-invalid configuration retains the last accepted declarations and retries on the next poll.

No fingerprinting, repeated-observation protocol, filesystem locking, or partial recovery is part of
this contract.

## Provider and catalog integration

Every accepted declaration materializes as an ordinary provider registration and uses the normal
provider catalog and binding paths. Provider registration carries an explicit provider kind; clients
do not infer it from provider IDs. Custom models appear in the ordinary model picker and require no
agent-specific inference path.

Custom endpoints own interpretation of provider-specific Chat Completions response variants. Their
response decoder normalizes supported reasoning and thinking representations into the same semantic
thought stream consumed by the agent. Shared stream reduction remains independent of provider wire
field names. When an endpoint supplies equivalent reasoning in more than one representation, the
decoder emits the text once.

## Selection behavior

Temporary unavailability, including a missing credential, preserves slot intent. Authoritative
removal is different: when a valid declaration removes a selected endpoint or model, every slot
selecting that identity becomes explicitly `Unassigned`. Magnitude never substitutes another model.

If the removed identity later reappears, it becomes selectable again but is not silently reselected.
Favorites and recency remain provider/model preferences and are not deleted with the declaration.

## Conformance

- Invalid external configuration never replaces the last accepted declarations.
- Credential values never enter configuration, catalog state, logs, or client state.
- Provider and model keys determine runtime identity.
- Missing credentials preserve selection and project unavailability.
- Valid removal clears affected slots without fallback.
- Reappearance does not restore cleared slot assignment.
- Custom models use the ordinary catalog, picker, slot, and bound-model paths.
- Supported custom reasoning and thinking responses produce one normalized thought stream without
  duplicated text.
