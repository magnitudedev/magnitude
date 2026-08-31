# Harness live model discovery and selection

Research date: 2026-08-31

## Executive conclusion

The useful answer is more optimistic than the current Magnitude connectors suggest. The connectors mostly write files and launch processes, but several harnesses expose runtime surfaces that Magnitude does not use yet.

For sessions created after the Magnitude integration is installed, every harness can reach bucket 3 except the normal Cline IDE surface. Cline can reach bucket 2 today and bucket 3 either through its Hub/ACP surfaces or through a small upstream/companion API addition. The hard cases are not model switching itself; they are discovering an independently started process and obtaining an authenticated control handle for it.

| Harness | Current connector | Best attainable integration | Main mechanism |
|---|---:|---:|---|
| Magnitude | 3 | **3** | ACN RPC |
| Pi | 2 | **3** | Globally installed extension bridge |
| OpenCode | 1 | **3** | Shared authenticated service; direct session model RPC |
| Hermes | 3-capable but unused | **3** | Gateway REST/JSON-RPC |
| OpenClaw | 3-capable but unused | **3** | Gateway `config.patch` + `sessions.patch` |
| Codex | 1 | **3** | Managed app-server + dynamic `/models` catalog |
| Claude Code | 1 | **3 for managed sessions** | Agent SDK live control protocol |
| Oh My Pi | 2 | **3** | Globally installed extension bridge |
| Cline | 1, isolated | **2 in the IDE; 3 in Hub/ACP** | Shared catalog now; addressable runtime for 3 |

“Best attainable” is not the same as attaching to a process that was opened before Magnitude was installed. Pi and Oh My Pi can adopt such a process after one explicit reload. An ordinary OpenCode TUI can be forced to reload its catalog, but direct selection requires a reachable server or bridge. Codex, Claude Code, and Cline do not offer a supported post-hoc attach API for an arbitrary private CLI/desktop process.

## What the buckets mean

The original buckets are cumulative product outcomes:

1. **Startup-only** — Magnitude can configure or select a model for a future process, but cannot update the model view of an already-running session.
2. **Live catalog** — Magnitude can cause a new model to become available in an already-running session programmatically.
3. **Live selection** — Magnitude has an authenticated handle for a session and can set the model used by its next turn programmatically.

There are three separate mechanisms underneath those outcomes:

- **Catalog source:** static file, reloadable file, provider `/models`, or runtime provider registration.
- **Session addressability:** PID, stdio child, Unix socket, HTTP/WebSocket server, or in-process extension/SDK handle.
- **Selection mutation:** model setter on a session, next-turn override, or persistent per-session patch.

A connector should report these separately. A bucket-3 setter does not imply that a new model has reached the target process, and a reloadable picker does not imply that Magnitude can select an entry in it.

## Detailed feasibility matrix

| Harness | Live catalog path | Live session setter | Can adopt an already-open ordinary process? | Recommended product shape |
|---|---|---|---|---|
| Pi | `registerProvider` is immediate; registry refresh is public | `pi.setModel` | Yes after `/reload` loads the bridge | Install one global Magnitude extension |
| OpenCode | Global config update disposes/rebuilds instances; `SIGUSR2` reload exists | `POST /api/session/:id/model` | Catalog yes by PID signal; setter only if server is reachable | Shared authenticated service or plugin bridge |
| Hermes | `/api/model/options?refresh=1` | REST session lock or WS `config.set` | Yes for a known gateway; local Desktop backend is private | Magnitude-owned `hermes serve` |
| OpenClaw | Gateway `config.patch`, reload events, `models.list` | `sessions.patch` | Yes through the configured Gateway | Connect as a normal Gateway client |
| Codex | Provider-owned `/models`, cache invalidation, catalog ETag | `thread/resume` and `turn/start` model override | No for an arbitrary private process | Magnitude-owned app-server and remote TUI |
| Claude Code | No general custom live picker catalog | SDK `set_model` | No supported local attach | Keep Agent SDK process alive and own it |
| Oh My Pi | Registry refresh and immediate `registerProvider` | extension `setModel` | Yes after plugin/runtime reload | Install one global Magnitude extension |
| Cline | Shared providers store plus dynamic provider registration; plugin provider registration | Core/ACP/Hub runtime setters exist | Not from the normal IDE's public API | Share config for 2; add IDE/Hub bridge for 3 |

## Pi: bucket 3 with an extension bridge

Pi does not need to be restricted to RPC mode. Its normal TUI loads global extensions from `~/.pi/agent/extensions/`, and auto-discovered extensions can be hot-reloaded with `/reload`. The extension API exposes all of the pieces needed for a durable bridge:

- `session_start` and `session_shutdown` lifecycle events;
- `ctx.modelRegistry` for resolving models;
- `pi.registerProvider(...)`, which takes effect immediately when called after startup;
- `pi.setModel(model)` for changing the active session model; and
- `ctx.reload()` plus automatic discovery of global extensions.

Source: [Pi extension documentation](https://github.com/badlogic/pi-mono/blob/853a80d26c90a14c1886f0ebb8ffaae133ca2185/packages/coding-agent/docs/extensions.md), [Pi RPC model setter](https://github.com/badlogic/pi-mono/blob/853a80d26c90a14c1886f0ebb8ffaae133ca2185/packages/coding-agent/src/modes/rpc/rpc-mode.ts).

Magnitude should install `~/.pi/agent/extensions/magnitude.ts`. On `session_start`, the extension registers an instance with the Magnitude daemon over an owner-only Unix socket. It should expose `syncCatalog(generation)` and `setModel(provider, model)`. Catalog synchronization can replace the `magnitude` runtime provider; selection can resolve the entry in `ctx.modelRegistry` and call `pi.setModel`.

This gives future ordinary Pi sessions bucket 3, even when the user starts `pi` directly. A Pi process that was already open when the extension was installed can run `/reload` once and then becomes addressable. RPC remains a good alternative when Magnitude intentionally owns the whole UI process, but it is not required.

There is no official Pi desktop app.

## Oh My Pi: bucket 3 with the same bridge pattern

Oh My Pi exposes the same core pattern and adds managed timers that are automatically cleaned up on `session_shutdown`. Its extension types expose `modelRegistry`, `setModel`, `registerProvider`, `unregisterProvider`, `setInterval`, `reload`, and the session lifecycle. Its registry also has a public refresh operation.

Source: [OMP extension API types](https://github.com/can1357/oh-my-pi/blob/65f79e76fcc89b96632fe86a598f314bd7cfc725/packages/coding-agent/src/extensibility/extensions/types.ts), [OMP model registry](https://github.com/can1357/oh-my-pi/blob/65f79e76fcc89b96632fe86a598f314bd7cfc725/packages/coding-agent/src/config/model-registry.ts), [OMP RPC setter](https://github.com/can1357/oh-my-pi/blob/65f79e76fcc89b96632fe86a598f314bd7cfc725/packages/coding-agent/src/modes/rpc/rpc-mode.ts).

Install the equivalent global Magnitude extension and have it maintain an authenticated outbound connection to the Magnitude daemon. Use runtime provider registration for catalog updates and `setModel` for selection. A session opened before installation needs one plugin/runtime reload or restart. After that, ordinary user-launched TUIs are bucket 3.

Oh My Pi has no equivalent official desktop app.

## OpenCode: native bucket 3, including current Desktop

OpenCode has three distinct integration levels.

### Existing ordinary TUI

The ordinary TUI runs an OpenCode server in a Worker using an in-process fetch transport. It listens for `SIGUSR2`; the handler invalidates configuration and disposes all instances so they are rebuilt. Magnitude can therefore make a newly configured model appear in an already-running ordinary TUI by identifying the correct PID and sending `SIGUSR2`. That is a real bucket-2 path, although process/workspace matching must be conservative.

Source: [TUI startup and signal handler](https://github.com/anomalyco/opencode/blob/04284921ac8f657555b5a182f5ff055f471543e4/packages/opencode/src/cli/cmd/tui.ts), [TUI worker reload](https://github.com/anomalyco/opencode/blob/04284921ac8f657555b5a182f5ff055f471543e4/packages/opencode/src/cli/tui/worker.ts).

### Shared server

OpenCode's protocol defines `POST /api/session/:sessionID/model`, implemented as `v2.session.switchModel`. The core publishes a durable `SessionEvent.ModelSwitched`, and subsequent provider turns use the new selection. The generated SDK exposes the same operation. ACP also exposes `unstable_setSessionModel`.

Global config updates through the server API dispose/rebuild instances and emit the global-disposed event, giving a transactional bucket-2 catalog refresh. `opencode serve` starts a headless server and `opencode attach <url>` attaches a TUI to it.

Source: [session protocol](https://github.com/anomalyco/opencode/blob/04284921ac8f657555b5a182f5ff055f471543e4/packages/protocol/src/groups/session.ts), [durable switch implementation](https://github.com/anomalyco/opencode/blob/04284921ac8f657555b5a182f5ff055f471543e4/packages/core/src/session.ts), [attach command](https://github.com/anomalyco/opencode/blob/04284921ac8f657555b5a182f5ff055f471543e4/packages/opencode/src/cli/cmd/attach.ts).

The recommended connector owns one loopback-only, password-protected OpenCode service and launches or encourages `opencode attach`. Magnitude then lists sessions, refreshes the catalog through the config API, and calls the session model endpoint. This is clean bucket 3.

If controlling arbitrary internally hosted TUIs is a hard requirement, a global OpenCode server plugin is a second option. Plugins receive a configured SDK client and the server URL, including the in-process fetch transport. A plugin can register each TUI with Magnitude and relay SDK calls. This is more invasive than the shared-service design but removes the post-hoc reachability problem.

### Desktop

Current OpenCode Desktop starts or reuses a persistent background CLI service. It runs `opencode service status`, starts the service when necessary, and obtains credentials with `opencode service get password`. Magnitude can use the same discoverable service, authenticate, list sessions, refresh configuration, and call `session.switchModel`.

Source: [Desktop background CLI integration](https://github.com/anomalyco/opencode/blob/04284921ac8f657555b5a182f5ff055f471543e4/packages/desktop/src/main/background-cli.ts).

This makes current Desktop a strong bucket-3 target. Older Desktop builds used a random-port sidecar and process-private password; gate support by version and either ask those versions to use a configured shared server or classify them as private.

## Hermes: native bucket 3; use a shared gateway for Desktop

Hermes already has the needed operations. Its JSON-RPC `config.set` handler special cases the `model` key with a `session_id`: it changes an idle session immediately, or queues `pending_model_switch` when a turn is active and applies it on the next safe boundary. Desktop uses this exact operation for its model picker. The REST API also exposes model inventory with explicit refresh and a session model endpoint.

Source: [Hermes TUI gateway](https://github.com/NousResearch/hermes-agent/blob/5a264f9a58c3437b9644fa47434b39f2a146499f/tui_gateway/server.py), [REST API server](https://github.com/NousResearch/hermes-agent/blob/5a264f9a58c3437b9644fa47434b39f2a146499f/gateway/platforms/api_server.py), [Desktop model API](https://github.com/NousResearch/hermes-agent/blob/5a264f9a58c3437b9644fa47434b39f2a146499f/apps/desktop/src/api/config.ts).

The local Desktop default starts `hermes serve --port 0` with a random dashboard token kept in Electron process memory. Scraping that private child is not a sound connector. Hermes Desktop also has a documented Remote Gateway mode. The robust design is for Magnitude to own `hermes serve` at a stable loopback endpoint with an explicit token and configure Desktop to connect to it. Both surfaces then share catalog, sessions, and model selection, and Magnitude has bucket 3 without private discovery.

Source: [Hermes Desktop documentation](https://hermes-agent.nousresearch.com/docs/user-guide/desktop), [Electron backend launch](https://github.com/NousResearch/hermes-agent/blob/5a264f9a58c3437b9644fa47434b39f2a146499f/apps/desktop/electron/main.ts).

## OpenClaw: native bucket 3 on CLI and macOS app

OpenClaw is the cleanest shared-daemon design in this set. Magnitude should stop treating its configuration file as the integration boundary and connect to the Gateway as an authenticated client:

1. call `config.patch` to update the Magnitude provider/catalog;
2. wait for the config reload/version event;
3. call `models.list` to verify the new generation; and
4. call `sessions.patch { key, model }` to persist the per-session override.

The default reload mode is hybrid, so safe changes hot-apply and changes that require a restart are handled by the Gateway lifecycle rather than inferred from file writes.

Source: [OpenClaw model documentation](https://docs.openclaw.ai/concepts/models), [Gateway protocol documentation](https://docs.openclaw.ai/gateway/protocol), [TUI Gateway calls](https://github.com/openclaw/openclaw/blob/e27a72435258a05d09ff5e98b2c2ac2fe18d49f4/src/tui/gateway-chat.ts), [Control UI behavior](https://github.com/openclaw/openclaw/blob/e27a72435258a05d09ff5e98b2c2ac2fe18d49f4/docs/web/control-ui.md).

The macOS app is another client of the same Gateway and includes `models.list`, `config.patch`, and session patch behavior. It shares the exact control plane, so an app session is bucket 3 once Magnitude has Gateway credentials and its session key.

## Codex: bucket 3 with a managed app-server and dynamic catalog

The current connector unnecessarily caps Codex at bucket 1 by writing a startup-only `model_catalog_json`. Codex has a better runtime path.

### Dynamic catalog

Codex's model manager can fetch a provider-owned OpenAI-compatible `GET /models` endpoint. It applies the returned catalog to the running model manager and persists an ETag-aware cache. Runtime refresh occurs on a cache miss and when a model-catalog ETag reported by a provider response changes. The custom provider must use command-backed auth; `has_command_auth()` enables remote model refresh, while a plain `env_key` alone does not.

Source: [model manager refresh logic](https://github.com/openai/codex/blob/b51b07785bcd1545f63893f363ee1957949526e0/codex-rs/models-manager/src/manager.rs), [provider `/models` client](https://github.com/openai/codex/blob/b51b07785bcd1545f63893f363ee1957949526e0/codex-rs/model-provider/src/models_endpoint.rs), [provider auth capability](https://github.com/openai/codex/blob/b51b07785bcd1545f63893f363ee1957949526e0/codex-rs/model-provider-info/src/lib.rs).

Recommended change:

- stop supplying `model_catalog_json` for managed sessions;
- make Magnitude's inference service implement Codex's `/models` response schema and catalog ETag;
- configure the `magnitude` provider with command-backed auth;
- isolate the managed app-server's cache in a Magnitude-owned `CODEX_HOME` or profile;
- on catalog generation change, invalidate only that owned `models_cache.json`, call app-server `model/list`, and verify the returned generation/models.

This makes a new model visible in the same running app-server. No app-server restart is required.

### Addressable session selection

App-server exposes `model/list`; `thread/resume` accepts `model` and `model_provider` overrides; and `turn/start` accepts a model for the next turn. A model override can therefore be applied directly to an existing thread. Official OpenAI documentation now also supports connecting the normal terminal UI to a separately managed app-server:

```text
codex app-server --listen unix:///path/to/magnitude-codex.sock
codex --remote unix:///path/to/magnitude-codex.sock
```

WebSocket transports and authentication token options are supported as well. See the [official OpenAI app-server documentation](https://developers.openai.com/codex/app-server).

The product design should be a persistent, authenticated, Magnitude-owned app-server. Magnitude and the TUI are peers on the same server; Magnitude retains thread IDs and uses next-turn model overrides. That is bucket 3.

### Desktop

Codex Desktop uses app-server internally, but an arbitrary local Desktop task is on a private app-owned connection. Shared configuration layers do not make that connection externally addressable, and the current separate `magnitude` profile is not activated by normal Desktop launches.

There are two support levels:

- **Existing arbitrary local Desktop task:** no documented local post-hoc attach; do not promise live switching.
- **Task hosted by the Magnitude-managed app-server/remote environment:** the same app-server thread is addressable and can be controlled by Magnitude. The Codex app-server daemon is explicitly intended for remote clients including desktop and mobile surfaces, though that daemon lifecycle remains experimental in source.

Source: [app-server daemon README](https://github.com/openai/codex/blob/b51b07785bcd1545f63893f363ee1957949526e0/codex-rs/app-server-daemon/README.md), [Codex configuration documentation](https://developers.openai.com/codex/config-basic).

## Claude Code: bucket 3 only when Magnitude owns the session transport

The Claude Agent SDK process protocol has a real `set_model` control request. The current TypeScript SDK surfaces it as `ClaudeSDKClient.setModel(model)`. In a persistent streaming-input session, this changes the model used by subsequent turns without restarting the process. The installed Claude Code 2.1.175 binary also contains the corresponding validated `set_model` handler and updates session metadata.

Source: [Claude Agent SDK package](https://www.npmjs.com/package/@anthropic-ai/claude-agent-sdk), [Claude Code CLI model option](https://code.claude.com/docs/en/cli-usage).

Magnitude should therefore launch Claude through the Agent SDK, keep the streaming process/control channel alive, assign a stable Magnitude session handle, and call `setModel` at an idle/next-turn boundary. This is bucket 3 for Magnitude-owned Claude sessions even though Claude does not expose a general third-party custom model picker.

Anthropic Remote Control does not solve local programmatic attach. It makes outbound TLS connections to Anthropic and exposes the session through claude.ai or Claude mobile; it does not open a local API for Magnitude. It can be enabled for an existing CLI or VS Code session, but the supported remote clients are Anthropic surfaces.

Source: [Claude Code Remote Control documentation](https://code.claude.com/docs/en/remote-control).

### Desktop

Claude Desktop Code uses the same underlying engine and shared settings/project memory, but Desktop and CLI keep separate histories. Desktop can change its own model through the dropdown, and `/desktop` hands a CLI session to Desktop by saving it and exiting the CLI. Neither feature provides an external setter for an arbitrary open Desktop tab.

Source: [Claude Desktop documentation](https://code.claude.com/docs/en/desktop), [Claude session documentation](https://code.claude.com/docs/en/sessions).

Accordingly:

- a Magnitude-owned Agent SDK session is bucket 3;
- an unrelated ordinary CLI or VS Code session is startup/manual-control only;
- an unrelated Desktop tab cannot currently be promoted to bucket 3 through a supported local API.

Trying to inject terminal keystrokes or automate the Desktop dropdown would be UI automation, not a durable harness integration, and should not drive the architecture.

## Cline: bucket 2 is available now; bucket 3 needs the addressable runtime

The current Magnitude connector creates an isolated `~/.magnitude/harnesses/cline` data root. That is why the user's normal VS Code or JetBrains Cline installation sees none of the Magnitude configuration. Current Cline has moved its applications toward shared file-backed storage under `~/.cline/data/`; comments and migration code explicitly describe CLI, extension, Hub, and desktop consumers sharing `providers.json` and `models.json`.

Source: [Cline storage paths](https://github.com/cline/cline/blob/c4e09725f8887ab8aaaaeac0eb18127c44ad7da9/sdk/packages/shared/src/storage/paths.ts), [provider settings manager](https://github.com/cline/cline/blob/c4e09725f8887ab8aaaaeac0eb18127c44ad7da9/sdk/packages/core/src/services/storage/provider-settings-manager.ts), [local provider registry](https://github.com/cline/cline/blob/c4e09725f8887ab8aaaaeac0eb18127c44ad7da9/sdk/packages/core/src/services/providers/local-provider-registry.ts).

### Bucket 2

Use the standard Cline data root and a unique provider ID such as `magnitude`, rather than taking ownership of the built-in `openai-compatible` entry. Provider settings are re-read and register providers when accessed. Cline's plugin API can also register a provider in-process. A small Magnitude Cline plugin is the cleanest way to push catalog generations into every future runtime; the shared file store is the lower-intrusion fallback.

Raw external writes to `models.json` alone are insufficient for guaranteed immediate refresh: the initial loader is once-per-process, while Cline's own registry write path calls `syncStoredProviderRegistration` to update the live collection. The connector must either use an in-process plugin/API or trigger a supported catalog invalidation, not merely overwrite the file and infer success.

### Bucket 3 surfaces that already exist

Cline already has the model mutation primitive in several places:

- `ClineCore.updateSessionModel(sessionId, modelId)`;
- `LocalRuntimeHost.updateSessionModel`, which updates the session connection;
- ACP `unstable_setSessionModel`; and
- the authenticated Hub command `session.update_connection`, whose payload can include `modelId`.

The Hub has a discoverable record containing its URL and random auth token, and a client can list sessions. Hub-managed sessions are therefore bucket 3 today.

Source: [ClineCore setter](https://github.com/cline/cline/blob/c4e09725f8887ab8aaaaeac0eb18127c44ad7da9/sdk/packages/core/src/ClineCore.ts), [runtime setter](https://github.com/cline/cline/blob/c4e09725f8887ab8aaaaeac0eb18127c44ad7da9/sdk/packages/core/src/runtime/host/local-runtime-host.ts), [Hub discovery](https://github.com/cline/cline/blob/c4e09725f8887ab8aaaaeac0eb18127c44ad7da9/sdk/packages/core/src/hub/discovery/index.ts), [Hub session connection handler](https://github.com/cline/cline/blob/c4e09725f8887ab8aaaaeac0eb18127c44ad7da9/sdk/packages/core/src/hub/server/handlers/session-handlers.ts), [ACP setter](https://github.com/cline/cline/blob/c4e09725f8887ab8aaaaeac0eb18127c44ad7da9/apps/cli/src/acp/acpAgent.ts).

### IDE limitation and shortest path to 3

The normal VS Code extension currently hosts its active local runtime in-process rather than registering that task as a Hub-owned session. Its public extension API exposes task/message/button actions, not model selection. Internally, however, `SdkSessionLifecycle.updateActiveSessionModel` already performs exactly the desired operation and the model-selection event calls it.

Source: [VS Code session lifecycle](https://github.com/cline/cline/blob/c4e09725f8887ab8aaaaeac0eb18127c44ad7da9/apps/vscode/src/sdk/sdk-session-lifecycle.ts), [Cline extension API](https://github.com/cline/cline/blob/c4e09725f8887ab8aaaaeac0eb18127c44ad7da9/apps/vscode/src/exports/index.ts).

The shortest clean options are:

1. upstream `setActiveModel(modelId)` and active-session identification on Cline's public VS Code extension API, then ship a tiny Magnitude companion extension;
2. route IDE sessions through the existing Hub/RemoteRuntimeHost and use `session.update_connection`; or
3. add a narrowly scoped authenticated bridge inside the Cline extension that calls `updateActiveSessionModel`.

Option 1 is the smallest upstream change. Option 2 creates the best long-term shared desktop/CLI control plane. Until one is implemented, classify the normal Cline IDE as bucket 2, not 3.

## Desktop summary

| Desktop/IDE surface | Shares model configuration? | Programmatic switching plan | Result |
|---|---|---|---|
| Magnitude Desktop | Yes, same ACN | Existing ACN model assignment | **3** |
| OpenCode Desktop v2 | Yes, persistent CLI service | Discover service/password; call session endpoint | **3** |
| Hermes Desktop | Yes in shared/remote gateway mode | Magnitude owns `hermes serve`; call `config.set`/REST | **3** |
| OpenClaw macOS/Control UI | Yes, same Gateway | `sessions.patch` | **3** |
| Codex Desktop | Shared config, but ordinary local task is private | 3 only for a task on managed app-server/remote environment | **Conditional 3** |
| Claude Desktop Code | Shared settings, separate history/process | No supported external setter for arbitrary tab | **1 for arbitrary tab** |
| Cline VS Code/JetBrains | Standard data root can be shared | Public API/Hub/bridge addition required | **2 now; 3 with small integration** |

Pi and Oh My Pi have no official equivalent desktop application.

## Recommended implementation sequence

1. **OpenClaw:** replace file-only success with Gateway discovery/auth, `config.patch`, generation verification, and `sessions.patch`.
2. **OpenCode:** integrate the persistent service and direct session model endpoint; support `SIGUSR2` only as a catalog-refresh fallback for ordinary TUIs.
3. **Hermes:** own a stable authenticated `hermes serve` and configure Desktop remote gateway mode against it.
4. **Pi and Oh My Pi:** install global authenticated extension bridges. This turns user-launched future TUIs into addressable sessions rather than forcing RPC-only UX.
5. **Codex:** replace `model_catalog_json` with the live provider `/models` path and move launches to a managed app-server plus `codex --remote`.
6. **Claude Code:** replace one-shot CLI launching with a persistent Agent SDK session owner and call `setModel` through the SDK control channel.
7. **Cline:** first stop using the isolated data root and unique-ify the provider ID; then pursue a small public extension API addition or make IDE sessions Hub-backed.

## Connector contract Magnitude needs

The implementation should not expose a single static bucket. It should return observed capabilities for a particular connected instance:

```text
HarnessConnection
  catalog
    mode: startup | reload | dynamic
    generation: string
    refresh(): Result<ObservedCatalog>
  sessions
    discover(): SessionHandle[]
    setModel(handle, model, expectedGeneration): Result<ObservedSelection>
  desktop
    relation: none | shared-config | shared-control-plane | private-process
```

`setModel` should succeed only after the target reports the intended catalog generation and acknowledges the selected model. For a running turn, the connector must explicitly report whether the change is immediate, queued for the next turn, or rejected.

Each bridge/service must also have:

- owner-only socket/file permissions or loopback authentication;
- a random per-install token, never a predictable unauthenticated port;
- process/session identity and cleanup on shutdown;
- version/capability negotiation;
- idle/streaming state and well-defined next-turn semantics; and
- no credential values in discovery records or logs beyond the minimum local token.

## Validation spikes before shipping

Run one end-to-end spike per distinct mechanism, not merely unit tests around config serialization:

- Pi/OMP: start TUI directly, install/reload bridge, add model, observe picker, switch, and verify the next request's model.
- OpenCode: attach both TUI and Desktop to one service, update catalog, switch one session, and prove the other session is unchanged.
- Hermes/OpenClaw: switch during an active turn and verify documented next-turn behavior plus UI synchronization.
- Codex: update `/models` and ETag in a running app-server, force a scoped cache miss, verify `model/list`, then use a next-turn override on an existing thread.
- Claude: hold a streaming Agent SDK session open across multiple turns and call `setModel` between turns.
- Cline: prove shared provider visibility in an already-open IDE, then prototype either the public extension setter or Hub-backed IDE session before promising bucket 3.

The key product decision is therefore not whether live control is possible in general. It is which harnesses Magnitude will actively own through a daemon/SDK/extension bridge, and whether “existing session” means “created after our integration was installed” or “any private process already open on the machine.” The former can be bucket 3 almost everywhere; the latter is intentionally impossible in several harnesses without an explicit adoption step.
