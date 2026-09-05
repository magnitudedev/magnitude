# Magnitude integrations for OpenCode, Pi, and Hermes

**Feasibility review - 2026-09-02**

## Executive conclusion

The proposed architecture is sound, but it will not deliver the complete feature on all three current stable harnesses without qualification.

| Harness | Run Magnitude CLI | Slash commands | Native status UI | Read live Magnitude loading/prefill stream | Overall |
|---|---:|---:|---:|---:|---|
| Pi 0.84.4 | Yes | Yes | Yes, footer status | Yes, through a supported provider stream override | **Green** |
| OpenCode 1.18.27 | Yes | Yes | Yes, prompt status row | No supported server-to-TUI path | **Yellow** |
| OpenCode V2 beta full TUI | Yes | Yes | Yes, `prompt.footer.status` | Yes, native response hook plus typed RPC events | **Green, but beta** |
| OpenCode V2 beta Desktop | Yes, in the shared server plugin | Yes, as server-side commands | No third-party native renderer/status slot | Yes, in the shared server plugin | **Backend green; native Desktop UI blocked** |
| Hermes 0.21.0 terminal | Yes | Yes | No contributed status segment | Yes, through a provider-supplied client interceptor | **Yellow; stock-terminal presentation seam missing** |
| Hermes 0.21.0 Desktop | Yes | Yes | Yes, desktop status bar | Yes, through a provider-supplied client interceptor | **Green with a unified Python/Desktop plugin** |

The important distinction is between three separate capabilities:

1. **Machine control** - invoking `magnitude models status`, `load`, and `stop` with a stable JSON result.
2. **Inference observation** - opting into and reading Magnitude-specific SSE progress without disrupting the harness's normal OpenAI-compatible parser.
3. **UI projection** - rendering that observed state in a supported, native status/footer contribution.

Every harness permits process execution. The compatibility differences are in capabilities 2 and 3.

## Magnitude's current protocol and CLI

Magnitude already has nearly the right inference observation protocol:

- Requests opt in with `Magnitude-Include-Progress: true`.
- Streaming responses then include progress-only SSE chunks with `choices: []`.
- Phases are `model_loading`, `queued`, `preparing`, `prefill`, and `generating`.
- `model_loading` carries `fraction`.
- `prefill` carries `completed_tokens`, `total_tokens`, and `cached_tokens`.
- Terminal timing metadata includes `prompt_ms`, `time_to_first_token_ms`, prompt token counts/rates, and generation timing.

Two consequences matter for the integrations:

- Generic OpenAI-compatible parsers commonly discard the progress chunks because they have no `choices[0]`. A plugin must tee/clone the raw response stream before the stock parser consumes it.
- Magnitude does not currently report a terminal, authoritative model-load duration. A plugin can show **live elapsed loading time** from its first `model_loading` event, but it cannot truthfully show an exact completed `model_load_ms` unless Magnitude adds that metric. Exact prefill duration is already available as `timings.prompt_ms`.

The current CLI exposes `models status [model-id]`, `models load <model-id>`, and `models stop`, but no structured-output flag. Current connector installation only mutates provider configuration; `--install-skill` is separate and no plugin ownership is recorded.

Relevant local sources:

- `inference/crates/icn-api/src/protocols/chat.rs`
- `packages/icn-protocol/src/generated/schemas.ts`
- `packages/icn/src/provider/source.ts`
- `cli/src/commands/inference.ts`
- `cli/src/features/model-setup/inference-runtime.ts`
- `cli/src/harness-connections/service.ts`
- `cli/src/features/model-setup/harness.tsx`

## Pi

### Verdict

Pi supports the complete design through documented extension and provider APIs. No Pi upstream change is required.

### Shell and slash commands

`ExtensionAPI.exec(command, args, options)` launches a process without a shell and returns structured `stdout`, `stderr`, exit code, and termination state. This is an appropriate boundary for `magnitude ... --json`: pass an argument array and decode the returned JSON.

`pi.registerCommand` registers slash commands. Recommended commands are namespaced to avoid collisions, for example:

- `/magnitude-models`
- `/magnitude-load`
- `/magnitude-stop`

### Status UI

`ctx.ui.setStatus(key, text)` updates an extension-owned line in Pi's built-in footer and immediately requests a render. This is a native status surface, not a toast or synthetic chat message.

### Completion telemetry

Pi's ordinary provider hooks are insufficient:

- `before_provider_request` can alter the request payload.
- `before_provider_headers` can add the Magnitude opt-in header.
- `after_provider_response` exposes status and headers, not the response body.
- Pi's stock OpenAI-compatible parser ignores empty-choice progress chunks.

The clean solution is a provider override for Magnitude's provider ID with a custom `streamSimple` implementation. It should delegate semantic parsing to Pi's exported stock OpenAI-completions `streamSimple`, while supplying a custom `fetch` function that:

1. Adds `Magnitude-Include-Progress: true`.
2. Calls the original fetch.
3. Clones or tees the response.
4. Parses only Magnitude progress/timing extensions from one branch.
5. Returns the untouched branch to Pi's stock parser.

This preserves Pi's message, tool-call, usage, and error semantics while giving the extension live loading and prefill state for `setStatus`.

### Packaging and lifecycle

Publish an npm package such as `@magnitudedev/pi-extension` with the `pi-package` keyword and `pi.extensions` entry. Pi's documented lifecycle is:

- Install: `pi install npm:@magnitudedev/pi-extension`
- Remove: `pi remove npm:@magnitudedev/pi-extension`
- Reload the running process with `/reload`, or restart it.

Pi extensions execute with the user's permissions, so CLI failures should be surfaced explicitly and never treated as successful state changes.

Primary sources:

- [Extension execution and provider registration](https://github.com/earendil-works/pi/blob/4e69b0c28060f0f02fbe38bfa7c21a2e2eb25057/packages/coding-agent/src/core/extensions/types.ts#L693-L714)
- [Commands and UI extension API](https://github.com/earendil-works/pi/blob/4e69b0c28060f0f02fbe38bfa7c21a2e2eb25057/packages/coding-agent/src/core/extensions/types.ts#L1252-L1300)
- [Pi stream options, including custom fetch](https://github.com/earendil-works/pi/blob/4e69b0c28060f0f02fbe38bfa7c21a2e2eb25057/packages/ai/src/api/simple-options.ts#L17-L49)
- [OpenAI-compatible stream parser](https://github.com/earendil-works/pi/blob/4e69b0c28060f0f02fbe38bfa7c21a2e2eb25057/packages/ai/src/api/openai-completions.ts#L551-L569)
- [Footer extension statuses](https://github.com/earendil-works/pi/blob/4e69b0c28060f0f02fbe38bfa7c21a2e2eb25057/packages/coding-agent/src/modes/interactive/components/footer.ts#L229-L243)
- [Pi package format and installation](https://github.com/earendil-works/pi/blob/4e69b0c28060f0f02fbe38bfa7c21a2e2eb25057/packages/coding-agent/docs/packages.md#L18-L66)

## OpenCode

### Stable 1.18.27 verdict

Stable OpenCode can execute Magnitude, register slash commands, install a dual server/TUI npm package, and render in the real prompt status row. It cannot carry raw Magnitude progress from the provider/server runtime to the TUI through a supported stable API.

### Shell and slash commands

The server plugin context includes a typed Bun shell helper. It supports cwd/environment control, text/line/JSON decoding, and non-throwing execution. The package is loaded as ordinary unsandboxed JavaScript under the OpenCode user's authority.

The TUI plugin API registers commands through a keymap layer, including a slash name and handler. Use TUI-native selectors/dialogs for model selection and invoke the JSON CLI through a small process adapter.

### Status UI

Stable OpenCode's TUI plugin API exposes `session_prompt_right`. OpenCode passes this slot to the right side of the prompt footer, where it is rendered on the same bottom metadata row as the model/provider information. This is the correct surface for live Magnitude state. `app_bottom` also exists, but is a less precise placement below the route.

### Why live progress does not reach the stable TUI

The server-side `chat.headers` hook can add Magnitude's progress opt-in header. However:

1. OpenCode's OpenAI-compatible provider parser reads `choices[0]` and drops a chunk when that choice is absent.
2. Its raw-chunk path is special-cased for GitHub Copilot metadata, not exposed as a generic plugin event.
3. The stable server plugin and TUI plugin run in separate workers.
4. Their public bridge carries fixed OpenCode events and UI actions; it does not support plugin-defined events or RPC.

A plugin-owned file, socket, or localhost bridge could make stable work, but that is additional infrastructure and lifecycle complexity, not the proposed minimal reuse of Chat Completions. Encoding telemetry into text/provider metadata or abusing toast/command events would be patchwork and should not be used.

### V2 beta verdict

OpenCode V2 beta cleanly supports the complete feature in its full-screen terminal TUI:

1. `model.request` adds `Magnitude-Include-Progress`.
2. `http.response` receives the native, one-shot `Response`; the plugin clones/tees it and parses Magnitude SSE.
3. The server package declares a typed custom RPC event and emits phase updates.
4. The TUI package subscribes through `context.client.rpc(...)`.
5. It renders state in `prompt.footer.status`.

This is the most direct implementation of the intended architecture, but V2 is explicitly beta and its package/API versions must be isolated from a stable integration release.

V2 is a client/server generation, not the name of one UI. The bare `opencode2` command launches the full-screen TUI; `opencode2 run` is noninteractive; `opencode2 mini` is a separate minimal terminal client; and Desktop is an Electron wrapper around the shared Solid web renderer. These clients connect to the same background service.

That separation matters for plugins. A package's main entrypoint is loaded by the background server and can observe the raw model response, execute Magnitude, and expose RPC methods/events regardless of whether the request originated in the TUI or Desktop. Its `./tui` entrypoint is loaded only by the full terminal client. The published plugin package has no Desktop/web renderer entrypoint, and the shared app renderer has no third-party status-slot host. Consequently, the integration can capture loading, prefill, and token-rate data for Desktop sessions, but a package cannot render those values in Desktop's native header/status popover through the supported V2 plugin API today.

Desktop's built-in terminal panel does not change this result. It is a Ghostty-Web emulator connected to a server-created PTY over a WebSocket. It runs an ordinary shell. If the user starts `opencode2` inside it, the TUI and its `prompt.footer.status` contribution appear nested inside the terminal panel; that is not a contribution to Desktop's native UI.

### Packaging and lifecycle

Stable 1.18.27 supports a single npm package exporting both `./server` and `./tui`; `opencode plugin <module> --global` installs the package and patches the corresponding configs. Stable has no first-class plugin removal command, so Magnitude must remove only its owned entries from both configs itself, or deactivate the integration.

V2 has first-class plugin add/list/check/update/remove commands. If both tracks are published, use separate compatibility declarations and release channels; do not make one package silently branch across materially different APIs.

Primary sources:

- [Stable shell context](https://github.com/anomalyco/opencode/blob/4b7e19e315cca414121ba1d61523fef74bb3ae8b/packages/plugin/src/index.ts)
- [Stable shell helper contract](https://github.com/anomalyco/opencode/blob/4b7e19e315cca414121ba1d61523fef74bb3ae8b/packages/plugin/src/shell.ts)
- [Stable TUI slots](https://github.com/anomalyco/opencode/blob/4b7e19e315cca414121ba1d61523fef74bb3ae8b/packages/plugin/src/tui.ts)
- [Stable prompt status-row placement](https://github.com/anomalyco/opencode/blob/4b7e19e315cca414121ba1d61523fef74bb3ae8b/packages/tui/src/routes/session/index.tsx#L1330)
- [Stable plugin installer](https://github.com/anomalyco/opencode/blob/4b7e19e315cca414121ba1d61523fef74bb3ae8b/packages/opencode/src/cli/cmd/plug.ts)
- [V2 native HTTP hooks](https://opencode.ai/v2/docs/build/plugins/#native-http)
- [V2 typed plugin RPC and events](https://opencode.ai/v2/docs/build/plugins/rpc/)
- [V2 CLI status slots](https://opencode.ai/v2/docs/build/plugins/cli/#slots)
- [V2 CLI, full TUI, Mini, and shared background service](https://opencode.ai/v2/docs/cli/)
- [V2 Desktop starts the shared background service](https://github.com/anomalyco/opencode/blob/beta/packages/desktop/src/main/service/background-service.ts)
- [V2 Desktop mounts the shared app renderer](https://github.com/anomalyco/opencode/blob/beta/packages/desktop/src/renderer/desktop-app.tsx)
- [V2 plugin package exports, including `./tui` and no Desktop renderer entrypoint](https://github.com/anomalyco/opencode/blob/beta/packages/plugin/package.json)
- [Desktop/web Ghostty terminal implementation](https://github.com/anomalyco/opencode/blob/beta/packages/app/src/session/terminal/terminal.tsx)
- [Desktop/web status popover reads the server plugin list](https://github.com/anomalyco/opencode/blob/beta/packages/app/src/shell/status/body.tsx)

## Hermes

### Verdict

Hermes plugins can run Magnitude and add slash commands. Hermes's ordinary lifecycle hooks do not expose Magnitude's raw completion extension fields, but its model-provider plugin API has a lower-level escape hatch: `ProviderProfile.create_client()` may supply the OpenAI-compatible client used by the transport. A Magnitude provider profile can therefore return a normal OpenAI client backed by an observing HTTP transport, parse the raw SSE bytes as they pass through, and yield those bytes unchanged to Hermes. This makes the progress metadata accessible without patching Hermes.

Hermes Desktop can then present the information cleanly through a unified plugin's Python backend plus plugin-scoped REST/WebSocket namespace and a Desktop status-bar contribution. The stock terminal TUI remains the limitation: general plugins do not receive a supported status-segment or wait-notice publisher.

### Shell and slash commands

Hermes general plugins execute in-process with full user permissions and may use Python `subprocess`. Slash commands are registered with `ctx.register_command`. A plugin can therefore invoke `magnitude ... --json` and expose model controls.

### Completion progress access

The strongest ordinary post-request hook exposes normalized response data, request duration/timestamps, first-chunk time, and canonical usage. During response construction, Hermes creates a selective response dictionary containing model, finish reason, assistant content/tool calls, and normalized usage. Arbitrary top-level `timings` fields and usage extensions are discarded, so `post_api_request` alone is insufficient.

There is also no raw streaming-chunk observer hook. The streaming delta hook contains normalized text/reasoning deltas, not the original SSE object. However, a model-provider plugin may override `ProviderProfile.create_client()`. The returned client can use an observing HTTP transport that sees Magnitude's progress-only SSE frames before the OpenAI SDK and Hermes discard their unknown fields. It should forward the original byte stream unchanged and publish only the decoded Magnitude progress state into plugin-owned state.

Hermes also has `llm_execution` middleware, but that wrapper surrounds Hermes's complete streaming call: `next_call()` does not return until Hermes has consumed the stream. It is useful for final cleanup or correlation, not for directly observing raw live chunks.

### Status UI limitation

- Hermes Desktop supports first-class `statusBar.left` and `statusBar.right` plugin contributions. A unified package can expose progress from its Python backend over the plugin-scoped REST/WebSocket namespace and render it in that bar.
- Hermes terminal configuration exposes visibility controls for built-in status fields, not a plugin status-segment registry. Hermes documents protected `HermesCLI` subclass hooks for wrapper CLIs, including `_get_extra_tui_widgets()`, but installing a normal plugin does not insert such a widget into the stock command.

A unified Hermes plugin repository may contain both the Python plugin and desktop `plugin.js`, but the desktop half has its own enable state. A remote server plugin cannot contribute UI to a local desktop client.

### Packaging and lifecycle

The documented third-party distribution path is a standalone Git plugin rather than npm:

- Install and enable: `hermes plugins install owner/repo --enable --ref <40-char-sha>`
- Remove: `hermes plugins remove <name>`

The installer performs a security scan that can prompt or block. The plugin is disabled by default without `--enable`. Removal can remove the unified desktop half, but installing/enabling the Python plugin does not automatically enable the desktop half in the Desktop application.

### Hermes terminal choices

There are three possible terminal approaches:

1. Contribute a terminal plugin status-segment or wait-notice publisher. This is the cleanest stock-Hermes solution and is the only upstream seam still strictly needed.
2. Ship a wrapper CLI subclass using Hermes's documented `_get_extra_tui_widgets()` hook. This is supported but requires users to launch the wrapper rather than the normal Hermes executable.
3. Rewrite progress frames temporarily into reasoning deltas so Hermes's existing `thinking.delta` path paints its status line, then strip those synthetic deltas from the final response in execution middleware. This can work, but it overloads reasoning as a UI protocol and is too brittle to recommend as the durable architecture.

Hermes already shows `avg_tps` in sufficiently wide terminal status bars. That value is a rolling calculation of output tokens divided by total API latency over recent calls, so it includes loading and prefill; it is not Magnitude's precise generation-only `predicted_per_second`. Desktop can show Magnitude's exact rate through its own contribution. The stock terminal needs the same small presentation seam if exact generation throughput is required there.

Primary sources:

- [Plugin permissions and subprocess capability](https://github.com/NousResearch/hermes-agent/blob/593aa74c6182ce2e5e23bc102daaaae71710c05d/website/docs/developer-guide/plugins/index.md#L185-L198)
- [Slash-command registration](https://github.com/NousResearch/hermes-agent/blob/593aa74c6182ce2e5e23bc102daaaae71710c05d/website/docs/developer-guide/plugins/index.md#L1145-L1188)
- [Post-API-request hook](https://github.com/NousResearch/hermes-agent/blob/593aa74c6182ce2e5e23bc102daaaae71710c05d/agent/conversation_loop.py#L7420-L7467)
- [Selective response construction](https://github.com/NousResearch/hermes-agent/blob/593aa74c6182ce2e5e23bc102daaaae71710c05d/run_agent.py#L3111-L3125)
- [Normalized usage construction](https://github.com/NousResearch/hermes-agent/blob/593aa74c6182ce2e5e23bc102daaaae71710c05d/run_agent.py#L3297-L3323)
- [Desktop status-bar contributions](https://github.com/NousResearch/hermes-agent/blob/593aa74c6182ce2e5e23bc102daaaae71710c05d/website/docs/developer-guide/desktop-plugin-sdk.md#L98-L124)
- [Unified Python/Desktop packages](https://github.com/NousResearch/hermes-agent/blob/593aa74c6182ce2e5e23bc102daaaae71710c05d/website/docs/developer-guide/desktop-plugin-sdk.md#L693-L722)
- [Model-provider client override](https://hermes-agent.nousresearch.com/docs/developer-guide/model-provider-plugin#overridable-hooks)
- [Documented wrapper-CLI widgets](https://hermes-agent.nousresearch.com/docs/developer-guide/extending-the-cli#_get_extra_tui_widgets)
- [Plugin install/remove CLI](https://github.com/NousResearch/hermes-agent/blob/593aa74c6182ce2e5e23bc102daaaae71710c05d/hermes_cli/subcommands/plugins.py#L16-L92)

## Recommended Magnitude architecture

Keep the three integrations independent, but give them the same four conceptual layers.

### 1. Existing RPC and a private bundled SDK

Model controls use the existing daemon RPC through the Effect-native Magnitude SDK. The SDK is a
private workspace dependency bundled into the TypeScript plugin, not a separately published
package. It handles fixed-endpoint admission, exact RPC-version checks, and recovery.

An injected starter may invoke `magnitude service start`; model operations never scrape CLI output
or use a second CLI JSON contract. Pi implements this path. OpenCode can consume the same TypeScript
SDK; a Python Hermes adapter would require contract-derived bindings, not handwritten duplicate
schemas. Those integrations remain future work.

### 2. Shared inference-observation semantics

Define one small semantic state model used by all adapters:

- `idle`
- `loading { fraction, elapsedMs }`
- `queued`
- `preparing`
- `prefill { completedTokens, totalTokens, cachedTokens, elapsedMs }`
- `generating`
- `complete { modelLoadMs?, promptMs?, ttftMs? }`
- `failed { message }`

Each plugin should tee the raw stream and translate only Magnitude extensions into this model. It should never reimplement the harness's full OpenAI-compatible semantic parser.

If exact completed loading time is a product requirement, add `model_load_ms` to Magnitude's terminal timing schema. Until then, label the UI as elapsed time while loading.

### 3. Harness-native projection

- Pi: `ctx.ui.setStatus`.
- OpenCode stable: `session_prompt_right`, but only after choosing an upstream or IPC data bridge.
- OpenCode V2 beta full TUI: `prompt.footer.status` fed by typed RPC events.
- OpenCode V2 beta Desktop: no supported third-party renderer slot; an upstream app contribution is required for native display.
- Hermes Desktop: `statusBar.left` or `statusBar.right`, fed by a provider-client interceptor through the unified plugin backend.
- Hermes terminal: wait for or contribute a terminal status-segment API, or deliberately ship a separately invoked wrapper CLI.

Status state must be keyed by request/session where the host supports concurrent work. Clear it on completion, cancellation, error, plugin reload, and disconnect.

### 4. Installation ownership and transactions

Treat plugin installation as part of a plugin-capable connection's desired state, separate from the optional Magnitude skill.

The connection manifest should record:

- harness and scope;
- exact package/repository spec and installed version/ref;
- whether Magnitude installed it or it pre-existed;
- prior enabled/configuration state;
- which connection records currently depend on it.

`connections add` should install/enable the plugin and roll it back if later connection setup fails, but only when Magnitude created it. `connections remove` should remove/restore it only when the last dependent Magnitude connection is removed. Never uninstall a package the user already owned before Magnitude connected it.

Interactive onboarding should explain that plugin code runs with the harness user's authority, show the exact package/source, perform installation, and report whether the running harness must reload or restart. Hermes Desktop's separately enabled UI half must be shown as a distinct result.

## Repository and release layout

The proposed layout is appropriate:

```text
integrations/
  opencode/
  pi/
  hermes/
```

Use separate versions and publish flows because the ecosystems differ:

- Pi: npm Pi package.
- OpenCode: npm dual server/TUI package; keep stable and V2-beta compatibility explicit.
- Hermes: Git-native plugin release, optionally submitted to the community index; include the Desktop contribution in the same repository if desired.

The repository's current root workspace pattern does not include `integrations/*`. Either add that workspace intentionally or keep each integration independently installable and releasable. CI is not technically required, but compatibility smoke tests against pinned harness versions are strongly advisable because all three plugin surfaces are evolving.

## Recommended execution order

1. Formalize the JSON CLI schemas and plugin ownership lifecycle in Magnitude.
2. Implement Pi end to end as the reference integration.
3. Implement Hermes's provider-client interceptor and Desktop status item; propose the small terminal presentation API, or deliberately choose the wrapper-CLI tradeoff.
4. Implement the OpenCode V2 beta server/TUI package. Treat native Desktop display as a separate upstream UI-extension proposal.
5. Add an authoritative `model_load_ms` terminal timing only if the UI must report completed loading duration rather than live elapsed time.

## Final decision

Proceed with the overall design, but do **not** present all three integrations as equally implementable today.

- **Pi:** ready for full implementation.
- **OpenCode stable:** controls and status UI are ready; live telemetry transport is missing.
- **OpenCode V2 beta full TUI:** full design is ready, subject to beta-version risk.
- **OpenCode V2 beta Desktop:** the same server plugin can observe progress and provide controls, but native display needs an upstream renderer extension point.
- **Hermes Desktop:** fully feasible with a unified provider/general/Desktop plugin package.
- **Hermes terminal:** progress can be captured, but the stock plugin surface cannot contribute its own status segment; exact Magnitude generation throughput has the same presentation limitation.

This conclusion is based on official documentation and source-level inspection of the versions/commits identified above, plus inspection of Magnitude's current CLI, connectors, and Chat Completions protocol.
