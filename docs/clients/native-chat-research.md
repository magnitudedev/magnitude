# Research: Native Chat and Open WebUI embedding in Magnitude

**Date:** 2026-10-17  
**Status:** research documented; Open WebUI embedding is untested; PR #202 remains a draft.

## Goal

Add a conversational experience reachable from Magnitude's sidebar while keeping chat separate from administration. Preserve local-first behavior, avoid surprise cloud fallbacks or prompt telemetry, and keep the canonical Magnitude service intact.

## Options

### A. Native Magnitude Chat

[Upstream PR #202](https://github.com/magnitudedev/magnitude/pull/202) proposes a first-party Chat page that reuses Magnitude's web components and inference API.

The working branch includes:

- A Chat destination in the shared navigation contract and web sidebar.
- A model selector populated with exact IDs from `/inference/v1/models`.
- OpenAI-compatible requests to `/inference/v1/chat/completions`, with SSE streaming and incremental rendering.
- Cancellation, new conversations, browser-local history via `localStorage`, conversation selection/deletion, and error states.
- An optional additional loopback listener on `127.0.0.1:10101` that opens Chat directly; the canonical service/API remains on `127.0.0.1:10100`.
- Documentation for local privacy and troubleshooting.

**Validation status (2026-10-17):** the fork's GitHub Actions run #10 passed dependency installation, `client-common`/web/ACN typechecks, web tests, and production build on head `bdcd27756ff80edd0f1163303cfa1fad016bc031`. Separate local smoke evidence recorded for this branch confirms MiniCPM5 Q4 returned visible non-streaming content and a real SSE stream reconstructed the expected response. These checks validate compilation, unit tests, and the API protocol—not browser-level UI behavior. Browser E2E, cancellation, history lifecycle, occupied-port behavior, cross-model compatibility, and real Open WebUI embedding remain unverified.

### B. Embed Open WebUI in a Chat page

Proposed flow:

`Magnitude → sidebar Chat → local Open WebUI iframe → Magnitude's OpenAI-compatible API`

This could reuse Open WebUI's conversation history, model selector, and streaming instead of maintaining two chat clients. Magnitude would remain the local inference engine and administration surface.

Illustrative iframe markup:

```html
<iframe
  src="http://127.0.0.1:3000/"
  title="Open WebUI Chat"
  loading="lazy"
  style="width: 100%; height: 100%; border: 0;"
></iframe>
```

The URL and port are examples only and must be confirmed against the actual installation. This snippet is not a production-ready integration.

## iframe and responsive sizing findings

Distinguish three features:

1. **The `<iframe>` element:** broadly supported by modern browsers.
2. **Lazy loading via `loading="lazy"`:** broadly supported and useful when Chat should not load until opened.
3. **New intrinsic responsive iframe sizing:** Chrome 154 announced `frame-sizing: content-height`, `window.requestResize()`, and the opt-in `<meta name="responsive-embedded-sizing" content="allow-origins=*">` on September 22, 2026. The consulted references mark these as limited/experimental availability, with no confirmed Firefox or Safari support.

A chat should occupy the available container height rather than expand to fit the entire conversation. Start with `width: 100%; height: 100%` and, where appropriate, `loading="lazy"`. Do not make the new sizing feature a requirement.

### Primary references

- [Chrome for Developers — New in Chrome 154](https://developer.chrome.com/blog/new-in-chrome-154) (2026-09-22).
- [MDN — CSS `frame-sizing`](https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Properties/frame-sizing).
- [MDN — `responsive-embedded-sizing`](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/meta/name/responsive-embedded-sizing).
- [MDN — `window.requestResize()`](https://developer.mozilla.org/en-US/docs/Web/API/Window/requestResize).
- [MDN — iframe element](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/iframe).
- [MDN — iframe lazy loading](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/iframe#loading).
- [Open WebUI — Hardening](https://docs.openwebui.com/getting-started/advanced-topics/hardening/).
- [Open WebUI — Environment configuration](https://docs.openwebui.com/reference/env-configuration/).

## Compatibility and security questions

Browser iframe support does not guarantee that Open WebUI allows itself to be framed. Before selecting this option, test:

1. **Server headers:** inspect `Content-Security-Policy` (especially `frame-ancestors`) and `X-Frame-Options`. Open WebUI hardening settings may deny embedding. Do not globally disable security protections as a shortcut; prefer an explicit, narrowly scoped and reviewed configuration.
2. **Authentication, cookies, and sessions:** different ports are different origins; cookie policies and separated storage can affect login.
3. **API connectivity:** configure Open WebUI to use Magnitude's OpenAI-compatible endpoint. If Open WebUI runs in Docker, `127.0.0.1` inside the container points to the container itself, not necessarily the host.
4. **Actual streaming:** test a full generation, not just page rendering.
5. **Lifecycle:** clearly report when Open WebUI is missing or unreachable; do not install or start services unexpectedly.
6. **Network exposure:** bind to loopback by default, keep secrets out of URLs, and do not expose an unauthenticated API to the network.
7. **Port 10101 conflicts:** verify availability. Magnitude's canonical API must remain on 10100; the optional listener must not terminate or replace another process.

## What was found in Magnitude's code

A search of Magnitude's default branch found an `iframe` reference in `packages/dom-extract/src/mappings.ts`, where it is listed as a DOM element examined during extraction. Searches for `frame-sizing` and `iframe component` did not identify a UI component for embedding external applications. This **does not prove that no recent work exists on another branch or PR**; identifying the specific commit/PR remains open.

## Validation plan

- [ ] Identify the recent Magnitude iframe change and confirm whether it is a UI component rather than DOM extraction or browser automation.
- [ ] Run Open WebUI locally and inspect HTTP headers, redirects, and authentication.
- [ ] Configure Open WebUI against Magnitude's local API and verify model listing and real generation.
- [ ] Test embedding in Chromium and Firefox; inspect console, cookies, session, and streaming.
- [ ] Verify conversation history and model selection in the embedded interface.
- [ ] Check `127.0.0.1:10100` health and `10101` behavior, including the occupied-port case.
- [ ] Run typecheck, tests, and web build for PR #202.
- [ ] Based on evidence, decide whether to simplify native Chat, adopt Open WebUI, or keep native Chat as an alternative/fallback.

## Provisional decision

**Do not replace native Chat with an iframe yet.** First verify that Open WebUI permits safe embedding, preserves sessions, and communicates correctly with Magnitude. Keep the experience local-first and make the integration opt-in or show a clear unavailable state when the external component is missing.

The architectural goal remains to separate conversation UI from administration, as in ODS, without duplicating features before technical feasibility is demonstrated.

## Work links

- [Upstream PR #202 — Native Chat](https://github.com/magnitudedev/magnitude/pull/202).
- [LEONES design and tracking branch](https://github.com/robertosantosx2/LEONES/tree/feat/magnitude-native-chat).
- [Current status](https://github.com/robertosantosx2/LEONES/blob/feat/magnitude-native-chat/docs/magnitude-native-chat/STATUS.md).
- [Open WebUI + iframe ADR](https://github.com/robertosantosx2/LEONES/blob/feat/magnitude-native-chat/docs/magnitude-native-chat/ADR-openwebui-iframe.md).


## Implementation update (2026-10-17)

The working branch now includes an opt-in Open WebUI view within the Chat page. It accepts only loopback HTTP(S) URLs (`localhost`, `127.0.0.1`, or `[::1]`), rejects URL credentials and non-web protocols, strips URL fragments, and persists the selected URL in browser local storage. A native Chat route remains available as a fallback. The URL validation helper has unit tests for accepted loopback URLs, remote-host rejection, credentials, malformed URLs, non-web schemes, and fragment removal.

The fork's CI typechecks, tests, and production build passed on commit `bdcd27756ff80edd0f1163303cfa1fad016bc031`, including unit tests for Open WebUI URL validation. A later mobile-history UX change has been committed separately and needs a fresh CI run. Passing unit tests do not establish that an actual Open WebUI instance can be embedded or authenticate successfully. The iframe cannot reliably inspect a cross-origin child's HTTP headers or authentication state, so server policy and real browser behavior remain manual integration checks. Do not claim Open WebUI embedding or inference has passed until the real-instance end-to-end checks are green.


The implementation guide for connecting an existing Open WebUI instance to Magnitude is available at [`docs/clients/open-webui.mdx`](https://github.com/robertosantosx2/magnitude/blob/feat/native-chat-section/docs/clients/open-webui.mdx). It explains the host-versus-container address distinction, API key/network-access requirements, embedding policy, and end-to-end acceptance checklist.


## Validation ledger (2026-10-17)

| Check | Result | Evidence / limitation |
|---|---|---|
| Fork CI: dependency install, three typechecks, web tests, production build | Pass | [Run #10](https://github.com/robertosantosx2/magnitude/actions/runs/39023175172), current PR head at time of run |
| SSE parser unit tests and Open WebUI URL validation tests | Pass | Included in the successful web test step |
| Real non-streaming API request | Pass for MiniCPM5 Q4 | Returned visible assistant content with `reasoning_effort: "none"` |
| Real SSE API request | Pass for MiniCPM5 Q4 | Incremental chunks reconstructed expected text and ended with `[DONE]` |
| Native Chat browser UI, Stop, localStorage lifecycle | Pending | Requires running the actual web app in a browser; API smoke alone is insufficient |
| Port 10101 occupied while 10100 remains healthy | Pending | Requires runtime conflict test |
| Cross-model/backend support for `reasoning_effort: "none"` | Pending | Only MiniCPM5 Q4 was confirmed |
| Open WebUI iframe, CSP, cookies/session, API and streaming | Pending | URL validation tests do not prove a real embedded session |
| Upstream maintainer review and upstream CI | Pending | Fork CI does not substitute for upstream checks |

The PR should remain a draft until the pending runtime acceptance checks are completed. The standalone page previously available on port 10101 is not evidence that this PR's Native Chat component works in-browser.


## Mobile usability follow-up (2026-10-17)

The mobile header now exposes both **New chat** and a **Conversation history** selector, because the desktop history sidebar is hidden on narrow screens. This closes the navigation gap where mobile users could create a conversation but could not switch back to another saved conversation. The change is in commit [c655dfe](https://github.com/robertosantosx2/magnitude/commit/c655dfe5175138dfa425ab006def8f4b5c325e9e). It still requires a fresh CI run and a real-browser check at narrow viewport sizes.
