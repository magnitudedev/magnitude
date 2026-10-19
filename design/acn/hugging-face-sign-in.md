---
applies_to:
  - packages/acn/src/hugging-face-**
  - packages/acn-protocol/src/boundary/hugging-face.ts
  - packages/acn-protocol/src/schemas/hugging-face.ts
  - packages/storage/src/auth/**
  - packages/client-common/src/operations/hugging-face.ts
  - packages/client-common/src/hugging-face/**
  - packages/utils/src/redact-secrets.ts
  - web/src/components/hugging-face.tsx
---

# Hugging Face sign-in

Magnitude has one way to sign in to Hugging Face, shared by the desktop app and the CLI. Signing in
gives access to gated models (after their terms are accepted), private repositories and higher Hub
rate limits. There is no pasted-token field and no `HF_TOKEN` environment variable path anywhere in
the product.

## Ownership

- ACN owns the credential: it runs the sign-in, stores, renews and forgets the tokens, and delivers
  the current access token to ICN. Clients observe only the account state; no token ever appears in
  an RPC response, a URL, a client, or a log.
- Magnitude uses its registered public OAuth app (no client secret) with the scopes
  `openid profile read-repos`. `read-repos` includes gated repositories the account was granted.

## Account state

The account is one of:

- `SignedOut`, optionally with why: `Denied` (declined in the browser), `Expired` (the sign-in wasn't
  finished in time), `Failed`, or `SessionExpired` (the saved sign-in could no longer be renewed);
- `SigningIn` with its method, `Browser` or `Device`;
- `SignedIn` with the Hugging Face username and optional avatar.

A sign-in in progress takes precedence in the state; a failed or cancelled attempt leaves an existing
sign-in untouched. Every change is published on `StreamChanges` so every client converges.

## Flows

Both flows go through ACN RPCs, so the CLI and desktop share one stored credential. Each start
returns an attempt ID; ACN remembers how recent attempts ended (pending, signed in, failed with its
reason, cancelled, superseded, or unknown after a restart), so a client follows its own attempt
rather than inferring it from the account, which an earlier sign-in may still hold. Starting a
sign-in supersedes the attempt in progress and clears the reason an earlier one left; cancelling
ends the attempt in progress, or only the named attempt while it is still in progress. A flow runs
in ACN's own scope, independent of the request that started it, until it signs in, fails, expires,
or is cancelled or superseded.

- **Browser** (desktop app): authorization code with PKCE (S256, a fresh verifier and `state` per
  attempt). ACN listens on an ephemeral `127.0.0.1` port at `/callback`, returns the authorization
  URL for the client to open, ignores callbacks whose `state` doesn't match, exchanges the code, and
  answers the browser with "Signed in to Magnitude, you can close this tab". The attempt expires
  after ten minutes.
- **Device** (CLI, SSH, headless, and the web UI in a browser): ACN requests a device code, returns
  the verification URL, the URL with the code filled in when Hugging Face returns one
  (`verification_uri_complete`; it currently doesn't), user code, lifetime and interval, and polls the token endpoint itself per RFC 8628: it keeps waiting on
  `authorization_pending` or an unreachable server, adds five seconds on `slow_down`, and ends with
  `Expired` when the code expires and `Denied` on `access_denied`.

After either flow, ACN reads the username from the OpenID userinfo endpoint.

## Which flow a client uses

Hugging Face allows only loopback redirects for this public app, so the browser flow completes only
when the browser runs on the same machine as ACN. The desktop app always does (it bundles and owns
ACN), so it uses the browser flow. The web UI can't tell where ACN runs: a browser on another device
reaches it at the server's address, but one using an SSH tunnel reaches it at a loopback origin too,
and ACN sees the tunnel as a local caller. So whenever there is no desktop host (the application
host has no window) the web UI uses the device flow, which is always correct and costs one extra
step. The CLI always uses the device flow.

The web UI's device sign-in (Settings → Hugging Face, and Sign in on a gated-model notice) shows,
while the account is `SigningIn` with `Device` and the code this window started hasn't expired:
"Enter this code on Hugging Face", the user code in large monospace type with Copy, an "Open
huggingface.co" button that opens the URL with the code filled in when there is one and the
verification URL otherwise, and the time the code expires. Cancel stays available; the row then
shows "Signed in as <username>" on success, or the same reason as the browser flow (declined,
expired, failed) when it ends without one. A device sign-in another client started shows only
"Waiting for the code to be entered…" with Cancel, since its code isn't this window's.

## Storage and renewal

- The credential (access token, refresh token when issued, expiry, username, avatar) is stored in
  `auth.json` under its own key beside, never among, the provider credentials. The file is
  owner-only (0600) and every write preserves the other entries.
- With a refresh token, ACN renews the access token five minutes before it expires. A renewal
  Hugging Face refuses signs out with `SessionExpired` and forgets the credential; while Hugging Face
  can't be reached, renewal is retried with backoff (at most every five minutes) and ICN falls back
  to anonymous requests if the token lapses meanwhile. Without a refresh token, the account signs
  out with `SessionExpired` at expiry.

## Delivery to ICN

ACN pushes the current access token (or none) to ICN through ICN's write-only credential endpoint
whenever it changes, and once when ICN starts; ICN applies it without restarting. ICN presents it as
a Bearer on every Hub request (search, listings, header reads, downloads), only to the Hub's own
hosts, retries anonymously when the Hub rejects it, and never reads a token from its environment or a
token file. See [catalog and acquisition](../model-management/catalog-and-acquisition.md).

What one credential read of a gated or private repository is never shown to another. ICN keys its
saved reads of such a repository's files (an inspection's files and headers, a search row's model
shape) by the credential that made them, anonymous included, so after a sign-out or a sign-in as
another account the next read asks the Hub again and gets that account's answer; public repositories
keep one shared cache. ICN also drops its in-memory search results on every credential change. Once
ICN has a different account's token, or none, ACN pokes clients to refetch search results,
inspections and drafter lists; a renewed token for the same account pokes nothing.

## Gated repositories

A gated repository's refused files are told apart by the Hub's answer: 401 means sign-in is needed,
403 with `X-Error-Code: GatedRepo` means the signed-in account hasn't accepted the terms (or awaits
the author's approval). Clients get `HuggingFaceAccessRequired` with the repository and `SignIn` or
`AcceptTerms`:

- `SignIn`: "Sign in to Hugging Face to use this model"; the desktop offers Sign in inline, the CLI
  names `magnitude hf login`.
- `AcceptTerms`: "Accept this model's terms on Hugging Face"; the desktop opens the repository page
  and offers Try again, the CLI prints the repository URL.

A repository the Hub won't show at all is worded by whether Magnitude is signed in.

## Redaction

Anything matching `hf_[A-Za-z0-9_]+` and every `Authorization` credential is redacted from log files
and ICN's output; tokens are held as redacted values in memory, and HTTP tracing redacts the
`Authorization` header.
