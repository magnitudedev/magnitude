---
applies_to:
  - packages/release/src/hosted-update/**
  - packages/release/resources/distribution/**
  - packages/release/scripts/build-distribution-server.ts
  - desktop/src/*update*
  - packages/daemon-management/src/desktop-native/linux-update*
  - packages/client-common/src/desktop/update.ts
  - packages/sdk/src/desktop-update.ts
---

# Hosted application update protocol

The hosted distribution boundary separates installation authentication from publisher authenticity.
An installation owns an Ed25519 private key outside its application bundle. The server identifies
it by SHA-256 of its raw public key. That identity proves possession, not a person's identity or
truthful platform reports. Publisher keys are a separate trust set supplied by application builds;
a server response never expands that trust set.

The desktop owner checks after startup and hourly, with bounded jitter and coalesced resume
wakeups. Manual checks reset the deadline. Check admission is independent of a download or prepared
update; a later check cannot replace an active transfer. Automatic downloads default on, with an
atomically persisted preference independent of ACN. Corrupt preferences pause automatic downloads
without replacing the file or suppressing checks. Turning the preference off cancels only automatic
transfers before native staging. Cancellation retains transfer admission until scoped cleanup ends.
Neither renderer disconnection nor window closure owns or cancels this work.

## Signed checks

The request signature matches Ollama: `GET,` followed by the exact request path and query, signed
with Ed25519. Authorization contains the base64 SSH Ed25519 public-key blob and base64 raw signature,
separated by a colon. Query encoding matches Go's sorted form encoding. Request metadata is bounded,
validated and covered by the signature; duplicate or unknown fields are invalid. The HTTP host
admits only its fixed origin, update route and GET method.

Timestamps are admitted within five minutes of server time. Nonces are admitted atomically and
retained beyond the timestamp acceptance window so a concurrent replay cannot record activity
again. Invalid authentication and expired requests return 401; replay returns 409; malformed input
returns 400. Storage or publisher-verification failures return 503, never “up to date.”

An admitted check returns 204 without a body when no qualifying release exists. A 200 response
contains a signed manifest envelope. Checks are private and uncacheable; deployment must not apply
ISR or shared response caching to them.

The deployment edge limits only update/download endpoints before they reach the database. Its
short-lived rate counter is separate from installation analytics; requests rejected at the edge
are not installation observations. Ordinary website pages do not share this endpoint limit.

## Release authenticity

The envelope signs a versioned, domain-separated payload with a trusted publisher Ed25519 key.
The payload binds release version, source commit, target OS/architecture/package, immutable artifact
path, byte count and SHA-256. Paths cannot escape the release object prefix. The client admits only
newer versions compatible with its target and existing stable/beta/alpha channel policy. Candidate
selection chooses the newest compatible version independently of storage enumeration order.

Cryptographic verification does not replace checksum verification after transfer or native
publisher verification before installation. Installation lifecycle belongs to desktop main.
The original signed envelope accompanies a verified candidate through any process handoff; a
parsed manifest is not a substitute for publisher proof at a later privileged boundary.
Windows additionally verifies the installer through WinVerifyTrust with whole-chain revocation
checking and no verification UI. The verified signer's certificate must contain the publisher
organization supplied by the installed build; a valid signature from another publisher is not enough.
Native acceptance uses explicitly trusted, temporary test certificates to exercise valid signatures,
unsigned files, payload tampering and publisher mismatch without weakening production trust.
Windows stages the installer and a copy of the bundled CLI in a unique private directory outside
the replaceable application. The temporary helper acknowledges readiness, waits for its desktop
owner's lifetime pipe to close, then runs the per-user installer and records its actual exit result
before relaunching. It owns no service and requests no elevation. The reopened desktop retires only
that completed staging directory after the helper exits; installation identity remains in the profile.

Artifact paths belong to their signed version directory. Immutable storage publication verifies
the local file, refuses overwrites, and verifies the complete remotely downloaded byte count and
digest before an artifact is eligible for promotion. An existing object requires the same remote
verification; existence alone is not an accepted publication.

The desktop owner creates its installation key only after native ownership is acquired. The key
survives application replacement, remains outside the bundle, and is private to the user. Corrupt
key material causes an explicit failure rather than silently resetting the installation identity.

## Database ownership

Distribution records belong to a private schema, separate from provider customer and billing data.
Public API roles receive no schema/table access. Runtime credentials receive only their necessary
privileges. Release promotion is a publication operation; an ordinary check cannot publish.

Nonce admission precedes release selection and telemetry. A verified check records installation
identity and daily platform/version/country activity, using server time for observation dates.
Country comes only from trusted host composition. Raw IPs, request signatures and private keys are
not telemetry fields. Nonces exist only in the expiring replay table. Daily observations use
atomic upserts so concurrent distinct checks preserve their count. A telemetry-only write failure
may lose an observation but must not suppress an otherwise valid update response.

Authenticated download requests use the same signature and replay admission, including the release
and artifact selectors. Static HTTP endpoints preserve the exact signed path and query; routing
parameters must never be appended to or overwrite signed fields. They resolve only a matching signed artifact and record download intent before returning
an uncached redirect. Authorization is not forwarded to object storage. CDN range requests do not
create additional download records. Request counts do not claim completed transfers or installs.

First installation uses a public installer endpoint with an explicit OS, architecture and package.
It selects only a verified stable installer and records anonymous daily download intent by artifact
and country. A browser download has no installation key and must not be counted as an identified
user. macOS DMGs serve first installation; ZIP archives remain the native update transport.

Detailed installation observations are retained for 90 days, then rolled up without installation
identifiers. Global daily active counts deduplicate across version and country changes; subgroup
distinct counts must not be summed into a global total. Installation identities expire after 180
inactive days. A database-owned scheduled function removes expired replay nonces and performs
retention. The request runtime cannot execute retention or modify release/channel records.

## Website deployment

The website consumes a private, content-addressed bundle of this server implementation. The bundle
contains the same request, manifest and database contracts; the website does not maintain a second
verifier. Native Vercel API routes adapt HTTP requests and trusted edge country metadata, with one
small pooled database connection set per function instance. Database credentials stay server-side,
use a restricted role, and validate the Supabase certificate chain and hostname.

## Acceptance

Tests must reject signature/payload changes, malformed key encodings, expired or duplicate requests,
wrong publishers, incompatible targets and downgrades. Concurrent duplicate requests yield one
admission. Real PostgreSQL checks must verify daily-count upserts and replay uniqueness, not just
an in-memory substitute. Deployments separately validate uncached routes, trusted country metadata
and absence of sensitive request fields from logs.

Native platform protocol probes establish crypto, networking and OS metadata behavior only. Full
platform acceptance additionally requires actual application download, publisher verification,
native replacement, relaunch and preserved service/installation ownership.

Desktop installer transfers allow a one-hour attempt and a bounded three-attempt total, with
the independent sixty-second stall timeout retained. A slow but progressing installer transfer
must not inherit the shorter general artifact-attempt budget. Publication diagnostics distinguish
HTTP failure, interrupted streams, byte-count mismatch and digest mismatch without exposing credentials.
