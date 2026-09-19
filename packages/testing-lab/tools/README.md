# Pinned harness tools

Run `npm ci --prefix packages/testing-lab/tools` to install the exact Pi 0.85.1 and OpenCode 1.18.31 clients and their locked dependencies. Worker image builds should install these into an isolated tools directory with this package manifest and lockfile; never consume ambient user installations. The live implementation probes used the same versions installed under `/tmp/magnitude-lab-tools`.

Regenerate the lockfile in an empty directory containing only `package.json`, then verify
`npm ci` there and both CLI versions before copying the lockfile back. Generating it beside
linked local installations can record absolute or relative developer paths that fail on
fresh workers. The lockfile regression test rejects those paths and links.

Hermes is qualified separately from source:

- Repository: https://github.com/NousResearch/hermes-agent
- Commit: `a11bac476bb2a2129fdffca9511d41260fa5f51f`
- Reported version: 0.21.3 (2026.9.14)
- Install: `uv sync --frozen --no-dev --python 3.12` in that exact checkout
- `uv.lock` SHA-256: `811a21647251a3fd024a3e2f49c90ac0600c678e500cc08ed51db38c452a6c65`
- Verified Python: 3.12.11; OpenAI SDK: 2.24.0

Hermes's Tirith startup dependency must finish installation before collecting strict JSONL. The local arm64 probe observed Tirith 0.4.2, executable SHA-256 `873d8834902dbc47f339f79088d0a839f8932f5a8c34dcaedacb60fa6f2d0922`. Platform-specific immutable dependency provisioning is still required; an automatic download of `latest` is not a qualified image lock.

These are test clients, not application runtime dependencies. Installed directories and ephemeral credentials must never enter the application package or source snapshot.

Pi and OpenCode H7 additionally requires `LAB_TERMINAL_NODE_EXECUTABLE` pointing to a real Node.js 24+
executable. The native terminal bridge runs under Node; the lab worker still runs under Bun.
Local Mac ARM64 qualification used Node 26.8.1, Bun 1.4.2 Pi 0.85.1 and OpenCode 1.18.31. Pin and verify the
Node distribution in each worker image; the minimum-version guard is not an image qualification.
Do not resolve Node through Bun's `--bun` PATH shim. Linux/Windows native terminal qualification
and the Hermes terminal journey remain outstanding.
