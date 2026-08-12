# GitHits Repository Integration

Magnitude uses GitHits' standalone CLI skills for public open-source research. The skills invoke
the existing shell tool; this repository does not add an MCP runtime or store GitHits credentials.

GitHits requires internet access and authentication. A `githits` executable on `PATH` is preferred;
the skills can fall back to the published npm CLI. Automation can provide `GITHITS_API_TOKEN`.
GitHits queries and public package or repository targets are sent to GitHits services, while skill
installation itself does not upload the local workspace. Start a new agent session after changing
these files so skill discovery and repository instructions are refreshed.

## Upstream

- Repository: `https://github.com/githits-com/githits-cli`
- Baseline commit: `8a8179db2be887d563510416bfcb312fc4508b58`
- Canonical paths: `skills/githits-code/**` and `skills/githits-package/**`

Keep the vendored files aligned with upstream. Review skill files, references, command changes, and
security guidance together when updating the baseline. Avoid local changes unless Magnitude has a
documented compatibility requirement.
