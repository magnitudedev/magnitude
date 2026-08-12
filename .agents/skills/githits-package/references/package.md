# GitHits Package CLI Reference

## Package Info

`githits pkg info <registry:name>` returns latest-version triage: license, description, repository popularity, downloads, publish age, and vulnerability status. Use `--verbose` for GitHub language/topics/last-pushed, recent advisories, and recent changes. Use `--json` for structured fields.

Supported registries include npm, PyPI, Hex, Crates, NuGet, Maven, Packagist, RubyGems, Go, Swift, vcpkg, and Zig.

## Vulnerabilities

`githits pkg vulns <registry:name[@version]>` lists known OSV/CVE advisories. Omit the version for latest.

Flags: `--severity low|medium|high|critical`, `--scope affected|non_affecting|all`, `--include-withdrawn`, `--verbose`, `--json`.

Supported registries: npm, PyPI, Hex, Crates, NuGet, Maven, Packagist, RubyGems, Go, Swift. vcpkg and Zig are unsupported for vulnerability data.

## Dependencies

`githits pkg deps <registry:name[@version]>` lists direct runtime dependencies by default.

Flags: `--lifecycle runtime|development|build|peer|optional|all`, `--depth 1-10`, `--verbose`, `--json`.

Use `--depth` to request transitive output capped to that traversal depth. Omit it for direct dependencies only.

## Changelog

`githits pkg changelog <registry:name>` returns recent release notes, newest first. `--limit` works in latest mode. `--from` switches to range mode, optionally capped by `--to`.

Flags: `--repo-url <url>`, `--from <version>`, `--to <version>`, `--limit 1-50`, `--git-ref <ref>`, `--verbose`, `--no-body`, `--json`.

Do not use `registry:name@version` for changelog. Use `--to <version>`.

## Upgrade Review

`githits pkg upgrade-review <registry:name@current> --to <target>` compares current and target package versions and reports upgrade evidence without assigning risk.

Batch form: `githits pkg upgrade-review --package <registry:name@current>..<target> --package <registry:name@current>..<target>`.

Evidence includes current and target direct vulnerabilities, changelog range evidence, target deprecation metadata, peer dependency changes, dependency changes, and transitive security by default. Dependency-issue diffs are opt-in with `--dependency-issues`.

Flags: `--package <spec>`, `--to <version>`, `--no-transitive-security`, `--dependency-issues`, `--min-severity low|medium|high|critical`, `--verbose`, `--json`.

Use `pkg upgrade-review` for dependency update assessment instead of inferring safety from semver alone. Use `pkg changelog` directly only when you need release notes without a current-to-target comparison.

## Command Name Mapping

- `githits pkg info` maps to MCP `pkg_info`.
- `githits pkg vulns` maps to MCP `pkg_vulns`.
- `githits pkg deps` maps to MCP `pkg_deps`.
- `githits pkg changelog` maps to MCP `pkg_changelog`.
- `githits pkg upgrade-review` maps to MCP `pkg_upgrade_review`.
