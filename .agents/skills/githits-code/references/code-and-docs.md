# GitHits Code And Docs CLI Reference

Package target syntax requires an explicit registry: `registry:name[@version]`, for example `npm:express@5.2.1`; omit `@version` for the latest release. Repository compact targets use `github:org/repo[#ref|@ref]`, `github.com/org/repo[#ref|@ref]`, or `https://github.com/org/repo[#ref|@ref]`; omitted refs request the backend default-branch intent. Output uses canonical `github:org/repo#ref` formatting so refs can contain `@` safely. `code` commands also support `--repo-url <url> [--git-ref <ref>]`.

## Search

`githits search "<query>" --in <target>` searches indexed dependency code, docs, and symbols. Repeat `--in` for multiple targets. Use `--source code`, `--source docs`, or `--source symbol` to force a source; omit it for auto-routing.

Useful filters: `--kind`, `--category`, `--path-prefix`, `--intent`, `--public`, `--name`, `--lang`, `--limit`, `--offset`, `--wait`, `--allow-partial`, `--json`.

If search returns a `searchRef`, do not repeat the original search. Continue with `githits search-status <searchRef> [--wait 0-60]`; the bounded wait defaults to 20 seconds.

## Code Files

`githits code files <spec> [path-prefix]` lists paths. Use this before `code read` when you do not know the exact file path.

Useful filters: `--path`, repeatable `--glob`, repeatable `--ext`, repeatable `--file-type`, repeatable `--language`, repeatable `--file-intent`, repeatable `--exclude-intent`, `--exclude-docs`, `--exclude-tests`, `--hidden`, `--limit`, `--wait`, `--verbose`, `--json`.

## Code Read

`githits code read <spec> <path>` reads one exact package-relative file. Use `--lines 10-80`, `--start`, or `--end` for focused windows. You can also append a range to the path: `src/index.js:10-80`.

For repository addressing: `githits code read --repo-url <url> [--git-ref <ref>] <path>`.

## Code Grep

`githits code grep <spec> <pattern> [path-prefix]` runs deterministic text grep. Use `--regex` for RE2 regex, `--case-sensitive`, `-C`, `-A`, `-B`, `--path`, repeatable `--glob`, repeatable `--ext`, `--exclude-docs`, `--exclude-tests`, `--limit`, `--per-file-limit`, `--cursor`, `--symbol-field`, `--wait`, `--verbose`, `--json`.

Use `search` for discovery and `code grep` only when you know the pattern.
When grep returns no matches, do not repeat it unchanged. Change or shorten the pattern, broaden the path/filter scope, or switch to `search` for conceptual intent.

## Docs

`githits docs list <spec>` browses available documentation pages. It is not topic search.

`githits docs read <pageId>` reads a page. Text output returns at most 150 lines per call, including larger explicit ranges; continue from the reported returned range when more context is needed. Use `--lines` for bounded windows and `--json` when extracting `totalLines` or source metadata.

For topic search, use `githits search "<topic>" --source docs --in <target>`, then pass the returned page ID to `docs read`.

When search reports partial documentation coverage, treat the evidence as incomplete and retry later when advised. Capped coverage means the current crawl stopped at a limit; report that limitation instead of retrying.

## Command Name Mapping

- `githits example` maps to MCP `get_example`.
- `githits languages` maps to MCP `search_language`.
- `githits search` maps to MCP `search`.
- `githits search-status` maps to MCP `search_status`.
- `githits code files` maps to MCP `code_files`.
- `githits code grep` maps to MCP `code_grep`.
- `githits code read` maps to MCP `code_read`.
- `githits docs list` maps to MCP `docs_list`.
- `githits docs read` maps to MCP `docs_read`.
