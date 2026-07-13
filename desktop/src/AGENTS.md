# Desktop Source Boundary Rules

Desktop code has two different hosts in play. Keep them separate in naming, APIs, and reasoning.

## Client Host

The client host is the machine running the Electron desktop app and renderer.

Client-host capabilities include:

- Electron shell operations such as opening external URLs or revealing/opening client-local paths.
- Electron/native file dialogs.
- Browser clipboard, notifications, menus, window state, and app storage.
- Desktop app bootstrap work such as locating the bundled daemon binary or reading app-local config.

These capabilities are allowed in `desktop/src` when the target is explicitly the user's desktop/client environment. Name variables accordingly, for example `clientPath`, `downloadPath`, or `selectedClientFile`.

Do not delete client-host APIs merely because they involve files. The boundary violation is passing agent-host paths into client-host APIs, or using client-host APIs to inspect agent-host state.

## Agent Host

The agent host is the environment served by the SDK/ACN. It owns sessions and agent-visible filesystem state.

Agent-host state includes:

- Session CWDs and project roots.
- Scratchpad paths and attachments.
- File mentions, file viewer paths, file trees, file contents, file watches, search results, and directory autocomplete.
- Any path that comes from `DisplayState`, session metadata, ACN RPCs, tool output, or agent events.

All agent-host file and directory operations must go through the SDK/ACN. Desktop code must not use Electron, Node `fs`, browser filesystem APIs, or shell APIs to read, list, validate, watch, open, stat, normalize, or infer anything about agent-host files.

## Decision Test

Before adding a file/path operation, ask: "Which host owns this path?"

- If it is an agent/session/project/scratchpad path, use an SDK RPC or add one.
- If it is a desktop/client-local path, use a desktop client capability and name it as client-local.
- If the answer is unclear, treat it as agent-host until provenance is explicit.

Do not assume the desktop client and ACN are on the same machine, even when local development happens that way.
