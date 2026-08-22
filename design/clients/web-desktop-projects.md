---
applies_to:
  - packages/storage/src/**
  - packages/acn-protocol/src/**
  - packages/acn/src/project*.ts
  - packages/acn/src/session-*.ts
  - packages/acn/src/agent-*.ts
  - packages/client-common/src/**
  - cli/src/**
  - web/src/**
  - desktop/src/**
---

# Web and desktop Projects

## Product model

A Project is a durable named registration of one directory. Its branded ID is identity; its `cwd`
is the registered absolute, lexically normalized directory, unique across Project records. A
Project contains no session IDs, session counts, activity, filesystem status, Git status, or cached
inspection state.

Sessions do not carry a Project ID at any boundary. A session is associated with a Project exactly
when `session.cwd === project.cwd`; both sides are the shared branded `DirectoryPath`, the
relationship is derived on demand, and it has no lifecycle of its own. Consequences:

- CLI and TUI sessions appear under an existing Project registered for the same cwd with zero
  write-path work.
- Changing a Project's cwd changes which sessions associate with it without rewriting sessions.
- Removing a Project never mutates sessions; sessions whose cwd has no active Project remain valid,
  inspectable history labeled by their cwd.
- Resuming a session needs no Project lookup: execution resolves from the session's own stored cwd,
  which is immutable session identity.

No service creates or mutates a Project because a session was created or read. Project
registration is always an explicit Project command. The CLI has no Project navigation or commands
and continues to create sessions from cwd.

## Lifecycle and presentation

Project collapse is renderer presentation state. Removing a Project flips its registration state
without deleting source files, sessions, or identity; creating a Project at a removed record's cwd
restores that identity and applies the submitted name. Creating at an already-active cwd returns
the existing record unchanged.

Archiving a session removes it from ordinary navigation without deleting history; archived sessions
remain searchable in Settings and restorable. Pinning is durable session metadata: the sidebar
renders pinned sessions once, in a Pinned section above Projects, newest pin first by the stable
pin timestamp, and excludes them from each Project's nested unpinned list. Archive and pin are
mutually exclusive: archiving clears a pin; pinning restores an archived session in one server
transition. Session rows reveal borderless Pin/Unpin and Archive on hover or focus.

Sidebar loading is intentionally minimal and explicitly paged:

- the first Project page loads through the paginated Project query (twenty per page, server
  recency order); a "Show more projects" control appears only while another page exists;
- each expanded Project loads at most five unpinned sessions initially; its "Show more" requests
  ten more per activation;
- there is no scroll-driven auto-loading and no global bottom-of-sidebar infinite loader; and
- Project rows carry no Git branch or directory-availability adornments. Directory and Git state
  render only where a surface explicitly requests inspection of one Project.

Search is server-bounded: session title/cwd search runs on the session read authority, while
Project-name search filters already-loaded Project records client-side. Search results group under
loaded Projects by cwd; unmatched sessions display their formatted cwd. The Archived-chats page
derives labels the same way and selects "all matching" by following cursors to completion through
query effects, never by requesting an unbounded page.

New Chat opens a draft in the main pane and asks "What would you like to do in {Project}?" with a
searchable Project chooser. The new-chat selection may retain a Project ID as presentation intent,
but session requests carry only the selected Project's cwd; the ID is never persisted in session
metadata and never accepted by session RPCs. Preload release carries the exact draft session ID.

Each Project has a menu for Edit Project, Reveal folder, and Remove Project. Edit reuses the
Project form ("Change the name or folder. Chats stay grouped by the folder they ran in."). Reveal
is an agent-host command that may report itself unsupported. Remove requires confirmation.

## Authority

`ProjectStore` (ACN) owns durable Project records, id/cwd uniqueness inside one serialized durable
transition, and bounded recency-ordered pages with opaque cursors. `ProjectManager` owns lifecycle
commands and validates a newly registered cwd against the host filesystem exactly once, at command
time. `SessionInspector` is the independent read authority for persisted session metadata: bounded
fingerprinted pages, exact-cwd pages served from the session cwd index, and recent-directory
aggregation. The two authorities never depend on each other; clients compose them by cwd.

Directory availability and Git state are host observations, never Project record fields. They are
obtained only through explicit single-Project inspection (`InspectProject`), which runs at most one
bounded Git command sequence and zero Git commands when the directory is unavailable. Nothing polls,
nothing probes `git --version` at startup, and Project listing performs zero filesystem or command
work.

Project and session changes remain independent invalidation domains, and each durable write
publishes exactly one notification. Their notifications are pokes on the connection-global
`StreamChanges` subscription, naming the queries they back, alongside snapshot change pokes;
transport multiplexing does not combine their caches or authority. Clients reread bounded authoritative pages on invalidation and never retain
change events as state. Editing a Project rebinds name and cwd as one minimal record
transition — it does not inspect sessions, release drafts, stop runtimes, or rewrite session cwd.

There is no session-Project migration, no optional compatibility field, and no orphan repair. A
stored session whose metadata cannot be decoded is unreadable: skipped with a structured warning in
pages and reported explicitly by direct reads.

## Client boundaries

Clients import Project contracts only through SDK/client-common and never import ACN, storage, or
Git implementations. Pagination is a query capability: the first page plus explicitly requested
continuations are all reactive query atoms keyed by the complete request identity, deduplicated by
stable ID, and reset by identity change; loading later pages never uses a mutation path. Server
truth never lands in component `useState` and is never synchronized with `useEffect`.

Electron and browser render the same React components. Electron supplies a native directory picker;
a browser file-system handle name is never submitted as cwd. Revealing a Project folder is an
agent-host operation through ACN; Electron never passes a Project path to a client-host shell API.
