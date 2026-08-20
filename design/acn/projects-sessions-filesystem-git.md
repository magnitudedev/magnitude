---
title: ACN Project storage, session inspection, filesystem, and Git boundaries
status: implemented
applies_to:
  - packages/acn-protocol/src/schemas/paths.ts
  - packages/acn-protocol/src/schemas/project.ts
  - packages/acn-protocol/src/schemas/session.ts
  - packages/acn-protocol/src/schemas/git.ts
  - packages/acn-protocol/src/errors.ts
  - packages/acn/src/project-store.ts
  - packages/acn/src/project-manager.ts
  - packages/acn/src/project-inspector.ts
  - packages/acn/src/session-inspector.ts
  - packages/acn/src/file-system-manager.ts
  - packages/acn/src/git-inspector.ts
  - packages/acn/src/project-file-manager.ts
  - packages/acn/src/file-mention-searcher.ts
  - packages/storage/src/sessions/**
---

# ACN Project storage, session inspection, filesystem, and Git boundaries

## The two authorities and the derived join

`ProjectStore` owns durable Project records: get, find-by-cwd, bounded recency-ordered pages with
opaque Schema-decoded cursors, and inserts/updates whose id and cwd uniqueness is decided inside
the storage document's serialized transition (domain conflicts are returned as `Either` from the
pure transition and re-raised, never smuggled through `instanceof` probing). The Store never
validates host directories, restores records, inspects sessions, or runs Git — its Layer requires
only `MagnitudeStorage`, so host access is structurally impossible.

`SessionInspector` is the ACN read model for persisted session metadata — deliberately not a
"SessionStore": it owns no persistence, directories, runtime state, or lifecycle commands. It
serves `get`, fingerprinted session pages, and recent-directory aggregation. Exact-cwd pages read
the session cwd index (maintained move-to-front under the metadata write lock) and only enough
records to fill the page plus its continuation probe; global pages read metadata with bounded
concurrency. A record whose timestamps or schema fail to decode is unreadable: skipped with one
structured warning in pages, reported explicitly by `get`, never coerced into a ranking.

The authorities never depend on each other. A session is associated with a Project exactly when
`session.cwd === project.cwd` (both the shared branded `DirectoryPath`); the join is derived on
demand by clients. Session metadata contains no Project ID at any boundary, there is no migration
or compatibility machinery, and nothing implicitly creates a Project from session activity.

Session cursors are base64url JSON decoded through one Schema transformation and carry a
fingerprint of the request predicates (not the limit); a cursor reused under different predicates
fails as invalid rather than silently restarting. Both authorities publish invalidation-only change
streams (`StreamProjectChanges`, `StreamSessionChanges`) after successful durable transitions.

## Commands, queries, observations

`ProjectManager` performs lifecycle commands: it normalizes names, converts wire cwd strings with
`FileSystemManager.normalizeDirectory`, validates a newly registered cwd by opening it once, and
performs the minimal Store transition. Create at an active cwd is idempotent; create at a removed
record's cwd restores that identity with the submitted name; remove flips registration state only;
edit rebinds name and cwd without touching sessions, drafts, or runtimes.

`ProjectInspector` is the only host observation of a Project: given one Project ID it inspects the
directory (one stat) and invokes Git only when the directory is available. Listing never inspects;
no service owns a polling fiber; no startup Git probe exists. Directory and Git state are closed
tagged value unions in the success channel, because a missing directory or absent Git executable is
a normal state the caller renders, not an operation failure.

## Filesystem

`FileSystemManager` is the single ACN filesystem service in this scope, implemented on
`@effect/platform` FileSystem/Path with platform errors translated once, by tag — never through
`unknown` property probing or `String(error)`. It exposes lexical path algebra (`DirectoryPath`,
`AbsolutePath`, `RelativePath` brands), directory/path observations, host-scope session file
operations (read, watch with a per-subscription polling fallback, subdirectory listing for the
directory picker), reveal, and `openDirectory`.

`openDirectory` returns an immutable `OpenedDirectory` value — the single containment
implementation. Every contained operation takes a branded `RelativePath` (the brand excludes
absolute paths, traversal, NULs, and separator ambiguity), walks existing ancestors rejecting
symlinks, and never follows symlink targets. It provides atomic temp-sibling writes with a
pre-commit guard, non-overwriting moves, bounded breadth-first file discovery, and recursive
watches. `ProjectFileManager` composes `ProjectStore` + `OpenedDirectory` for project file
browsing/editing with `FileContentHash` stale-write protection and per-Project-cwd keyed mutation
locks. `FileMentionSearcher` composes `FileSystemManager` + `GitInspector` (plus optional ripgrep
through `CommandExecutor`, falling back to bounded traversal) for the general file RPCs and
composer mention discovery.

## Git

`GitInspector` runs read-only Git through the injected `CommandExecutor` with `LC_ALL=C`, capturing
exit code, stdout, and stderr, bounded by a three-second timeout. Outcomes map to closed unions:
executable missing → `git_unavailable`; the stable `fatal: not a git repository` diagnostic →
`not_git_repository`; timeout or any other failure → `git_inspection_failed`; success parses the
repository root and branch or detached revision. Recent-file discovery is one bounded `git log`
whose non-repository outcomes are returned to the caller, never swallowed into an empty list. The
service runs nothing at construction, during listings, or on an interval, and never mutates a
repository.
