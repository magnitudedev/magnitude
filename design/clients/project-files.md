---
title: Project file browsing and editing
status: implemented
applies_to:
  - packages/acn-protocol/src/schemas/project-files.ts
  - packages/acn-protocol/src/boundary/project-files.ts
  - packages/acn/src/project-file-manager.ts
  - packages/acn/src/file-system-manager.ts
  - packages/client-common/src/project-files/**
---

# Project file browsing and editing

## Authority and boundaries

A project is the authority for the source tree visible to web and desktop clients. `ProjectFileManager` resolves a `ProjectId` through `ProjectStore` and opens the Project's cwd through `FileSystemManager`; every file operation then goes through the returned opened-directory capability, which is the single containment implementation. Clients never send an absolute host path and never access the host filesystem directly. The SDK wire contract and client-common state service are the only client boundary.

Project paths are the shared branded `RelativePath`: normalized POSIX-style relative paths. The empty path identifies only the project root directory. Absolute paths, traversal, NULs, Windows separators, and symbolic links (targets and intermediate ancestors alike) are rejected. `.git` is not exposed in directory listings.

## Reading and writing

Directory listing is lazy and sorted with directories before files. Regular UTF-8 text is editable up to 5 MiB. PNG, JPEG, GIF, and WebP images are previewable up to 10 MiB. Other binary or oversized files return an explicit unsupported result instead of being decoded or partially displayed.

Every readable file snapshot has a content-derived `FileContentHash` — a digest of the file's bytes, not a Project revision. A write or deletion supplies the hash it was initiated from as `expectedContentHash`. ACN writes through an exclusive temporary sibling and re-checks the hash in the atomic-write guard immediately before rename. Deletion likewise checks the current hash immediately before removal. A mismatch fails as `ProjectFileChanged` with the complete current text snapshot and never overwrites or removes anything. Mutations coordinate per Project cwd, so unrelated Projects never serialize against each other.

Successful writes and deletions invalidate the exact file and parent-directory queries. While the project-files pane is open, ACN recursively watches that project's source tree and publishes invalidation-only notifications for external changes. Establishing or re-establishing the watch invalidates the project's active snapshots before further notifications are consumed. Directory snapshots have a short idle retention window so recently collapsed and prefetched folders can reopen without another round trip; invalidation marks retained snapshots stale, and observing a stale snapshot rereads ACN authority. Notifications are coalesced and cause active directory or file queries to reread authoritative snapshots; they never carry a second copy of filesystem state. External edits are also detected by content-hash comparison before a write or deletion commits, so stale client state cannot overwrite or remove newer disk state.

## Moving entries

Drag and drop requests a move of one file or directory into another directory in the same Project. It does not express sibling order: directory listings remain directories-first and alphabetically sorted, and a drop into the current parent is rejected as a no-op. The ACN derives the destination name from the source basename; clients cannot use drag and drop to rename an entry.

ACN validates both paths against the Project authority, rejects the project root, symbolic links, destination collisions, and moving a directory into itself or one of its descendants. Project-file mutations are serialized per Project cwd, and the destination is checked immediately before rename so one Magnitude mutation cannot overwrite another. A successful move returns the exact source path, destination path, and entry kind.

The rendered tree may display drag intent, but it never becomes filesystem authority. A controlled tree submits the move to ACN and converges from invalidated directory and file queries. Only a successful acknowledgement may translate client-owned presentation paths such as expanded directories, the selected document, and unsaved draft keys. Rejection leaves those paths unchanged and exposes the mutation failure.

## Acceptance guarantees

- Browsing and editing are scoped to the selected project.
- Reopening the pane and external source-tree changes refresh retained directory and file snapshots without periodic whole-tree polling.
- Initial display reads only the complete root level; deeper restored branches are requested only after their expanded ancestors resolve.
- Hover and keyboard-focus prefetch is silent, bounded, and never recursively materializes a collapsed subtree.
- A stale draft cannot silently overwrite newer disk content.
- Dragging a file or directory moves it only within the same Project and never overwrites an existing destination.
- Dragging within the same directory does not create a fictitious manual order.
- Moving a directory preserves open-document, draft, and expansion presentation paths beneath it after acknowledgement.
- Closing a document does not close the workspace or project tree.
- Browser and document tabs share the workspace width; the internal project tree retains its own width.
- Opening the project pane reduces the chat layout's available width instead of overlaying it.
- The resize edge supports pointer and keyboard interaction.
- No editor assets are loaded from a CDN.
- Code, Markdown source/preview, images, unsupported files, write, delete, and move persistence, conflicts, path traversal, collision, descendant-move, and symlink rejection are covered by automated or live integration verification.
