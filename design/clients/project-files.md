---
title: Project file browsing and editing
status: implemented
applies_to:
  - packages/acn-protocol/src/schemas/project-files.ts
  - packages/acn-protocol/src/rpcs/project-files.ts
  - packages/acn/src/project-files.ts
  - packages/client-common/src/project-files/**
  - web/src/components/project-files/**
  - web/src/lib/workspace-panel-layout.ts
  - web/src/state/web-atoms.ts
  - web/src/app.tsx
---

# Project file browsing and editing

## Authority and boundaries

A project is the authority for the source tree visible to web and desktop clients. The ACN resolves a `ProjectId` through `ProjectRegistry`; clients never send an absolute host path and never access the host filesystem directly. The SDK wire contract and client-common state service are the only client boundary.

Project paths are normalized POSIX-style relative paths. The empty path identifies only the project root directory. Absolute paths, traversal, NULs, Windows separators, and symbolic links are rejected. `.git` is not exposed in directory listings.

## Reading and writing

Directory listing is lazy and sorted with directories before files. Regular UTF-8 text is editable up to 5 MiB. PNG, JPEG, GIF, and WebP images are previewable up to 10 MiB. Other binary or oversized files return an explicit unsupported result instead of being decoded or partially displayed.

Every readable file snapshot has a content-derived revision. A write or deletion supplies the revision it was initiated from. ACN writes through an exclusive temporary sibling and performs a second revision check immediately before atomic rename. Deletion likewise checks the current revision immediately before removal. A mismatch returns the complete current text snapshot as a conflict and never overwrites or removes it.

Successful writes and deletions invalidate the exact file and parent-directory queries. While the project-files pane is open, ACN recursively watches that project's source tree and publishes invalidation-only notifications for external changes. Establishing or re-establishing the watch invalidates the project's active snapshots before further notifications are consumed. Directory snapshots have a short idle retention window so recently collapsed and prefetched folders can reopen without another round trip; invalidation marks retained snapshots stale, and observing a stale snapshot rereads ACN authority. Notifications are coalesced and cause active directory or file queries to reread authoritative snapshots; they never carry a second copy of filesystem state. External edits are also detected by revision comparison before a write or deletion commits, so stale client state cannot overwrite or remove newer disk state.

## Moving entries

Drag and drop requests a move of one file or directory into another directory in the same Project. It does not express sibling order: directory listings remain directories-first and alphabetically sorted, and a drop into the current parent is rejected as a no-op. The ACN derives the destination name from the source basename; clients cannot use drag and drop to rename an entry.

ACN validates both paths against the Project authority, rejects the project root, symbolic links, destination collisions, and moving a directory into itself or one of its descendants. Project-file mutations are serialized, and the destination is checked immediately before rename so one Magnitude mutation cannot overwrite another. A successful move returns the exact source path, destination path, and entry kind.

The rendered tree may display drag intent, but it never becomes filesystem authority. A controlled tree submits the move to ACN and converges from invalidated directory and file queries. Only a successful acknowledgement may translate client-owned presentation paths such as expanded directories, the selected document, and unsaved draft keys. Rejection leaves those paths unchanged and exposes the mutation failure.

## Client state and interaction

Project file queries and writes use the shared connection-scoped AtomRpc client through client-common. Web presentation state owns only whether the pane is open, expanded tree branches, the selected relative path, and unsaved editor drafts keyed by project and path. Server snapshots and directory listings are not copied into presentation atoms.

The project pane is one surface of the full-height right-hand workspace panel. When collapsed, its toggle belongs to the app titlebar's upper-right controls. When expanded, that same control is the first item in the pane header, directly before the surface selector. The pane participates in the horizontal application layout and reduces the chat's available width rather than covering it, matching the left sidebar's layout behavior. Opening and closing animate that width over the same short duration as the left sidebar, while reduced-motion preferences suppress the transition. Its inner edge is pointer- and keyboard-resizable within viewport-aware bounds. The file tree, document, and browser retain independent presentation widths; the tree begins narrowest, a document opens wider, and the browser begins widest. Viewport clamping never overwrites any preference. Narrow layouts use the full width and do not expose a meaningless resize edge. Its tree is the default file view; selecting a file replaces the tree with the document and Back restores it. The complete root listing loads before the tree appears. Folder discovery is demand-loaded one directory level at a time: restored expansion proceeds breadth-first through successful parent listings, while a short pointer-hover or keyboard-focus dwell silently prefetches likely next folders. Prefetch never recursively walks a collapsed subtree. A cold expanded folder shows loading in its disclosure control rather than masquerading as empty, and collapsing it does not immediately discard or interrupt the listing. Text uses locally bundled Monaco. Markdown defaults to formatted preview with a source mode; MDX remains source. Saving is explicit through Save or platform Mod-S. Conflicts remain in the pane and use Monaco diff presentation. File actions provide revision-checked removal behind an explicit confirmation dialog.

## Acceptance guarantees

- Browsing and editing are scoped to the selected project.
- Reopening the pane and external source-tree changes refresh retained directory and file snapshots without periodic whole-tree polling.
- Initial display reads only the complete root level; deeper restored branches are requested only after their expanded ancestors resolve.
- Hover and keyboard-focus prefetch is silent, bounded, and never recursively materializes a collapsed subtree.
- A stale draft cannot silently overwrite a newer disk revision.
- Dragging a file or directory moves it only within the same Project and never overwrites an existing destination.
- Dragging within the same directory does not create a fictitious manual order.
- Moving a directory preserves open-document, draft, and expansion presentation paths beneath it after acknowledgement.
- Closing a document does not close the project pane.
- Browser and document widths remain independently adjustable within responsive bounds.
- Opening the project pane reduces the chat layout's available width instead of overlaying it.
- The resize edge supports pointer and keyboard interaction.
- No editor assets are loaded from a CDN.
- Code, Markdown source/preview, images, unsupported files, write, delete, and move persistence, conflicts, path traversal, collision, descendant-move, and symlink rejection are covered by automated or live integration verification.
