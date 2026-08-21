---
title: Desktop and web right workspace
status: implemented
applies_to:
  - web/src/app.tsx
  - web/src/components/workspace-panel.tsx
  - web/src/components/browser-panel.tsx
  - web/src/components/project-files/**
  - web/src/hooks/use-open-workspace-file.ts
  - web/src/lib/workspace-panel-layout.ts
  - web/src/lib/workspace-tabs.ts
  - web/src/state/web-atoms.ts
---

# Desktop and web right workspace

## Purpose and ownership

The right workspace is client-owned presentation for inspecting project files and browsing the web
beside a chat. It is one full-height panel that participates in the application flex layout and
reduces the chat's available width rather than covering it. The workspace does not become authority
for project files or browser navigation: project snapshots remain ACN-owned and native browser state
remains Electron-main-owned.

The presentation model owns only mixed tab order, the selected workspace tab, whether the project
tree dock is visible, preferred panel and tree widths, expanded project directories, and unsaved
editor drafts. File tabs identify a Project and optional relative path. Browser tabs identify a
native browser tab. These identities are not interchangeable and no server or native state is copied
into the presentation model.

## Header and tabs

The header contains, in order, the workspace collapse action, a horizontally scrollable mixed tab
strip, a pinned new-tab action immediately after the strip, the project-tree toggle, and the close
action. File and browser tabs share selection, neighbor fallback, close, and keyboard traversal.
Closing an active tab selects its right neighbor when present and otherwise its left neighbor.

The new-tab menu creates either an empty file tab or a browser tab. File is unavailable without a
selected Project, and Browser is unavailable when the platform does not provide embedded browsing.
Opening the workspace for a Project with no compatible tabs creates one empty file tab and opens the
tree so the next file selection fills that tab. Selecting an already-open file activates its existing
tab; otherwise it fills the active empty file tab or appends a new one. Project changes remove file
tabs owned by the previous Project while retaining browser tabs.

Browser creation returns the authoritative native browser-tab ID. Electron may publish its browser
snapshot before or after that command resolves, so reconciliation deduplicates by native ID and
preserves the mixed presentation order. Browser snapshot changes may add popup tabs or remove native
tabs, but cannot replace file tabs or their ordering.

## Layout and tree dock

File content and browser content use one shared preferred panel width of 600 pixels. Viewport-aware
clamping preserves the application's minimum chat and left-sidebar space; narrow windows give the
workspace the available width and suppress an unusable outer resize edge. Collapsing and expanding
the panel uses the same short width transition regardless of active tab, with reduced-motion support.
Changing tabs never replays the panel-open animation.

The project tree is an optional internal right dock. It begins at 200 pixels, is independently
resizable between its own bounds, and consumes space inside the workspace; showing it never increases
the outer panel width. It remains available beside either a file or browser tab. File selection from
the tree opens or activates a file tab while leaving the dock visible.

## Lifecycle guarantees

- Closing the workspace hides the active native browser viewport but retains tabs and editor drafts.
- Switching to a file hides native browser content before rendering the file surface.
- Switching to a browser activates its authoritative native tab and bounds it only to the browser
  content area, excluding the project tree and both resize handles.
- Unsaved file tabs require an explicit discard decision before close. Saved, deleted, or discarded
  drafts are removed by their Project-and-path key.
- A successful filesystem move translates open file paths, draft keys, and expanded directories only
  after ACN acknowledges the move.
- No active tab is a valid empty state; the user can create the next file or browser tab explicitly.

## Acceptance guarantees

- Mixed file and browser tabs can be created, selected, traversed, and closed without duplicate
  native tabs or duplicate open files.
- The new-tab action remains visible when the tab strip overflows.
- Browser form state and file drafts survive switching among workspace tabs.
- The tree can be shown beside either content type, changes only internal content width, and supports
  pointer and keyboard resizing.
- Panel collapse, reopen, project change, native popup creation, final-tab close, and reduced motion
  converge to one valid presentation state without controlled/uncontrolled UI transitions.
