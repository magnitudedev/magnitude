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

A Project is the stable, named owner of one source directory and the sessions created for it. Its
branded ID is identity. Its name and canonical absolute source directory are editable, while the
current source directory is unique across Project records.

Every persisted session has exactly one Project ID. The Project's current source directory is the
authority for future session execution. A source rebind therefore applies to every session in the
Project; session metadata cannot independently retain a competing current execution directory.

Projects are a web and Electron product surface. The CLI has no Project navigation or commands and
continues to create sessions from cwd. ACN resolves that cwd to the unique Project, creating a
default basename-named Project when necessary, so the storage invariant holds for every client.

## Lifecycle and presentation

Project collapse is renderer presentation state. It hides child sessions and has no server
transition. Removing a Project marks its registration removed from the ordinary sidebar and Recent
Projects without deleting source files, sessions, or Project identity. Selecting the same source
through New Project restores that record.

Closing a session is non-destructive sidebar membership. Deleting a session remains a distinct
destructive operation.

The ordinary browser sidebar order is:

1. sidebar expand/collapse and New Chat compose icon toolbar;
2. Search Sessions;
3. full-row New Project; and
4. the expandable Project/session hierarchy.

Electron places sidebar expand/collapse and New Chat in the persistent native title-bar row rather
than adding a second toolbar inside the sidebar. While expanded, the sidebar background and right
border continue through the full title-bar height and the two actions sit at the right edge of that
region; the current session title starts in the main pane. When collapsed, the sidebar region and
border disappear and those actions move beside the native window controls, followed by the session
title. In dark mode the main pane uses slate-900 and the sidebar uses opaque slate-850 on every
desktop platform; native vibrancy does not alter the sidebar's rendered color. macOS reserves the
configured traffic-light region. Linux and Windows use Electron's Window
Controls Overlay safe-area variables because native controls may appear on either side according to
the desktop environment. A collapsed Electron sidebar has no residual rail: the main pane takes the
available width and Settings and appearance controls float at bottom-left. The composer keeps the
same wide, centered maximum measure whether the sidebar is expanded or collapsed, so collapsing
changes its position within the main pane rather than its width. The narrower new-chat chooser
retains its own independent reading measure. Settings keeps the same application chrome but does
not display the session title as a Settings page heading.

New Chat opens a Project-owned draft in the main pane. It retains the current available Project or,
when no valid selection exists, defaults to the most recently active available Project. The empty
draft asks “What would you like to do in {Project}?” as one sentence; the matching inline Project
name is underlined and opens the searchable Project chooser. The full recent-Project list is not
part of the default empty state. New Project
opens a modal first. The modal's Select source action invokes the host directory selector, fills the
editable Project name from the selected basename, and requires explicit Create new or Cancel. A
successful creation makes that Project the current new-chat selection.

The new-chat selection retains the Project ID as its stable intent across subsequent New Chat
actions. Preload and create requests carry that ID, and ACN resolves the Project's current source at
execution time; a copied renderer path is display context only and cannot become a competing source
authority.
Preload release carries the exact draft session ID so cleanup from an older source selection cannot
release a replacement draft for the same Project and client owner.

Each Project has a vertical-ellipsis menu for Edit Project, host-supported reveal, and Remove
Project. Edit reuses the Project form. Remove requires a separate confirmation dialog.
Creating with an already-active canonical source returns that existing record without changing its
name. Creating with a removed source restores the retained identity and applies the submitted name.

## Authority

ACN owns Project records, canonical path uniqueness, session association, edit/remove/restore
transitions, current execution-directory resolution, and the product Project summary. Storage owns
the durable Project state document and required session Project ID.

Directory availability and Git repository/head information are observations of host state. They
are derived by ACN and never stored as Project truth or reconstructed by clients. Git state supports
ordinary branches, detached HEAD, nested source directories, and worktrees.
ACN observes directory and Git changes and emits bounded invalidation notifications; clients always
reread the complete Project snapshot rather than retaining watch events as state.

Client-common exposes the ACN-backed Project capability and invalidates authoritative snapshots when
ACN announces a change. Web owns only presentation state such as the selected draft Project,
collapsed IDs, open dialog/menu, and sidebar width/collapse.

## Source rebinding

Renaming does not affect session lifecycle. Source rebinding serializes against new session work,
rejects while a Project session is working, releases Project draft preloads, and retires idle live
runtimes before the new source becomes available for resume. It never interrupts agent work.

The mutation validates the new directory and uniqueness before committing. After commit, all new or
resumed execution resolves through the new source and Project directory/Git observations are
invalidated.

## Storage migration

Existing session metadata is migrated idempotently before normal Project-backed session use:

1. group legacy sessions by canonical working directory;
2. ensure one Project per directory with the directory basename as its default name; and
3. atomically add the exact Project ID to each session metadata record.

The migration boundary may decode the legacy optional field. The normal domain requires Project ID;
there is no permanent optional fallback. Missing source directories preserve Project and session
history while preventing new execution. The legacy session working-directory field remains only as
migration provenance and a recomputable index input; it is never consulted as current Project source
authority. A session that references an absent durable Project is a visible state-integrity failure;
recovery does not guess a replacement identity or source.

## Client boundaries

Clients import Project contracts only through SDK/client-common. They never import ACN, storage, or
Git implementations. Project queries are observational. Mutations synchronize against the exact
Project identity and expected refreshed state. Renderer state never becomes a second authority for
Project records, branch, directory availability, or session membership.

Electron and browser render the same React components. Electron supplies a native directory picker.
A browser file-system handle name is not an absolute daemon path and must never be submitted as cwd;
without a trusted host bridge, standalone browser source selection uses daemon-visible absolute path
entry. Revealing a Project source is an agent-host operation submitted through ACN; Electron never
passes a Project path to a client-host shell API.
