---
title: Embedded browser
status: implemented
applies_to:
  - desktop/src/embedded-browser*.ts
  - desktop/src/desktop-rpc.ts
  - desktop/src/main.ts
  - desktop/src/platform.ts
  - desktop/src/preload.ts
  - packages/client-common/src/platform/embedded-browser.ts
  - packages/client-common/src/platform/types.ts
  - web/src/components/browser-panel.tsx
  - web/src/components/workspace-panel-header.tsx
  - web/src/lib/workspace-panel-layout.ts
---

# Embedded browser

## Authority and boundary

The embedded browser is a desktop platform capability. Electron's main process owns tab identity,
navigation state, native web-content lifetimes, permissions, downloads, and guest viewport bounds.
The isolated renderer receives schema-validated snapshots and submits semantic commands through the
desktop bridge. Web presentation never owns or directly accesses a guest `WebContents`, and the
standalone web client does not advertise the capability.

Browser tabs are transient application state. They share one persistent browser session for normal
web cookies and storage, but the tab list and active selection do not survive application restart.
At least one tab always exists; closing the final tab replaces it with a blank tab.

## Workspace behavior

Browser and project files are peer surfaces in the full-height right workspace panel. The panel
pushes the chat horizontally rather than covering it. Each surface retains an independent preferred
width, subject to a minimum chat width and viewport-aware bounds. Narrow windows give the panel the
available width. Opening, closing, and surface changes preserve the same reduced-motion and resize
semantics as the project-file surface. The native browser viewport excludes the resize handle's
hitbox, which sits immediately outside the page with its indicator on the panel boundary, so remote
content cannot intercept resizing and no visual gap is introduced.

The browser surface provides a managed tab strip, location field, history, reload or stop, external
open, download activity, and explicit blank, loading, navigation-failure, crash, and insecure-HTTP
states. New-window requests become managed tabs rather than additional Magnitude windows. Browser
keyboard commands operate while guest content has focus.

## Guest isolation and navigation policy

Every remote guest runs sandboxed with Node integration disabled, context isolation enabled, no
Magnitude preload, web security enabled, and insecure mixed content disabled. Remote content never
receives the renderer's desktop bridge.

HTTPS and loopback HTTP navigation may proceed directly. Non-loopback HTTP requires an explicit
continue-once decision scoped to that origin and navigation chain. Privileged or unsupported schemes
are rejected inside the browser surface. A rejected, aborted, or stale navigation completion cannot
overwrite the state of a newer navigation. Same-document history and fragment changes update the
committed address without starting a document-loading phase, and are ignored while a newer document
navigation remains pending.

Only the active, visible, successfully loading or loaded tab owns visible native bounds. Switching
tabs, changing surfaces, collapsing the panel, or showing a permission decision hides the guest
immediately. Reactivating a retained guest raises its native view above the other window content
before making it visible. Hidden guests may be discarded after an idle interval; reactivating a
discarded tab recreates its isolated view and reloads its last committed URL.

## Permissions and downloads

Permission handling is deny-by-default. Harmless browser mechanics may be allowed automatically;
sensitive supported permissions require a Magnitude decision prompt. Allow-once grants are scoped to
one tab, origin, and permission and are cleared by navigation, tab close, timeout, or application
shutdown. Concurrent requests for the same tab, origin, and permission share one decision prompt;
the result completes every associated Electron callback. Unsupported requests are denied, and every
Electron permission callback is completed. Permission checks use the requesting and embedding origins
when Electron does not expose the requesting tab's web contents.

Downloads remain in the browser session and use the operating system's save destination dialog.
Magnitude exposes current progress and terminal status, supports cancellation, and can reveal a
completed download. Downloads never implicitly target the active project. Terminal download history
is bounded so an application-long session cannot retain unbounded native download objects.

## Lifecycle and recovery

The browser owner is an Effect scoped service belonging to one desktop window. Shutdown is
idempotent and releases permission handlers, pending callbacks, download listeners, timers, native
views, and guest web contents before the window is destroyed. Guest crashes are isolated to their
tab and recover through reload. Closing or reopening the workspace panel does not destroy tab state.

## Acceptance guarantees

- Remote pages cannot access Node, Electron, Magnitude preload APIs, or privileged URL schemes.
- Browser truth has one main-process owner and crosses the renderer boundary only through schemas.
- Multiple tabs share their dedicated session while retaining independent navigation and form state.
- Popups become managed tabs, and closing the final tab leaves one blank tab.
- A newer navigation cannot be overwritten by a stale success, failure, or abort from an older load.
- Permission requests are origin-visible, time-bounded, allow-once or denied, and always terminalized.
- Downloads prompt for a destination, report progress and terminal status, cancel, and reveal safely.
- Collapse, surface switching, resize, narrow-window layout, hidden-tab discard, crash, and shutdown
  preserve their stated ownership and cleanup guarantees.
- Navigation, history, redirects, forms, cookies, permissions, downloads, popups, prohibited schemes,
  insecure navigation, tabs, panel reopening, isolation, and shutdown have automated verification in
  a production-built Electron process.
