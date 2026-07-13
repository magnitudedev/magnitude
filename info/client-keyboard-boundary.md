# Client Keyboard Boundary

Magnitude has two different keyboard environments:

- Web and desktop render real browser controls. Native text editing, selection,
  clipboard, undo/redo, IME, and accessibility behavior should stay native.
- The TUI has no native text input. It must implement terminal editor behavior
  itself from OpenTUI key events.

## Desktop/Web

Desktop and web should treat browser text inputs as the owner of normal editing
keys. Do not intercept `Cmd/Ctrl+A`, `Cmd/Ctrl+C`, `Cmd/Ctrl+V`, `Cmd/Ctrl+Z`,
word movement, selection movement, or IME-related input in React unless there is
a concrete browser limitation.

Electron desktop still needs the standard `Edit` menu roles. Those roles install
the native accelerators for the focused web contents:

- `undo`
- `redo`
- `cut`
- `copy`
- `paste`
- `pasteAndMatchStyle`
- `delete`
- `selectAll`

The desktop main process owns those roles through the application menu. The
renderer should not reimplement them with global `keydown` handlers.

Web and desktop may use DOM keyboard handlers for app-owned commands only:

- Escape behavior
- send-message submission
- slash/file mention suggestion navigation
- app commands such as new session, transcript mode, settings, or search focus

These handlers must avoid stealing standard editing chords from focused inputs.

## TUI

The TUI receives terminal key events rather than browser-native text input. Its
composer is responsible for editor behavior: cursor movement, deletion,
selection-like segment handling, paste fallback, history navigation, and submit
handling.

TUI key handling should stay in CLI/TUI source unless it is pure app intent
state that does not encode terminal editor semantics.

## Client-Common

`client-common` may contain shared state machines and contracts for app-owned
interactions:

- slash command filtering and suggestion selection
- file mention filtering and suggestion selection
- recent-chat list navigation
- paste ingestion coordination, currently consumed by the TUI paste layer
- platform menu action types

The `KeyEvent` type in `client-common` is only a small structural contract for
those app-owned interactions. It is not a shared text-editor abstraction.

Avoid adding generic composer editing classifiers to `client-common`. A common
classifier tends to flatten real platform differences and can accidentally make
web/desktop imitate terminal behavior.

## Known Watch Point

`Cmd/Ctrl+R` is currently used as an app shortcut for focusing sidebar search
while Electron also has a standard reload role. That duplicate accelerator should
be resolved intentionally rather than left ambiguous.
