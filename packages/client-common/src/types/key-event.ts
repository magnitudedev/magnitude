/**
 * Generic app-owned key event interface.
 *
 * Use this only for interactions Magnitude owns across clients, such as
 * suggestion-menu navigation, overlay navigation, and terminal paste fallback.
 * Do not use it as a common text-editing layer for the web/desktop composer:
 * browser and Electron text inputs must keep native selection, clipboard, undo,
 * and IME behavior, while the TUI must implement those editor semantics itself.
 */
export interface KeyEvent {
  name: string
  ctrl: boolean
  meta: boolean
  shift: boolean
  option?: boolean
  defaultPrevented?: boolean
  preventDefault?: () => void
}
