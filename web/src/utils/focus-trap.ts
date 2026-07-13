/**
 * Focus trap utility — traps Tab key within a container element.
 * Spec Appendix: "Modal dialogs should trap focus and close on Esc."
 *
 * Usage: pass the returned handler as `onKeyDown` to the modal container.
 * The container should have `tabIndex={-1}` so it can receive focus.
 */
import type { KeyboardEvent } from "react"

const FOCUSABLE_SELECTOR = [
  "button:not([disabled])",
  "input:not([disabled])",
  "textarea:not([disabled])",
  "select:not([disabled])",
  "a[href]",
  '[tabindex]:not([tabindex="-1"])',
].join(",")

/**
 * Create a keydown handler that traps Tab focus within a container.
 * Call with a ref to the container element.
 */
export function createFocusTrapHandler(
  containerRef: { current: HTMLElement | null },
  onEscape?: () => void,
): (e: KeyboardEvent) => void {
  return (e: KeyboardEvent) => {
    if (e.key === "Escape") {
      e.preventDefault()
      e.stopPropagation()
      onEscape?.()
      return
    }

    if (e.key !== "Tab") return

    const container = containerRef.current
    if (!container) return

    const focusable = Array.from(
      container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR),
    ).filter((el) => el.offsetParent !== null) // visible only

    if (focusable.length === 0) {
      e.preventDefault()
      container.focus()
      return
    }

    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    const active = document.activeElement as HTMLElement | null

    if (e.shiftKey) {
      // Shift+Tab: if on first element (or container), wrap to last
      if (active === first || active === container || !container.contains(active)) {
        e.preventDefault()
        last.focus()
      }
    } else {
      // Tab: if on last element (or container), wrap to first
      if (active === last || active === container || !container.contains(active)) {
        e.preventDefault()
        first.focus()
      }
    }
  }
}
