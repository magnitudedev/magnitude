/**
 * Responsive store — tracks window width breakpoints.
 * Uses useSyncExternalStore with matchMedia — no useEffect.
 */

type Breakpoint = "narrow" | "normal" | "wide"

const NARROW_QUERY = "(max-width: 640px)"

let currentIsNarrow = false
const listeners = new Set<() => void>()

// Initialize from matchMedia
if (typeof window !== "undefined") {
  const mql = window.matchMedia(NARROW_QUERY)
  currentIsNarrow = mql.matches
  mql.addEventListener("change", (e) => {
    currentIsNarrow = e.matches
    listeners.forEach((l) => l())
  })
}

function subscribeResponsive(cb: () => void): () => void {
  listeners.add(cb)
  return () => listeners.delete(cb)
}

function getIsNarrow(): boolean {
  return currentIsNarrow
}

export { subscribeResponsive, getIsNarrow }
