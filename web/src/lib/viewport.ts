import { useSyncExternalStore } from "react"

/** Below Tailwind's `md` breakpoint (48rem): the shell uses a drawer instead of a docked sidebar. */
const NARROW_QUERY = "(max-width: 47.99rem)"

const subscribe = (onChange: () => void) => {
  const query = window.matchMedia(NARROW_QUERY)
  query.addEventListener("change", onChange)
  return () => query.removeEventListener("change", onChange)
}

export const useNarrowViewport = (): boolean =>
  useSyncExternalStore(subscribe, () => window.matchMedia(NARROW_QUERY).matches, () => false)
