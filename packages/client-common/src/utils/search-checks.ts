/**
 * Quick checks (each repository's best quant) a search runs at once. A browser client reaches ACN over at most six
 * HTTP/1.1 connections, some held by the application's subscriptions, so requests past what's left wait in the
 * browser's own queue, out of this schedule's order.
 */
export const QUICK_SEARCH_CHECKS = 4

/** Full checks (every quant of a repository) a search runs at once, while no quick check waits. */
export const FULL_SEARCH_CHECKS = 2

/** Repositories below the last one on screen whose quick checks run before they're scrolled to. */
export const SEARCH_LOOKAHEAD = 8

/** A listed repository's quick check, as a search schedule reads it. */
export type QuickSearchCheck = "Running" | "Runnable" | "Settled"

/**
 * The order a search checks its listed repositories in: those on screen, then the next `SEARCH_LOOKAHEAD` below them,
 * in list order. Before anything is on screen, the first `SEARCH_LOOKAHEAD`.
 */
export const searchCheckOrder = (listed: readonly string[], visible: ReadonlySet<string>): readonly string[] => {
  const shown = listed.flatMap((key, index) => visible.has(key) ? [index] : [])
  const last = shown.length > 0 ? shown[shown.length - 1]! : -1
  return [...shown.map(index => listed[index]!), ...listed.slice(last + 1, last + 1 + SEARCH_LOOKAHEAD)]
}

/**
 * Reads a search's checks in the order the schedule runs them, two tiers deep. Reading a check is what runs it, so only
 * what should run is read. Quick checks come first: in `searchCheckOrder`, every settled one, and the unsettled up to
 * `QUICK_SEARCH_CHECKS`. Once every repository in that order has its quick check running or settled, the ones that run
 * here have their full checks read, in the same order, up to `FULL_SEARCH_CHECKS` unsettled. Scrolling changes the
 * order, so the repositories newly on screen take the quick checks' turns again.
 */
export const readSearchChecks = (input: {
  readonly listed: readonly string[]
  readonly visible: ReadonlySet<string>
  readonly quick: (key: string) => QuickSearchCheck
  readonly full: (key: string) => boolean
}): void => {
  const order = searchCheckOrder(input.listed, input.visible)
  const runnable: string[] = []
  let running = 0
  for (const key of order) {
    if (running === QUICK_SEARCH_CHECKS) return
    const check = input.quick(key)
    if (check === "Running") running += 1
    if (check === "Runnable") runnable.push(key)
  }
  let fullRunning = 0
  for (const key of runnable) {
    if (fullRunning === FULL_SEARCH_CHECKS) return
    if (!input.full(key)) fullRunning += 1
  }
}
