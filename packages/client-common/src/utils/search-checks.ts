/**
 * Quick checks (each repository's best quant) a search runs at once. A browser client reaches ACN over at most six
 * HTTP/1.1 connections, some held by the application's subscriptions, so requests past what's left wait in the
 * browser's own queue, out of this schedule's order.
 */
export const QUICK_SEARCH_CHECKS = 4

/** Full checks (every quant of a repository) a search runs at once, while no quick check waits. */
export const FULL_SEARCH_CHECKS = 2

/** Settled results revealed per batch before scrolling asks for more. */
export const SEARCH_BATCH_SIZE = 10

/**
 * A listed repository's quick check, as a search schedule reads it:
 * - `"Checking"`: initial quick check is still in flight; holds the settled frontier while within the grace window.
 * - `"Retrying"`: a previously shown unchecked row is being retried; keeps its place without pulling the frontier back.
 * - `"Runnable"`: settled and offers a quant on this machine; shown and eligible for background full inspection.
 * - `"Unchecked"`: settled without a verdict and actionable (e.g. gated repository or explicitly looked up); shown with retry.
 * - `"Hidden"`: settled as unsupported, not a model, or non-actionable failure; never shown and skipped silently.
 */
export type QuickSearchCheck = "Checking" | "Retrying" | "Runnable" | "Unchecked" | "Hidden"

/** What one pass of `readSearchChecks` established over `listed`. */
export interface SearchChecksProgress {
  /** Index of the first repository whose initial quick check has not yet settled (or the end of the checked prefix). */
  readonly frontier: number
  /** Number of visible rows (`Runnable`, `Unchecked`, or `Retrying`) in `listed.slice(0, frontier)`. */
  readonly shown: number
  /** Whether any initial quick check is currently running toward `target`. */
  readonly checking: boolean
  /** Whether every repository in `listed` settled without reaching `target`. */
  readonly exhausted: boolean
}

/**
 * Reads a search's checks strictly in list order (most downloaded first), two tiers deep. Reading a check is what runs
 * it, so only what should run is read.
 *
 * When a user expands a repository (`urgent`), its full quant inspection runs first and pauses starting new quick
 * checks or background prefetches until its quants finish, so an opened card never competes with background search
 * work for browser/service connections.
 *
 * Quick checks walk `listed` from index 0, keeping at most `QUICK_SEARCH_CHECKS` running at once, until `target`
 * visible rows (`Runnable` / `Unchecked` / `Retrying`) have settled or `listed` ends. Any repository already in
 * `revealed` is always kept read so an already-shown card can never disappear even if an earlier slow check finishes
 * later.
 *
 * Hovered/focused repositories (`priority`) start full quant inspection immediately when no opened card is loading.
 * Once no quick check is running, every settled runnable row in the current batch (10 initially, +10 per scroll) has
 * its full check prefetched in list order, up to `FULL_SEARCH_CHECKS` running at once. Full checks already in
 * `prefetched` are kept read so already-prefetched cards do not lose their details while later batches check.
 */
export const readSearchChecks = (input: {
  readonly listed: readonly string[]
  readonly target: number
  readonly revealed?: ReadonlySet<string>
  readonly prefetched?: ReadonlySet<string>
  /** Expanded repository whose full quant inspection takes strict priority over background checks. */
  readonly urgent?: string | null
  /** User-intent repositories (e.g. hovered/focused) whose full quant inspection starts ahead of background work. */
  readonly priority?: readonly string[]
  readonly quick: (key: string) => QuickSearchCheck
  readonly full: (key: string) => boolean
}): SearchChecksProgress => {
  const listedSet = new Set(input.listed)
  const fullState = new Map<string, boolean>()
  let fullRunning = 0
  const callFull = (key: string): boolean => {
    const existing = fullState.get(key)
    if (existing !== undefined) return existing
    const done = input.full(key)
    fullState.set(key, done)
    if (!done) fullRunning += 1
    return done
  }

  const urgentLoading = input.urgent != null && listedSet.has(input.urgent) && !callFull(input.urgent)

  const runnable: string[] = []
  let frontier = 0
  let frontierOpen = true
  let shownInFrontier = 0
  let shownTotal = 0
  let running = 0
  let initialRunning = 0
  let reachedEnd = true
  let pausedByUrgent = false
  let remainingRevealed = input.revealed ? input.listed.filter(key => input.revealed!.has(key)).length : 0

  for (let index = 0; index < input.listed.length; index += 1) {
    const key = input.listed[index]!
    const alreadyRevealed = input.revealed?.has(key) ?? false
    if (alreadyRevealed) remainingRevealed -= 1
    const capped = shownTotal >= input.target || running >= QUICK_SEARCH_CHECKS || urgentLoading
    if (!alreadyRevealed && capped) {
      reachedEnd = false
      if (urgentLoading && shownTotal < input.target) pausedByUrgent = true
      if (remainingRevealed === 0) break
      continue
    }
    const state = input.quick(key)
    if (state === "Checking") {
      frontierOpen = false
      running += 1
      initialRunning += 1
      continue
    }
    if (state === "Retrying") {
      running += 1
      shownTotal += 1
      if (frontierOpen) {
        shownInFrontier += 1
        frontier = index + 1
      }
      continue
    }
    if (state === "Runnable") {
      shownTotal += 1
      runnable.push(key)
      if (frontierOpen) {
        shownInFrontier += 1
        frontier = index + 1
      }
      continue
    }
    if (state === "Unchecked") {
      shownTotal += 1
      if (frontierOpen) {
        shownInFrontier += 1
        frontier = index + 1
      }
      continue
    }
    if (frontierOpen) {
      frontier = index + 1
    }
  }

  if (!urgentLoading && input.priority) {
    for (const key of input.priority) {
      if (listedSet.has(key)) callFull(key)
    }
  }

  if (input.prefetched) {
    for (const key of input.prefetched) {
      if (listedSet.has(key)) callFull(key)
    }
  }

  for (const key of runnable) {
    if (fullState.has(key)) continue
    if (urgentLoading || running > 0 || fullRunning >= FULL_SEARCH_CHECKS) break
    callFull(key)
  }

  return {
    frontier,
    shown: shownInFrontier,
    checking: initialRunning > 0 || pausedByUrgent,
    exhausted: reachedEnd && initialRunning === 0 && !pausedByUrgent && shownTotal < input.target,
  }
}
