import { describe, expect, it } from "vitest"
import { FULL_SEARCH_CHECKS, QUICK_SEARCH_CHECKS, SEARCH_BATCH_SIZE, readSearchChecks, type QuickSearchCheck } from "./search-checks"

const listed = Array.from({ length: 30 }, (_, index) => `repo-${index}`)
const range = (from: number, to: number) => listed.slice(from, to)

const run = (input: {
  readonly items?: readonly string[]
  readonly target?: number
  readonly quick?: Readonly<Record<string, QuickSearchCheck>>
  readonly full?: ReadonlySet<string>
  readonly revealed?: ReadonlySet<string>
  readonly prefetched?: ReadonlySet<string>
  readonly urgent?: string | null
  readonly priority?: readonly string[]
}) => {
  const quick: string[] = []
  const full: string[] = []
  const progress = readSearchChecks({
    listed: input.items ?? listed,
    target: input.target ?? SEARCH_BATCH_SIZE,
    revealed: input.revealed,
    prefetched: input.prefetched,
    urgent: input.urgent,
    priority: input.priority,
    quick: key => { quick.push(key); return input.quick?.[key] ?? "Checking" },
    full: key => { full.push(key); return input.full?.has(key) ?? false },
  })
  return { quick, full, progress }
}

const stateMap = (keys: readonly string[], state: QuickSearchCheck) => Object.fromEntries(keys.map(key => [key, state]))

describe("readSearchChecks", () => {
  it("evaluates strictly from the top of the list up to QUICK_SEARCH_CHECKS at once and runs no full checks while quick checks run", () => {
    const { quick, full, progress } = run({})
    expect(quick).toEqual(range(0, QUICK_SEARCH_CHECKS))
    expect(full).toEqual([])
    expect(progress).toEqual({ frontier: 0, shown: 0, checking: true, exhausted: false })
  })

  it("holds later settled results behind an earlier checking repo so nothing jumps ahead out of order", () => {
    // repo-0 is still checking while repo-1 is hidden and repo-2 is runnable.
    const { quick, full, progress } = run({
      quick: { "repo-1": "Hidden", "repo-2": "Runnable" },
    })
    expect(quick).toEqual(range(0, 2 + QUICK_SEARCH_CHECKS))
    expect(full).toEqual([])
    expect(progress).toEqual({ frontier: 0, shown: 0, checking: true, exhausted: false })
  })

  it("advances the settled frontier past hidden and runnable repos once the head settles", () => {
    // repo-0 runnable, repo-1 hidden, repo-2 runnable, repo-3 still checking.
    const { quick, progress } = run({
      quick: { "repo-0": "Runnable", "repo-1": "Hidden", "repo-2": "Runnable" },
    })
    expect(quick).toEqual(range(0, 3 + QUICK_SEARCH_CHECKS))
    expect(progress).toEqual({ frontier: 3, shown: 2, checking: true, exhausted: false })
  })

  it("stops checking once target visible rows have settled and prefetches full inspections in list order", () => {
    const quick = {
      ...stateMap(range(0, 20), "Runnable"),
      "repo-1": "Hidden" as const,
      "repo-4": "Unchecked" as const,
    }
    const { quick: readQuick, full, progress } = run({ target: 5, quick })
    // 5 visible rows: repo-0 (Runnable), repo-2 (Runnable), repo-3 (Runnable), repo-4 (Unchecked), repo-5 (Runnable); repo-1 was Hidden.
    expect(readQuick).toEqual(range(0, 6))
    expect(progress).toEqual({ frontier: 6, shown: 5, checking: false, exhausted: false })
    // Full checks only prefetch Runnable rows (skipping Hidden repo-1 and Unchecked repo-4), up to FULL_SEARCH_CHECKS at once.
    expect(full).toEqual(["repo-0", "repo-2"])
  })

  it("continues prefetching every settled runnable row in the batch once quick checks finish", () => {
    const quick = stateMap(range(0, 10), "Runnable")
    const { full } = run({ target: 10, quick, full: new Set(range(0, 5)) })
    expect(full).toEqual(range(0, 5 + FULL_SEARCH_CHECKS))
  })

  it("keeps already-prefetched full checks read while a later batch runs quick checks", () => {
    const quick = stateMap(range(0, 5), "Runnable")
    const prefetched = new Set(range(0, 3))
    const { full, progress } = run({ target: 10, quick, full: prefetched, prefetched })
    expect(progress).toEqual({ frontier: 5, shown: 5, checking: true, exhausted: false })
    expect(full).toEqual(range(0, 3))
  })

  it("keeps rows below a retrying unchecked row in place without pulling the frontier back", () => {
    const quick: Record<string, QuickSearchCheck> = {
      ...stateMap(range(0, 5), "Runnable"),
      "repo-1": "Retrying",
    }
    const { quick: readQuick, full, progress } = run({ target: 5, quick })
    expect(readQuick).toEqual(range(0, 5))
    expect(progress).toEqual({ frontier: 5, shown: 5, checking: false, exhausted: false })
    expect(full).toEqual([])
  })

  it("marks exhausted only when every listed repository settled short of target", () => {
    const items = range(0, 6)
    const quick = { ...stateMap(items, "Hidden"), "repo-2": "Runnable" as const }
    const { progress } = run({ items, target: 5, quick })
    expect(progress).toEqual({ frontier: 6, shown: 1, checking: false, exhausted: true })
  })

  it("stops starting more quick checks once target visible rows are already settled behind a slow head", () => {
    const quick = stateMap(range(1, 6), "Runnable")
    const { quick: readQuick, progress } = run({ target: 5, quick })
    expect(readQuick).toEqual(range(0, 6))
    expect(progress).toEqual({ frontier: 0, shown: 0, checking: true, exhausted: false })
  })

  it("pauses new quick checks and background prefetches while an opened (urgent) repo loads its quants", () => {
    const quick = stateMap(range(0, 2), "Runnable")
    const revealed = new Set(range(0, 2))
    // User clicked repo-1 while target=10 still has unchecked repos behind it.
    const loading = run({ target: 10, quick, revealed, urgent: "repo-1" })
    expect(loading.full).toEqual(["repo-1"])
    // Revealed rows stay read, but NO new quick checks start for repo-2.. while repo-1 is loading.
    expect(loading.quick).toEqual(["repo-0", "repo-1"])
    expect(loading.progress).toEqual({ frontier: 2, shown: 2, checking: true, exhausted: false })

    // Once repo-1's full quant inspection finishes, quick checks immediately resume.
    const done = run({ target: 10, quick, revealed, urgent: "repo-1", full: new Set(["repo-1"]) })
    expect(done.quick).toEqual(range(0, 2 + QUICK_SEARCH_CHECKS))
  })

  it("warms a hovered (priority) repo's quants immediately even while quick checks are running", () => {
    const quick = stateMap(range(0, 2), "Runnable")
    const { quick: readQuick, full } = run({ target: 10, quick, priority: ["repo-1"] })
    expect(readQuick).toEqual(range(0, 2 + QUICK_SEARCH_CHECKS))
    expect(full).toEqual(["repo-1"])
  })

  it("keeps already-revealed rows read even after an earlier slow repo finishes and reaches target", () => {
    // repo-1..repo-5 were already revealed while repo-0 was slow; now repo-0 finishes too (target = 5).
    const quick = stateMap(range(0, 6), "Runnable")
    const { quick: readQuick } = run({ target: 5, quick, revealed: new Set(range(1, 6)) })
    expect(readQuick).toEqual(range(0, 6))
  })
})
