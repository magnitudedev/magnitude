import { describe, expect, it } from "vitest"
import { FULL_SEARCH_CHECKS, QUICK_SEARCH_CHECKS, SEARCH_LOOKAHEAD, readSearchChecks, searchCheckOrder, type QuickSearchCheck } from "./search-checks"

const listed = Array.from({ length: 50 }, (_, index) => `repo-${index}`)
const range = (from: number, to: number) => listed.slice(from, to)

const reads = (input: {
  readonly visible: readonly string[]
  readonly quick?: Readonly<Record<string, QuickSearchCheck>>
  readonly full?: ReadonlySet<string>
  readonly read?: readonly string[]
}) => {
  const quick: string[] = []
  const full: string[] = []
  readSearchChecks({
    listed,
    visible: new Set(input.visible),
    read: new Set(input.read ?? []),
    quick: key => { quick.push(key); return input.quick?.[key] ?? "Running" },
    full: key => { full.push(key); return input.full?.has(key) ?? false },
  })
  return { quick, full }
}

const settled = (keys: readonly string[], check: QuickSearchCheck) => Object.fromEntries(keys.map(key => [key, check]))

describe("searchCheckOrder", () => {
  it("puts the repositories on screen first, then the next ones below them", () => {
    expect(searchCheckOrder(listed, new Set(range(20, 27)))).toEqual(range(20, 27 + SEARCH_LOOKAHEAD))
  })

  it("starts from the top before anything is on screen", () => {
    expect(searchCheckOrder(listed, new Set())).toEqual(range(0, SEARCH_LOOKAHEAD))
  })
})

describe("readSearchChecks", () => {
  it("runs the quick checks on screen first, a few at a time, and no full check while any waits", () => {
    const { quick, full } = reads({ visible: range(0, 7) })
    expect(quick).toEqual(range(0, QUICK_SEARCH_CHECKS))
    expect(full).toEqual([])
  })

  it("keeps reading settled quick checks past the running ones", () => {
    const { quick } = reads({ visible: range(0, 7), quick: settled(range(0, 5), "Runnable") })
    expect(quick).toEqual(range(0, 5 + QUICK_SEARCH_CHECKS))
  })

  it("fully checks the repositories that run, in order, once every quick check has started", () => {
    const quick = { ...settled(range(0, 15), "Runnable"), "repo-1": "Settled" as const }
    const { full } = reads({ visible: range(0, 7), quick })
    expect(full).toEqual(["repo-0", ...range(2, 1 + FULL_SEARCH_CHECKS)])
  })

  it("keeps reading settled full checks past the running ones", () => {
    const { full } = reads({ visible: range(0, 7), quick: settled(range(0, 15), "Runnable"), full: new Set(range(0, 3)) })
    expect(full).toEqual(range(0, 3 + FULL_SEARCH_CHECKS))
  })

  it("gives the repositories scrolled to the quick checks' turns before any full check", () => {
    const { quick, full } = reads({ visible: range(20, 27), quick: settled(range(0, 15), "Runnable") })
    expect(quick).toEqual(range(20, 20 + QUICK_SEARCH_CHECKS))
    expect(full).toEqual([])
  })

  it("keeps reading a check read before once its row leaves the screen, so what it found stays known", () => {
    // repo-2 settled as not running here: its row is hidden, so it leaves `visible` and falls out of the order.
    const { quick } = reads({ visible: [...range(0, 2), ...range(3, 7)], quick: settled(range(0, 15), "Settled"), read: ["repo-2"] })
    expect(quick).toContain("repo-2")
  })

  it("starts no check only because it is in no order, unless it was read before", () => {
    const { quick } = reads({ visible: range(0, 7), read: ["repo-40"] })
    expect(quick).toEqual(["repo-40", ...range(0, QUICK_SEARCH_CHECKS - 1)])
  })

  it("counts the running checks of rows that left the screen toward the quick checks run at once", () => {
    const hidden = range(40, 40 + QUICK_SEARCH_CHECKS)
    const { quick } = reads({ visible: range(0, 7), read: hidden })
    expect(quick.filter(key => !hidden.includes(key))).toEqual([])
  })
})
