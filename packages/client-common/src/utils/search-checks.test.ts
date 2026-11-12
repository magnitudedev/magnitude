import { describe, expect, it } from "vitest"
import { FULL_SEARCH_CHECKS, QUICK_SEARCH_CHECKS, SEARCH_LOOKAHEAD, readSearchChecks, searchCheckOrder, type QuickSearchCheck } from "./search-checks"

const listed = Array.from({ length: 50 }, (_, index) => `repo-${index}`)
const range = (from: number, to: number) => listed.slice(from, to)

const reads = (input: {
  readonly visible: readonly string[]
  readonly quick?: Readonly<Record<string, QuickSearchCheck>>
  readonly full?: ReadonlySet<string>
}) => {
  const quick: string[] = []
  const full: string[] = []
  readSearchChecks({
    listed,
    visible: new Set(input.visible),
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
})
