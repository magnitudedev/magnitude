import { describe, expect, it } from "vitest"
import { Atom, Registry, Result } from "@effect-atom/atom-react"
import { Option } from "effect"
import type { AgentClient } from "@magnitudedev/client-common"
import type { HuggingFaceRepositoryCheck, HuggingFaceSearchResult } from "@magnitudedev/sdk"
import { dedupeResults, exploreFirst, inspectReference, listedFirst, matchingRecommendations, quickState, searchSchedule } from "./explore"
import type { OfferedModel } from "./expanding-model"
import type { OfferedRecommendation } from "./recommendations"

type CheckState = { readonly result: Result.Result<HuggingFaceRepositoryCheck, unknown>; readonly isStale: boolean }

const unsupported = () => ({ _tag: "Model", unsupported: Option.some("qwen3"), recommended: Option.none() }) as unknown as HuggingFaceRepositoryCheck
const runnable = (id: string) => ({ _tag: "Model", unsupported: Option.none(), recommended: Option.some({ modelId: id }) }) as unknown as HuggingFaceRepositoryCheck
const incomplete = () => ({ _tag: "Model", unsupported: Option.none(), recommended: Option.none() }) as unknown as HuggingFaceRepositoryCheck

const repositories = ["owner/a", "owner/b", "owner/c", "owner/d"]
const makeResults = (names: readonly string[], gated = false) =>
  names.map(repository => ({ repository, commit: "c", architecture: Option.none(), gated, private: false }) as unknown as HuggingFaceSearchResult)
const listed = makeResults(repositories)

const setup = (initialTarget = 10, initialListed: readonly HuggingFaceSearchResult[] = listed) => {
  const checks = new Map<string, Atom.Writable<CheckState>>(initialListed.map(item => [item.repository, Atom.make<CheckState>({ result: Result.initial(true), isStale: false })]))
  const inspected = new Map<string, Atom.Writable<{ readonly result: Result.Result<unknown, unknown> }>>(initialListed.map(item => [inspectReference(item.repository, item.commit), Atom.make({ result: Result.initial() as Result.Result<unknown, unknown> })]))
  const inspectedRefs: string[] = []
  const client = {
    Models: {
      CheckHuggingFaceRepository: ({ repository }: { readonly repository: string }) => {
        if (!checks.has(repository)) checks.set(repository, Atom.make<CheckState>({ result: Result.initial(true), isStale: false }))
        return checks.get(repository)!
      },
      InspectModelCandidate: ({ reference }: { readonly reference: string }) => {
        inspectedRefs.push(reference)
        if (!inspected.has(reference)) inspected.set(reference, Atom.make({ result: Result.initial() as Result.Result<unknown, unknown> }))
        return inspected.get(reference)!
      },
    },
  } as unknown as AgentClient
  const listedAtom = Atom.make<readonly HuggingFaceSearchResult[]>(initialListed)
  const targetAtom = Atom.make(initialTarget)
  const releasedFrontier = Atom.make<string | null>(null)
  const openedAtom = Atom.make<string | null>(null)
  const hoveredAtom = Atom.make<string | null>(null)
  const registry = Registry.make()
  const schedule = searchSchedule(client, listedAtom, targetAtom, { releasedFrontier, opened: openedAtom, hovered: hoveredAtom })
  const passes: unknown[] = []
  registry.subscribe(schedule, value => passes.push(value), { immediate: true })
  return { registry, checks, inspected, inspectedRefs, listedAtom, targetAtom, releasedFrontier, openedAtom, hoveredAtom, schedule, passes }
}

describe("Explore's search schedule", () => {
  it("settles: a check's bookkeeping changing without its result changing is no new schedule, so nothing redraws", () => {
    const { registry, checks, passes } = setup()
    const results = new Map(repositories.map(repository => [repository, Result.success(unsupported())]))
    for (const repository of repositories) registry.set(checks.get(repository)!, { result: results.get(repository)!, isStale: false })
    const settled = passes.length
    for (let round = 0; round < 5; round++) {
      for (const repository of repositories) registry.set(checks.get(repository)!, { result: results.get(repository)!, isStale: round % 2 === 0 })
    }
    expect(passes.length).toBe(settled)
  })

  it("reveals rows strictly in download order within the grace window", () => {
    const { registry, checks, schedule } = setup()
    // owner/c finishes first, owner/b is unsupported, while owner/a is still checking: held briefly behind owner/a.
    registry.set(checks.get("owner/c")!, { result: Result.success(runnable("c")), isStale: false })
    registry.set(checks.get("owner/b")!, { result: Result.success(unsupported()), isStale: false })
    expect(registry.get(schedule).revealed).toEqual([])

    // Once owner/a settles, [owner/a, owner/c] appear together in exact download order (skipping unsupported owner/b).
    registry.set(checks.get("owner/a")!, { result: Result.success(runnable("a")), isStale: false })
    expect(registry.get(schedule).revealed).toEqual(["owner/a", "owner/c"])
  })

  it("releases ready results after the grace window without ever jumping or reordering when the slow head finishes", () => {
    const { registry, checks, releasedFrontier, schedule } = setup(2)
    // owner/a is slow; owner/b and owner/c are ready.
    registry.set(checks.get("owner/b")!, { result: Result.success(runnable("b")), isStale: false })
    registry.set(checks.get("owner/c")!, { result: Result.success(runnable("c")), isStale: false })
    expect(registry.get(schedule).revealed).toEqual([])

    // Grace window expires for owner/a -> owner/b and owner/c appear immediately in download order.
    registry.set(releasedFrontier, "owner/a")
    expect(registry.get(schedule).revealed).toEqual(["owner/b", "owner/c"])

    // Later owner/a finishes -> appends BELOW already-revealed rows instead of jumping above them, and owner/c is never dropped.
    registry.set(checks.get("owner/a")!, { result: Result.success(runnable("a")), isStale: false })
    expect(registry.get(schedule).revealed).toEqual(["owner/b", "owner/c", "owner/a"])
    expect(registry.get(schedule).checks.has("owner/c")).toBe(true)
  })

  it("hides non-gated failed or timed-out repos from multi-result searches instead of showing broken cards", () => {
    const { registry, checks, schedule } = setup()
    registry.set(checks.get("owner/a")!, { result: Result.success(incomplete()), isStale: false })
    registry.set(checks.get("owner/b")!, { result: Result.success(runnable("b")), isStale: false })
    registry.set(checks.get("owner/c")!, { result: Result.success(unsupported()), isStale: false })
    registry.set(checks.get("owner/d")!, { result: Result.success(runnable("d")), isStale: false })
    expect(registry.get(schedule).revealed).toEqual(["owner/b", "owner/d"])
  })

  it("never resurrects hidden unsupported rows or resets prefetched cards when a second page appends", () => {
    const { registry, checks, inspected, listedAtom, schedule } = setup(10)
    registry.set(checks.get("owner/a")!, { result: Result.success(runnable("a")), isStale: false })
    registry.set(checks.get("owner/b")!, { result: Result.success(unsupported()), isStale: false })
    registry.set(checks.get("owner/c")!, { result: Result.success(runnable("c")), isStale: false })
    registry.set(checks.get("owner/d")!, { result: Result.success(unsupported()), isStale: false })
    registry.set(inspected.get("owner/a@c")!, { result: Result.success({}) })
    const beforePage2 = registry.get(schedule)
    expect(beforePage2.revealed).toEqual(["owner/a", "owner/c"])
    expect(beforePage2.full.has("owner/a")).toBe(true)

    // Append page 2: page 1's hidden rows stay hidden, revealed stays stable, and owner/a stays prefetched.
    registry.set(listedAtom, [...listed, ...makeResults(["owner/e", "owner/f"])])
    const afterPage2 = registry.get(schedule)
    expect(afterPage2.revealed).toEqual(["owner/a", "owner/c"])
    expect(quickState(afterPage2.checks.get("owner/b")!, false)).toBe("Hidden")
    expect(quickState(afterPage2.checks.get("owner/d")!, false)).toBe("Hidden")
    expect(afterPage2.full.has("owner/a")).toBe(true)
  })

  it("keeps gated unchecked rows retryable in place without hiding rows below them", () => {
    const gatedListed = [makeResults(["owner/a"], true)[0]!, ...makeResults(["owner/b", "owner/c", "owner/d"])]
    const { registry, checks, schedule } = setup(10, gatedListed)
    registry.set(checks.get("owner/a")!, { result: Result.success(incomplete()), isStale: false })
    registry.set(checks.get("owner/b")!, { result: Result.success(runnable("b")), isStale: false })
    registry.set(checks.get("owner/c")!, { result: Result.success(runnable("c")), isStale: false })
    registry.set(checks.get("owner/d")!, { result: Result.success(unsupported()), isStale: false })
    expect(registry.get(schedule).revealed).toEqual(["owner/a", "owner/b", "owner/c"])

    // Retry gated owner/a -> stays in place and rows below remain visible.
    registry.set(checks.get("owner/a")!, { result: Result.success(incomplete(), { waiting: true }), isStale: false })
    expect(quickState(registry.get(schedule).checks.get("owner/a")!, true)).toBe("Retrying")
    expect(registry.get(schedule).revealed).toEqual(["owner/a", "owner/b", "owner/c"])
  })

  it("pins full inspection to repository@commit and prioritizes opened and hovered repos", () => {
    const { registry, checks, inspectedRefs, openedAtom, hoveredAtom, schedule } = setup(10)
    registry.set(checks.get("owner/a")!, { result: Result.success(runnable("a")), isStale: false })
    // Hovering owner/a starts full inspection immediately pinned to @commit even while owner/b..d are still checking.
    registry.set(hoveredAtom, "owner/a")
    expect(registry.get(schedule).full.has("owner/a")).toBe(true)
    expect(inspectedRefs).toContain("owner/a@c")

    // Opening owner/a takes priority and keeps owner/a in full.
    registry.set(openedAtom, "owner/a")
    expect(registry.get(schedule).full.has("owner/a")).toBe(true)
  })

  it("is a new schedule when a check finds something new", () => {
    const { registry, checks, schedule } = setup()
    const before = registry.get(schedule)
    registry.set(checks.get("owner/a")!, { result: Result.success(unsupported()), isStale: false })
    expect(registry.get(schedule)).not.toBe(before)
  })
})

describe("dedupeResults", () => {
  it("drops duplicate repositories across live pages while keeping their highest-ranked position", () => {
    const combined = [...listed, listed[1]!, ...makeResults(["owner/e"]), listed[0]!]
    expect(dedupeResults(combined).map(result => result.repository)).toEqual(["owner/a", "owner/b", "owner/c", "owner/d", "owner/e"])
  })
})

describe("a search for a model's other quants", () => {
  it("lists the repository the model came from first, and the rest in the order found", () => {
    expect(listedFirst(listed, listed[2]).map(result => result.repository)).toEqual(["owner/c", "owner/a", "owner/b", "owner/d"])
    expect(listedFirst(listed, undefined)).toBe(listed)
  })

  it("leads with the model's own repository even when the search didn't find it, as for LiquidAI/LFM3-1.7B-GGUF", () => {
    const own = { ...listed[0]!, repository: "LiquidAI/LFM3-1.7B-GGUF" } as HuggingFaceSearchResult
    expect(listedFirst(listed, own).map(result => result.repository)).toEqual(["LiquidAI/LFM3-1.7B-GGUF", "owner/a", "owner/b", "owner/c", "owner/d"])
  })
})

describe("what leads a search for a model's other quants", () => {
  const lfm3 = { key: "lfm3-1.7b", catalog: true, quants: [{ repository: "LiquidAI/LFM3-1.7B-GGUF" }] } as unknown as OfferedModel

  it("is the recommended model published in its repository, which Hugging Face search leaves out, as it is recommended here", () => {
    expect(exploreFirst("LiquidAI/LFM3-1.7B-GGUF", [{ model: lfm3 }])).toEqual({ _tag: "Offered", model: lfm3 })
  })

  it("is any other repository looked up by name", () => {
    expect(exploreFirst("bartowski/LFM3-1.7B-GGUF", [{ model: lfm3 }])).toEqual({ _tag: "Repository", repository: "bartowski/LFM3-1.7B-GGUF" })
  })
})

describe("matchingRecommendations", () => {
  const makeOffered = (
    base: string,
    name: string,
    developerName: string,
    repository: string,
    downloadable = true,
  ): OfferedRecommendation => ({
    recommendation: {
      base,
      name,
      developer: { lab: developerName.toLowerCase(), name: developerName, country: "US" },
      chosen: downloadable ? Option.some(`${base}:gguf:q4`) : Option.none(),
      rungs: [`${base}:gguf:q4`],
      meetsTarget: downloadable,
    } as unknown as OfferedRecommendation["recommendation"],
    model: {
      key: base,
      catalog: true,
      name,
      quants: [{ key: `${base}:gguf:q4`, bits: 4, name: "UD-Q4_K_XL", repository }],
    } as unknown as OfferedModel,
  })

  const offered = [
    makeOffered("qwen3.8-27b", "Qwen3.8 27B", "Alibaba", "unsloth/Qwen3.8-27B-GGUF"),
    makeOffered("gemma-4-12b-it-qat", "Gemma 4 12B", "Google", "unsloth/gemma-4-12B-it-qat-GGUF"),
    makeOffered("lfm3-1.7b", "Liquid LFM3 1.7B", "Liquid AI", "LiquidAI/LFM3-1.7B-GGUF"),
    makeOffered("qwen3.8-flash-next", "Qwen3.8 Flash Next", "Alibaba", "unsloth/Qwen3.8-Flash-Next-GGUF", false),
    makeOffered("qwen3.6-35b-a3b", "Qwen3.6 35B-A3B", "Alibaba", "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"),
    makeOffered("qwen3.5-4b", "Qwen3.5 4B", "Alibaba", "unsloth/Qwen3.5-4B-MTP-GGUF"),
    makeOffered("qwen3.5-9b", "Qwen3.5 9B", "Alibaba", "unsloth/Qwen3.5-9B-MTP-GGUF"),
  ]

  it("matches across letter-digit boundaries and spaces while ignoring non-downloadable models", () => {
    expect(matchingRecommendations("gemma4", offered, new Set()).map(entry => entry.model.key)).toEqual(["gemma-4-12b-it-qat"])
    expect(matchingRecommendations("qwen 3.8", offered, new Set()).map(entry => entry.model.key)).toEqual(["qwen3.8-27b"])
    expect(matchingRecommendations("google", offered, new Set()).map(entry => entry.model.key)).toEqual(["gemma-4-12b-it-qat"])
  })

  it("does not falsely match different version numbers or quantizer names like unsloth", () => {
    expect(matchingRecommendations("gemma 3", offered, new Set())).toEqual([])
    expect(matchingRecommendations("qwen 2", offered, new Set())).toEqual([])
    expect(matchingRecommendations("unsloth", offered, new Set())).toEqual([])
    expect(matchingRecommendations("llama", offered, new Set())).toEqual([])
  })

  it("matches pasted Hugging Face repository references and URLs", () => {
    expect(matchingRecommendations("https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF", offered, new Set()).map(entry => entry.model.key)).toEqual(["lfm3-1.7b"])
    expect(matchingRecommendations("unsloth/gemma-4-12b-it-qat-gguf", offered, new Set()).map(entry => entry.model.key)).toEqual(["gemma-4-12b-it-qat"])
  })

  it("caps pinned results at three, notes already-owned quants, and deduplicates an explicit first offered model", () => {
    const owned = new Set(["qwen3.8-27b:gguf:q4"])
    const picks = matchingRecommendations("qwen", offered, owned, { _tag: "Offered", model: offered[4]!.model })
    expect(picks.map(entry => entry.model.key)).toEqual(["qwen3.6-35b-a3b", "qwen3.8-27b", "qwen3.5-4b"])
    expect(picks[1]!.note).toBe("You have the 4-bit quant")
  })
})
