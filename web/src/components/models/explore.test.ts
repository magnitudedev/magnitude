import { describe, expect, it } from "vitest"
import { Atom, Registry, Result } from "@effect-atom/atom-react"
import { Option } from "effect"
import type { AgentClient } from "@magnitudedev/client-common"
import type { HuggingFaceRepositoryCheck, HuggingFaceSearchResult } from "@magnitudedev/sdk"
import { exploreFirst, listedFirst, searchSchedule } from "./explore"
import type { OfferedModel } from "./expanding-model"

type CheckState = { readonly result: Result.Result<HuggingFaceRepositoryCheck, unknown>; readonly isStale: boolean }

const unsupported = () => ({ _tag: "Model", unsupported: Option.some("qwen3"), recommended: Option.none() }) as unknown as HuggingFaceRepositoryCheck
const repositories = ["owner/a", "owner/b", "owner/c"]
const listed = repositories.map(repository => ({ repository, commit: "c", architecture: Option.none(), gated: false, private: false }) as unknown as HuggingFaceSearchResult)

const setup = () => {
  const checks = new Map(repositories.map(repository => [repository, Atom.make<CheckState>({ result: Result.initial(true), isStale: false })]))
  const client = {
    Models: {
      CheckHuggingFaceRepository: ({ repository }: { readonly repository: string }) => checks.get(repository)!,
      InspectModelCandidate: () => Atom.make({ result: Result.initial() }),
    },
  } as unknown as AgentClient
  const visible = Atom.make<ReadonlySet<string>>(new Set(repositories))
  const registry = Registry.make()
  const schedule = searchSchedule(client, listed, visible)
  const passes: unknown[] = []
  registry.subscribe(schedule, value => passes.push(value), { immediate: true })
  return { registry, checks, visible, schedule, passes }
}

describe("Explore's search schedule", () => {
  it("settles: a check's bookkeeping changing without its result changing is no new schedule, so nothing redraws", () => {
    const { registry, checks, passes } = setup()
    // A query keeps its result while its staleness, fetch status or update time changes around it.
    const results = new Map(repositories.map(repository => [repository, Result.success(unsupported())]))
    for (const repository of repositories) registry.set(checks.get(repository)!, { result: results.get(repository)!, isStale: false })
    const settled = passes.length
    for (let round = 0; round < 5; round++) {
      for (const repository of repositories) registry.set(checks.get(repository)!, { result: results.get(repository)!, isStale: round % 2 === 0 })
    }
    expect(passes.length).toBe(settled)
  })

  it("keeps a finished check's result once its hidden row leaves the screen, so it never shows as checking again", () => {
    const { registry, checks, visible, schedule } = setup()
    for (const repository of repositories) registry.set(checks.get(repository)!, { result: Result.success(unsupported()), isStale: false })
    registry.set(visible, new Set())
    const after = registry.get(schedule)
    expect(repositories.every(repository => Result.isSuccess(after.checks.get(repository)!))).toBe(true)
    const again = registry.get(schedule)
    expect(again).toBe(after)
  })

  it("is a new schedule when a check finds something new", () => {
    const { registry, checks, schedule } = setup()
    const before = registry.get(schedule)
    registry.set(checks.get("owner/a")!, { result: Result.success(unsupported()), isStale: false })
    expect(registry.get(schedule)).not.toBe(before)
  })
})

describe("a search for a model's other quants", () => {
  it("lists the repository the model came from first, and the rest in the order found", () => {
    expect(listedFirst(listed, listed[2]).map(result => result.repository)).toEqual(["owner/c", "owner/a", "owner/b"])
    expect(listedFirst(listed, undefined)).toBe(listed)
  })

  it("leads with the model's own repository even when the search didn't find it, as for LiquidAI/LFM3-1.7B-GGUF", () => {
    const own = { ...listed[0]!, repository: "LiquidAI/LFM3-1.7B-GGUF" } as HuggingFaceSearchResult
    expect(listedFirst(listed, own).map(result => result.repository)).toEqual(["LiquidAI/LFM3-1.7B-GGUF", "owner/a", "owner/b", "owner/c"])
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
