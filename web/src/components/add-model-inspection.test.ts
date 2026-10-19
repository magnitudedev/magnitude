import { describe, expect, it } from "vitest"
import { Registry, Result } from "@effect-atom/atom-react"
import { Cause, Data, Effect, Layer, Option, Queue, Stream } from "effect"
import { Client, Query } from "@magnitudedev/effect-query"
import { InvalidModelReference, type CandidateInspectionProgress, type CandidateInspectionUpdate } from "@magnitudedev/sdk"
import { inspectionView, submissionState } from "./add-model-dialog"
import { fits, inspection, unsupported } from "../../test/models/fixtures"

const failure = new InvalidModelReference({ reference: "owner/repo", message: "PRIVATE" })
const inspected: CandidateInspectionUpdate = { _tag: "Inspected", inspection: inspection() }
const progress = (phase: CandidateInspectionProgress["phase"], verdicts: readonly ("fits" | "unsupported" | undefined)[]): CandidateInspectionUpdate => ({
  _tag: "Inspecting",
  progress: {
    phase,
    versions: ["Q4_K_M", "Q2_K"].map((label, index) => ({
      label,
      sizeBytes: 1_000_000_000,
      assessment: verdicts[index] === "fits" ? Option.some(fits)
        : verdicts[index] === "unsupported" ? Option.some(unsupported({ _tag: "WeightTypes", types: [{ ggmlType: "Q2_K", tensorCount: 2 }] }))
        : Option.none(),
    })),
  },
})

describe("inspection after an error", () => {
  it("shows the error, then reading while Retry runs, then the inspection", () => {
    expect(inspectionView(Result.failure(Cause.fail(failure)))).toEqual({ _tag: "Failed", error: Option.some(failure) })
    expect(inspectionView(Result.failure(Cause.fail(failure), { waiting: true }))).toEqual({ _tag: "Reading" })
    expect(inspectionView(Result.initial(true))).toEqual({ _tag: "Reading" })
    expect(inspectionView(Result.success(inspected))).toEqual({ _tag: "Inspected", inspection: inspection() })
  })
  it("re-runs a failed inspection on Retry and shows the result once it succeeds", async () => {
    const registry = Registry.make()
    const effectQuery = Client.make(Layer.empty)
    let attempts = 0
    const inspect = Query.make("InspectForRetry", {
      key: ({ reference }: { readonly reference: string }) => Data.struct({ reference }),
      effect: () => Effect.suspend(() => ++attempts === 1 ? Effect.fail(failure) : Effect.succeed<CandidateInspectionUpdate>(inspected)),
      staleTime: "5 minutes",
    })
    const atom = effectQuery.query(inspect, { reference: "owner/repo" })
    const unmount = registry.mount(atom)
    const view = () => inspectionView(registry.get(atom).result)
    const settle = () => Effect.runPromise(Effect.sleep("5 millis"))
    await settle()
    expect(view()).toEqual({ _tag: "Failed", error: Option.some(failure) })
    registry.refresh(atom)
    await settle()
    expect(attempts).toBe(2)
    expect(view()).toEqual({ _tag: "Inspected", inspection: inspection() })
    unmount()
    registry.dispose()
  })
})

describe("inspection as it goes", () => {
  const settle = () => Effect.runPromise(Effect.sleep("5 millis"))
  it("shows each step's progress, then the inspection, and an inspection already made at once", async () => {
    const registry = Registry.make()
    const effectQuery = Client.make(Layer.empty)
    const updates = Effect.runSync(Queue.unbounded<CandidateInspectionUpdate>())
    let opened = 0
    const inspect = Query.fromStream("StreamInspection", {
      key: ({ reference }: { readonly reference: string }) => Data.struct({ reference }),
      stream: () => Stream.suspend(() => {
        opened += 1
        return Stream.fromQueue(updates)
      }),
      reduce: (_: Option.Option<CandidateInspectionUpdate>, update: CandidateInspectionUpdate) => update,
    })
    const atom = effectQuery.query(inspect, { reference: "owner/repo" })
    const view = () => inspectionView(registry.get(atom).result)
    const unmount = registry.mount(atom)
    await settle()
    expect(view()).toEqual({ _tag: "Reading" })

    const steps = [
      progress({ _tag: "ReadingHeaders", completed: 1, total: 2 }, []),
      progress({ _tag: "CheckingVersions", completed: 1, total: 2 }, [undefined, "unsupported"]),
      progress({ _tag: "CheckingVersions", completed: 2, total: 2 }, ["fits", "unsupported"]),
    ]
    for (const step of steps) {
      Effect.runSync(Queue.offer(updates, step))
      await settle()
      expect(view()).toEqual({ _tag: "Inspecting", progress: step._tag === "Inspecting" && step.progress })
    }
    Effect.runSync(Queue.offer(updates, inspected))
    await settle()
    expect(view()).toEqual({ _tag: "Inspected", inspection: inspection() })
    unmount()

    const reopened = registry.mount(atom)
    expect(view()).toEqual({ _tag: "Inspected", inspection: inspection() })
    await settle()
    expect(opened).toBe(1)
    reopened()
    registry.dispose()
  })
})

describe("adding after an earlier failure", () => {
  const failed = Result.failure<unknown, unknown>(Cause.fail(failure))
  it("ignores another opening's pending or failed add until this one submits", () => {
    expect(submissionState(false, failed)).toEqual({ pending: false, failure: undefined })
    expect(submissionState(false, Result.initial(true))).toEqual({ pending: false, failure: undefined })
  })
  it("shows this submission's progress and then its error", () => {
    expect(submissionState(true, Result.initial(true))).toEqual({ pending: true, failure: undefined })
    expect(submissionState(true, failed)).toEqual({ pending: false, failure: { title: "Magnitude can’t open this model", description: expect.any(String), retryable: false } })
    expect(submissionState(true, Result.failure(Cause.die("transport")))).toEqual({ pending: false, failure: { title: "Couldn’t add this model", description: "Try again.", retryable: true } })
  })
})
