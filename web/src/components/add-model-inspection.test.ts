import { describe, expect, it } from "vitest"
import { Registry, Result } from "@effect-atom/atom-react"
import { Cause, Data, Effect, Layer, Option } from "effect"
import { Client, Query } from "@magnitudedev/effect-query"
import { InvalidModelReference } from "@magnitudedev/sdk"
import { inspectionView, submissionState } from "./add-model-dialog"
import { inspection } from "../../test/models/fixtures"

const failure = new InvalidModelReference({ reference: "owner/repo", message: "PRIVATE" })

describe("inspection after an error", () => {
  it("shows the error, then reading while Retry runs, then the inspection", () => {
    expect(inspectionView(Result.failure(Cause.fail(failure)))).toEqual({ _tag: "Failed", error: Option.some(failure) })
    expect(inspectionView(Result.failure(Cause.fail(failure), { waiting: true }))).toEqual({ _tag: "Reading" })
    expect(inspectionView(Result.initial(true))).toEqual({ _tag: "Reading" })
    expect(inspectionView(Result.success(inspection()))).toEqual({ _tag: "Inspected", inspection: inspection() })
  })
  it("re-runs a failed inspection on Retry and shows the result once it succeeds", async () => {
    const registry = Registry.make()
    const effectQuery = Client.make(Layer.empty)
    let attempts = 0
    const inspect = Query.make("InspectForRetry", {
      key: ({ reference }: { readonly reference: string }) => Data.struct({ reference }),
      effect: () => Effect.suspend(() => ++attempts === 1 ? Effect.fail(failure) : Effect.succeed(inspection())),
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

describe("adding after an earlier failure", () => {
  const failed = Result.failure<unknown, unknown>(Cause.fail(failure))
  it("ignores another opening's pending or failed add until this one submits", () => {
    expect(submissionState(false, failed)).toEqual({ pending: false, failure: undefined })
    expect(submissionState(false, Result.initial(true))).toEqual({ pending: false, failure: undefined })
  })
  it("shows this submission's progress and then its error", () => {
    expect(submissionState(true, Result.initial(true))).toEqual({ pending: true, failure: undefined })
    expect(submissionState(true, failed)).toEqual({ pending: false, failure: { title: "Magnitude can’t open this model", description: expect.any(String), retryable: false } })
    expect(submissionState(true, Result.failure(Cause.die("transport")))).toEqual({ pending: false, failure: { title: "Couldn’t add this model", description: "Try again." } })
  })
})
