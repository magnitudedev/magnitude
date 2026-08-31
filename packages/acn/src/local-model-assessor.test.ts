import { describe, expect, it } from "vitest"
import { Effect, Stream } from "effect"
import { ModelIdSchema } from "@magnitudedev/acn-protocol"
import type { AssessModelsEvent, CatalogModel } from "@magnitudedev/icn-protocol/schemas"
import {
  catalogAssessmentDemands,
  consumeAssessmentEvents,
  type AssessmentExpectation,
  type CoordinatedLocalModelAssessment,
} from "./local-model-assessor"
import type { LocalModelSourcesState } from "./local-model-sources"

const ready = {
  profile: { contextLength: 32_768 },
  metadata: {},
  capabilities: {},
} as never

const state = (localState: CatalogModel["localState"]): LocalModelSourcesState => ({
  catalogRevision: 7,
  discoveryRevision: 3,
  reconciliationComplete: true,
  catalogModels: [{
    id: ModelIdSchema.make("model:gguf:q4") as never,
    source: { desired: ready, localState } as unknown as Omit<CatalogModel, "id">,
  }],
  discoveredModels: [],
})

describe("catalog assessment demand", () => {
  it("assesses desired material for an uninstalled catalog model", () => {
    expect(catalogAssessmentDemands(state({ _tag: "NotInstalled" }))).toMatchObject([
      { selection: "Desired", requestId: "catalog-7-0" },
    ])
  })

  it("assesses the effective installed material rather than desired update material", () => {
    expect(catalogAssessmentDemands(state({
      _tag: "Installed",
      effective: { _tag: "Ready", model: ready },
      installation: {} as never,
      updateState: { _tag: "Available", requiredDownloadBytes: 1 },
    }))).toMatchObject([{ selection: "Effective" }])
  })

  it("does not assess an installed model that ICN says is unavailable", () => {
    expect(catalogAssessmentDemands(state({
      _tag: "Installed",
      effective: { _tag: "Unavailable", failure: {
        code: "invalid_artifact", message: "Invalid model", retryable: false,
      } },
      installation: {} as never,
      updateState: { _tag: "Current" },
    }))).toEqual([])
  })
})

const assessmentExpectation = (): AssessmentExpectation => ({
  modelId: ModelIdSchema.make("model:gguf:q4"),
  subject: { _tag: "Catalog", modelId: "model:gguf:q4", selection: "Desired" },
  profiles: [{
    profile: { contextLength: 32_768 },
    performanceContextTokens: [25_000, 32_768],
  }],
})

const failedResult = (): AssessModelsEvent => ({
  _tag: "Result",
  result: {
    _tag: "Failed",
    requestId: "request-1",
    subject: { _tag: "Catalog", modelId: "model:gguf:q4", selection: "Desired" },
    failure: { code: "native_failure", message: "native failure", retryable: false },
  },
})

const assessmentEvents = (middle: readonly AssessModelsEvent[], completed = true): readonly AssessModelsEvent[] => [
  { _tag: "Started", revision: 7, environmentId: "environment-1", totalTargets: 1 },
  ...middle,
  ...(completed
    ? [{ _tag: "Completed", revision: 7, environmentId: "environment-1", totalTargets: 1 } as const]
    : []),
]

describe("assessment stream correlation", () => {
  const run = async (events: readonly AssessModelsEvent[]) => {
    const published = new Map<string, CoordinatedLocalModelAssessment>()
    await Effect.runPromise(consumeAssessmentEvents(
      { events: Stream.fromIterable(events) },
      7,
      new Map([["request-1", assessmentExpectation()]]),
      (modelId, value) => Effect.sync(() => { published.set(modelId, value) }),
    ))
    return published.get("model:gguf:q4")
  }

  it("accepts exactly correlated, explicitly completed results", async () => {
    expect(await run(assessmentEvents([failedResult()]))).toEqual({
      _tag: "Failed",
      failure: { code: "native_failure", message: "native failure", retryable: false },
    })
  })

  it("rejects duplicate results instead of trusting the first one", async () => {
    expect(await run(assessmentEvents([failedResult(), failedResult()]))).toMatchObject({
      _tag: "Failed",
      failure: { code: "invalid_assessment_response" },
    })
  })

  it("keeps a correlated result when the stream omits its completion event", async () => {
    expect(await run(assessmentEvents([failedResult()], false))).toMatchObject({
      _tag: "Failed",
      failure: { code: "native_failure" },
    })
  })

  it("rejects an echoed subject that does not match the request", async () => {
    const event = failedResult()
    if (event._tag !== "Result") throw new Error("test fixture must be a result")
    expect(await run(assessmentEvents([{ ...event, result: {
      ...event.result,
      subject: { _tag: "Catalog", modelId: "other:gguf:q4", selection: "Desired" },
    } }]))).toMatchObject({
      _tag: "Failed",
      failure: { code: "invalid_assessment_response" },
    })
  })

  it("fails pending assessments when the ICN result stream stops making progress", async () => {
    const published = new Map<string, CoordinatedLocalModelAssessment>()
    await Effect.runPromise(consumeAssessmentEvents(
      { events: Stream.never },
      7,
      new Map([["request-1", assessmentExpectation()]]),
      (modelId, value) => Effect.sync(() => { published.set(modelId, value) }),
      "10 millis",
    ))
    expect(published.get("model:gguf:q4")).toMatchObject({
      _tag: "Failed",
      failure: { code: "assessment_stream_failed", retryable: true },
    })
  })
})
