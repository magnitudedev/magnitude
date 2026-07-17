import { Effect, Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import {
  LocalModelIndexSchema,
  emptyLocalModelIndex,
  makeLocalModelIndexStore,
} from "./local-model-index"
import { LlamaFitAssessmentKey, NormalizedLlamaModelPath } from "./llamacpp"

describe("LocalModelIndexStore", () => {
  it("serializes independent artifact and discovered-property updates", async () => {
    const persisted: Array<typeof LocalModelIndexSchema.Type> = []
    const store = await Effect.runPromise(makeLocalModelIndexStore({
      initialIndex: Option.some(emptyLocalModelIndex()),
      persist: (index) => Effect.sync(() => { persisted.push(index) }),
    }))
    const capturedAt = new Date("2026-07-16T00:00:00.000Z")

    await Effect.runPromise(store.replaceArtifacts({ capturedAt, sets: [], issues: [] }))
    await Effect.runPromise(store.putDiscoveredProperties({
      modelPath: "/models/a.gguf",
      visionInspections: [{ routeId: "managed:a", fingerprint: "fingerprint-a", value: true }],
      reasoningInspections: [],
    }))
    await Effect.runPromise(store.putDiscoveredProperties({
      modelPath: "/models/a.gguf",
      visionInspections: [{ routeId: "managed:a", fingerprint: "fingerprint-a", value: true }],
      reasoningInspections: [],
    }))

    const snapshot = await Effect.runPromise(store.snapshot)
    expect(snapshot.artifacts.capturedAt).toEqual(capturedAt)
    expect(snapshot.discoveredProperties).toHaveLength(1)
    expect(snapshot.fitAssessments).toHaveLength(0)
    expect(persisted).toHaveLength(2)
  })

  it("uses the current index shape", () => {
    const index = emptyLocalModelIndex()

    expect(index).toEqual({
      artifacts: {
        capturedAt: expect.any(Date),
        sets: [],
        issues: [],
      },
      discoveredProperties: [],
      fitAssessments: [],
    })
    expect(Schema.is(LocalModelIndexSchema)(index)).toBe(true)
  })

  it("rejects invalid cache contents", () => {
    const decode = Schema.decodeUnknownOption(LocalModelIndexSchema)

    expect(Option.isNone(decode({
      capturedAt: "2026-07-16T00:00:00.000Z",
      sets: [],
      issues: [],
      discoveredProperties: [],
      fitAssessments: [],
    }))).toBe(true)
    expect(Option.isNone(decode({
      artifacts: {
        capturedAt: "2026-07-16T00:00:00.000Z",
        sets: [],
        issues: [],
      },
      discoveredProperties: [{ invalid: true }],
      fitAssessments: [],
    }))).toBe(true)
  })

  it("serializes fit-assessment updates independently", async () => {
    const store = await Effect.runPromise(makeLocalModelIndexStore({
      initialIndex: Option.some(emptyLocalModelIndex()),
      persist: () => Effect.void,
    }))
    const modelPath = NormalizedLlamaModelPath.make("/models/a.gguf")
    const key = LlamaFitAssessmentKey.make("fit-key")
    const assessment = {
      estimatedTotalBytes: 12,
      domains: [{ memoryDomainId: "system", estimatedBytes: 12, stableCapacityBytes: 10, marginBytes: -2 }],
      result: "capacity_risk" as const,
    }

    await Effect.runPromise(store.putFitAssessment({ modelPath, key, assessment }))

    const found = await Effect.runPromise(store.fitAssessment(modelPath, key))
    expect(found).toEqual(Option.some({ modelPath, key, assessment }))
    expect((await Effect.runPromise(store.snapshot)).artifacts.sets).toEqual([])
  })
})
