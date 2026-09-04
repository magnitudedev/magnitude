import { Either, Schema } from "effect"
import { describe, expect, it } from "vitest"
import fixture from "../fixtures/v1.json"
import { ModelsStatusEnvelopeSchema, ModelsLoadEnvelopeSchema, ModelsStopEnvelopeSchema, jsonFailureEnvelopeSchema, MagnitudeProgressSchema, MagnitudeTimingsSchema } from "../src"

describe("public integration wire contract", () => {
  it("round trips the producer/consumer fixtures", () => {
    const roundTrip = <A, I>(schema: Schema.Schema<A, I>, wire: unknown) => expect(Schema.encodeSync(schema)(Schema.decodeUnknownSync(schema)(wire))).toEqual(wire)
    roundTrip(ModelsStatusEnvelopeSchema, fixture.status)
    roundTrip(ModelsStatusEnvelopeSchema, fixture.initializing)
    roundTrip(ModelsLoadEnvelopeSchema, fixture.load)
    roundTrip(ModelsStopEnvelopeSchema, fixture.stop)
    roundTrip(jsonFailureEnvelopeSchema("models.load"), fixture.failure)
    roundTrip(MagnitudeProgressSchema, fixture.progress)
    roundTrip(MagnitudeTimingsSchema, fixture.timings)
  })
  it("accepts additive fields but rejects incompatible versions, commands, states and malformed counters", () => {
    expect(Schema.decodeUnknownSync(ModelsStatusEnvelopeSchema)({ ...fixture.status, future: true })).toEqual(Schema.decodeUnknownSync(ModelsStatusEnvelopeSchema)(fixture.status))
    for (const wire of [{ ...fixture.status, schemaVersion: 2 }, fixture.load, { ...fixture.status, data: { state: "initializing", models: fixture.status.data.models } }]) {
      expect(Either.isLeft(Schema.decodeUnknownEither(ModelsStatusEnvelopeSchema)(wire))).toBe(true)
    }
    expect(Either.isLeft(Schema.decodeUnknownEither(MagnitudeTimingsSchema)({ ...fixture.timings, predicted_ms: -1 }))).toBe(true)
    expect(Either.isLeft(Schema.decodeUnknownEither(MagnitudeProgressSchema)({ ...fixture.progress, total_tokens: 1.2 }))).toBe(true)
  })
})
