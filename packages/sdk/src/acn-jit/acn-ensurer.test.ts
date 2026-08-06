import { AcnInstanceIdSchema, AcnReady, ProcessStartIdentitySchema } from "@magnitudedev/acn-protocol"
import { Effect, Exit, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { runAcnEnsure, type ReadyAcn } from "./acn-ensurer"
import { SDK_VERSION } from "../version"

const ready: ReadyAcn = {
  id: AcnInstanceIdSchema.make("test-acn"),
  identity: SDK_VERSION,
  url: "http://test-acn",
  pid: 1,
  processStartIdentity: ProcessStartIdentitySchema.make("test-process"),
  lifecycle: new AcnReady({}),
}

describe("AcnEnsurer contract", () => {
  it("returns one terminal readiness", async () => {
    expect(await Effect.runPromise(runAcnEnsure(Stream.succeed({
      _tag: "Ready",
      instance: ready,
    })))).toEqual(ready)
  })

  it("rejects missing, duplicate, and post-ready progress", async () => {
    const missing = await Effect.runPromise(Effect.exit(runAcnEnsure(Stream.empty)))
    const duplicate = await Effect.runPromise(Effect.exit(runAcnEnsure(Stream.fromIterable([
      { _tag: "Ready" as const, instance: ready },
      { _tag: "Ready" as const, instance: ready },
    ]))))
    const after = await Effect.runPromise(Effect.exit(runAcnEnsure(Stream.fromIterable([
      { _tag: "Ready" as const, instance: ready },
      { _tag: "Observation" as const, observation: { _tag: "Starting" as const, phase: "Discovering" as const } },
    ]))))
    expect(Exit.isFailure(missing)).toBe(true)
    expect(Exit.isFailure(duplicate)).toBe(true)
    expect(Exit.isFailure(after)).toBe(true)
  })
})
