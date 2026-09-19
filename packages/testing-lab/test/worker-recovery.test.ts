import { Effect, Schema } from "effect"
import { expect, test } from "vitest"
import { ApplicationIdentity } from "../src/application-identity"
import { DesktopDriver } from "../src/desktop-driver"
import { AssertionFailure, InfrastructureFailure } from "../src/domain"
import { GenerationExecution } from "../src/generation-evidence"
import { CliTests } from "../src/suites/cli"
import { verifyWorkerRecovery } from "../src/suites/worker-recovery"
import { WorkerFault, WorkerFaultReceipt } from "../src/worker-fault"

// Any unexpected operation is a defect, including stop/restart attempts that would hide recovery failures.
const strict = <A extends object>(value: Partial<A>): A => new Proxy(value, {
  get(target, key) { if (!(key in target)) throw new Error(`Unexpected operation ${String(key)}`); return Reflect.get(target, key) },
}) as A
const owner = Schema.decodeUnknownSync(ApplicationIdentity)({ applicationPid: 10, servicePid: 11, serviceInstance: "original" })
const receipt = Schema.decodeUnknownSync(WorkerFaultReceipt)({ workerPid: 42, parentPid: 20, parentStart: "123", executable: "/profile/releases/runtime/magnitude-inference", terminated: true })
const generation = (workerGeneration: string) => Schema.decodeUnknownSync(GenerationExecution)({
  generation: { requestId: "completion", model: "fixture", text: "HELLO", chunks: 1 },
  native: { traceId: "a".repeat(32), model: "fixture", workerPid: 42, workerGeneration, requestId: "7", allocations: [{ kind: "host", model_bytes: 1024 }] },
})

test.each(["healthy", "before-attestation", "crash", "owner-after-crash", "load", "same-generation", "after-attestation", "parent", "owner-after-recovery"])("worker recovery: %s", async mode => {
  const events: string[] = [], evidence: string[] = []
  let observations = 0, identities = 0, attestations = 0
  const step = (name: string) => Effect.suspend(() => {
    events.push(name)
    return mode === name ? Effect.fail(new InfrastructureFailure({ operation: name, message: "Injected failure" })) : Effect.void
  })
  const result = await Effect.runPromise(verifyWorkerRecovery("/profile", Effect.sync(() => generation(++observations === 1 || mode === "same-generation" ? "1" : "2")),
    () => Effect.suspend(() => {
      const stage = ++attestations === 1 ? "before-attestation" : "after-attestation"
      events.push(stage)
      return mode === stage ? Effect.fail(new AssertionFailure({ message: stage })) : Effect.void
    }), item => Effect.sync(() => { evidence.push(item._tag) })).pipe(
    Effect.provideService(DesktopDriver, strict<DesktopDriver>({ ready: () => Effect.void, identity: () => Effect.sync(() => {
      identities++
      return (mode === "owner-after-crash" && identities === 2) || (mode === "owner-after-recovery" && identities === 3)
        ? { ...owner, applicationPid: ApplicationIdentity.fields.applicationPid.make(99) } : owner
    }) })),
    Effect.provideService(CliTests, strict<CliTests>({ failedModel: step("failed"), loadModel: step("load") })),
    Effect.provideService(WorkerFault, { crash: request => {
      expect(request).toEqual({ owner, workerPid: 42, profile: "/profile" })
      return step("crash").pipe(Effect.as(receipt))
    }, verifyParent: actual => { expect(actual).toEqual(receipt); return step("parent") } }), Effect.either))
  expect(result._tag).toBe(mode === "healthy" ? "Right" : "Left")
  expect(evidence[0]).toBe("Before")
  if (mode === "before-attestation") expect(events).toEqual(["before-attestation"])
  if (mode === "crash") expect(events).toEqual(["before-attestation", "crash"])
  if (mode === "owner-after-crash") expect(events).not.toContain("load")
  if (mode === "load") expect(observations).toBe(1)
  if (mode === "same-generation" || mode === "after-attestation") expect(events).not.toContain("parent")
  if (mode === "healthy") {
    expect(events).toEqual(["before-attestation", "crash", "failed", "load", "after-attestation", "parent"])
    expect(evidence).toEqual(["Before", "Fault", "After", "Owners"])
    expect(observations).toBe(2) // Numeric PID reuse is allowed; the native generation must change.
  }
})
