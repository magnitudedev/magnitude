import { Effect, Layer, Schema } from "effect"
import { expect, test } from "vitest"
import { ApplicationIdentity } from "../src/application-identity"
import { DisposableDesktopUser } from "../src/desktop-environment"
import { DesktopDriver } from "../src/desktop-driver"
import { DownloadProgress } from "../src/download-controls"
import { AssertionFailure, InfrastructureFailure } from "../src/domain"
import { GenerationExecution } from "../src/generation-evidence"
import { ModelFileReceipt } from "../src/model-files"
import { NetworkFault, NetworkIsolation, NetworkProbeAddress } from "../src/network-fault"
import { CliTests } from "../src/suites/cli"
import { verifyDownloadRecovery } from "../src/suites/download-recovery"

const strict = <A extends object>(value: Partial<A>): A => new Proxy(value, { get(target, key) {
  if (!(key in target)) throw new Error(`Unexpected operation ${String(key)}`)
  return Reflect.get(target, key)
} }) as A
const owner = Schema.decodeUnknownSync(ApplicationIdentity)({ applicationPid: 10, servicePid: 11, serviceInstance: "original" })
const address = NetworkProbeAddress.make({ address: "192.0.2.1", family: 4, port: 443 })
const isolation = Schema.decodeUnknownSync(NetworkIsolation)({ uid: 1000, rule: `magnitude_lab_${"a".repeat(32)}` })
const files = Schema.decodeUnknownSync(ModelFileReceipt)({ files: [{ repository: "owner/model", revision: "a".repeat(40), path: "model.gguf", bytes: 100, sha256: "b".repeat(64) }] })
const generation = Schema.decodeUnknownSync(GenerationExecution)({ generation: { requestId: "completion", model: "fixture", text: "HELLO", chunks: 1 },
  native: { traceId: "a".repeat(32), model: "fixture", workerPid: 42, workerGeneration: "2", requestId: "7", allocations: [{ kind: "host", model_bytes: 1024 }] } })

test.each(["healthy", "no-guest", "preflight", "baseline", "transferring", "fault", "failure-ui", "restoration", "retry", "bytes", "generation", "attest", "owner"])("interrupted download recovery preserves real preconditions: %s", async mode => {
  const events: string[] = [], evidence: string[] = []
  const cleanup: string[] = []
  let isolated = false, cuts = 0, fileReads = 0, starts = 0, identities = 0
  const step = (name: string) => Effect.suspend(() => {
    events.push(name)
    return mode === name ? Effect.fail(new AssertionFailure({ message: name })) : Effect.void
  })
  const result = await Effect.runPromise(verifyDownloadRecovery("fixture", Effect.suspend(() => step(++fileReads === 1 ? "baseline" : "bytes").pipe(Effect.as(files))),
    step("generation").pipe(Effect.as(generation)), () => step("attest"), value => Effect.sync(() => { evidence.push(value._tag) }), message => { cleanup.push(message) }).pipe(
    Effect.provideService(NetworkFault, { resolve: Effect.succeed(address), reachable: () => Effect.sync(() => !isolated && !(mode === "restoration" && cuts === 2)),
      isolate: () => Effect.acquireRelease(Effect.suspend(() => {
        cuts++
        if (mode === "preflight" || (mode === "fault" && cuts === 2)) return Effect.fail(new InfrastructureFailure({ operation: "fault", message: "Fault setup failed" }))
        isolated = true; events.push("cut"); return Effect.succeed(isolation)
      }), () => Effect.sync(() => { isolated = false; events.push("restore") })) }),
    Effect.provideService(CliTests, strict<CliTests>({ removeModel: step("remove") })),
    Effect.provideService(DesktopDriver, strict<DesktopDriver>({ identity: () => Effect.sync(() => ++identities === 2 && mode === "owner"
      ? { ...owner, applicationPid: ApplicationIdentity.fields.applicationPid.make(99) } : owner), search: () => step("search"), load: () => step("load"),
      downloads: { absent: () => step("absent"), begin: () => Effect.suspend(() => step(++starts === 1 ? "begin" : "retry")),
        transferring: () => step("transferring").pipe(Effect.as(DownloadProgress.make({ completedBytes: 10, totalBytes: 100 }))),
        failed: () => Effect.suspend(() => { expect(isolated).toBe(true); return step("failure-ui") }), complete: () => step("complete") } })),
    Effect.provide(mode === "no-guest" ? Layer.empty : Layer.succeed(DisposableDesktopUser, { home: "/disposable" })), Effect.either))
  expect(result._tag).toBe(mode === "healthy" ? "Right" : "Left")
  expect(isolated).toBe(false)
  expect(cleanup.length).toBe(mode === "restoration" ? 1 : 0)
  if (["no-guest", "preflight", "baseline"].includes(mode)) expect(events).not.toContain("remove")
  if (["transferring", "fault", "failure-ui", "restoration"].includes(mode)) expect(events).not.toContain("retry")
  if (mode === "bytes") expect(events).not.toContain("generation")
  if (mode === "attest") expect(evidence).toContain("Recovered")
  if (mode === "failure-ui") expect(evidence).toContain("Restored")
  if (mode === "healthy") {
    expect(events).toEqual(["cut", "restore", "baseline", "remove", "search", "absent", "begin", "transferring", "cut", "failure-ui", "restore", "retry", "complete", "bytes", "load", "generation", "attest"])
    expect(evidence).toEqual(["Baseline", "Transferring", "Isolation", "Interrupted", "Restored", "Recovered", "Owners"])
    expect(fileReads).toBe(2)
  }
})
