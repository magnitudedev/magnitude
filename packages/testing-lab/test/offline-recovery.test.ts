import { Effect, Schema } from "effect"
import { expect, test } from "vitest"
import { ApplicationIdentity } from "../src/application-identity"
import { DesktopDriver } from "../src/desktop-driver"
import { AssertionFailure } from "../src/domain"
import { GenerationExecution } from "../src/generation-evidence"
import { NetworkFault, NetworkIsolation, NetworkProbeAddress } from "../src/network-fault"
import { CliTests } from "../src/suites/cli"
import { verifyOfflineRecovery } from "../src/suites/offline-recovery"

const strict = <A extends object>(value: Partial<A>): A => new Proxy(value, { get(target, key) {
  if (!(key in target)) throw new Error(`Unexpected operation ${String(key)}`)
  return Reflect.get(target, key)
} }) as A
const owner = Schema.decodeUnknownSync(ApplicationIdentity)({ applicationPid: 10, servicePid: 11, serviceInstance: "original" })
const address = NetworkProbeAddress.make({ address: "192.0.2.1", family: 4, port: 443 })
const isolation = Schema.decodeUnknownSync(NetworkIsolation)({ _tag: "linux", uid: 1000, controlPlane: [], resolvers: [], rule: `magnitude_lab_${"a".repeat(32)}` })
const generation = (workerGeneration: string) => Schema.decodeUnknownSync(GenerationExecution)({
  generation: { requestId: "completion", model: "fixture", text: "HELLO", chunks: 1 },
  native: { traceId: "a".repeat(32), model: "fixture", workerPid: 42, workerGeneration, requestId: "7", allocations: [{ kind: "host", model_bytes: 1024 }] },
})

test.each(["healthy", "baseline-offline", "unblocked", "reload", "offline-attestation", "lost-isolation", "restore", "owner", "same-worker"])("offline recovery preserves restoration and evidence: %s", async mode => {
  const events: string[] = [], evidence: string[] = [], cleanup: string[] = []
  let isolated = false, removed = false, observations = 0, checks = 0, identities = 0
  const fail = () => Effect.fail(new AssertionFailure({ message: "Injected failure" }))
  const result = await Effect.runPromise(verifyOfflineRecovery(Effect.sync(() => {
    observations++
    if (observations === 2) expect(isolated).toBe(true)
    return generation(observations === 1 || mode === "same-worker" ? "1" : "2")
  }), () => observations === 2 && mode === "offline-attestation" ? fail() : Effect.void,
  item => Effect.sync(() => { evidence.push(item._tag) }), message => { cleanup.push(message) }).pipe(
    Effect.provideService(NetworkFault, { resolve: Effect.succeed(address), reachable: actual => Effect.sync(() => {
      expect(actual).toEqual(address)
      checks++
      events.push(`probe:${isolated ? "offline" : "online"}`)
      return mode === "baseline-offline" ? false : !isolated || mode === "unblocked" || (mode === "lost-isolation" && checks === 3)
    }), isolate: onCleanupError => Effect.acquireRelease(Effect.sync(() => { isolated = true; events.push("cut"); return isolation }), () => Effect.sync(() => {
      events.push("restore"); removed = true
      if (mode === "restore") onCleanupError("Remove failed")
      else isolated = false
    })) }),
    Effect.provideService(DesktopDriver, strict<DesktopDriver>({ ready: () => Effect.void, identity: () => Effect.sync(() => {
      return ++identities > 1 && mode === "owner" ? { ...owner, applicationPid: ApplicationIdentity.fields.applicationPid.make(99) } : owner
    }) })),
    Effect.provideService(CliTests, strict<CliTests>({ reloadModel: Effect.suspend(() => { events.push("reload"); expect(isolated).toBe(true); return mode === "reload" ? fail() : Effect.void }) })),
    Effect.either))
  expect(result._tag).toBe(mode === "healthy" ? "Right" : "Left")
  if (mode === "baseline-offline") expect(events).toEqual(["probe:online"])
  else {
    expect(removed).toBe(true)
    expect(events.at(-1)).toBe(mode === "restore" ? "probe:offline" : "probe:online")
    if (mode !== "restore") expect(evidence).toContain("Restored")
  }
  if (mode === "unblocked") {
    expect(events).not.toContain("reload")
    expect(evidence).toContain("Isolation")
  }
  if (mode === "healthy") {
    expect(evidence).toEqual(["Before", "Isolation", "Offline", "Restored", "Owners"])
    expect(events).toEqual(["probe:online", "cut", "probe:offline", "reload", "probe:offline", "restore", "probe:online"])
  }
  expect(cleanup.length > 0).toBe(mode === "restore")
})
